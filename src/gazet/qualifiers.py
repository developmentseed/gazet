"""Qualified place queries: "Paris, Texas", "Raleigh, NC", "Mount Etna, Italy".

Fuzzy matching scores the whole query against bare names, so a trailing
qualifier only adds noise: "Paris, Texas" matched Panama's Parita District.
:func:`parse_qualified_query` splits the name from its qualifiers and resolves
the qualifiers to an admin area, which the search then restricts itself to.

:func:`token_score` is the check that rejects near-misses Jaro-Winkler still
scores highly ("Mount Etna" -> "Mount Eba" at 0.94): every word of the query
must closely match some word of the candidate's name.
"""

import re
import unicodedata
from dataclasses import dataclass

import duckdb

#: Two-letter postal codes, as users write them after a US place ("Fremont, CA").
#: Checked before country codes, so "CA" is California, not Canada.
US_STATE_CODES = frozenset(
    "AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS "
    "MO MT NE NV NH NJ NM NY NC ND OH OK OR PA PR RI SC SD TN TX UT VT VA WA WV "
    "WI WY".split()
)

#: Country names users write that match no Overture name exactly.
COUNTRY_ALIASES = {
    "usa": "US",
    "us": "US",
    "u.s.": "US",
    "u.s.a.": "US",
    "united states": "US",
    "united states of america": "US",
    "america": "US",
    "uk": "GB",
    "u.k.": "GB",
    "united kingdom": "GB",
    "great britain": "GB",
    "britain": "GB",
}

#: Leading words that describe part of a place rather than name it
#: ("Southern Brazil"). Used only when the full query matches nothing.
DIRECTIONAL_WORDS = frozenset(
    "north south east west northern southern eastern western central "
    "northeast northwest southeast southwest northeastern northwestern "
    "southeastern southwestern".split()
)


#: Feature-type and filler words a source may leave out of a name.
GENERIC_WORDS = frozenset(
    "lake lakes river mount mountain mountains mt valley desert basin plateau "
    "bay gulf sea ocean island islands cape falls national park forest "
    "the of de del la le".split()
)


@dataclass(frozen=True)
class AreaFilter:
    """An admin area a search is restricted to."""

    label: str
    id: str
    country: str
    region: str | None
    bbox: tuple[float, float, float, float]  # minx, miny, maxx, maxy


def _lookup_area(
    con: duckdb.DuckDBPyConnection, where: str, params: list, label: str
) -> AreaFilter | None:
    # Read at call time, not import time, so the path can be redirected.
    from . import search

    row = con.execute(
        f"""
        SELECT id, country, CASE WHEN subtype = 'country' THEN NULL ELSE region END,
               ST_XMin(geometry), ST_YMin(geometry), ST_XMax(geometry), ST_YMax(geometry)
        FROM read_parquet(?)
        WHERE subtype IN ('country', 'dependency', 'region') AND ({where})
        ORDER BY admin_level ASC NULLS LAST
        LIMIT 1
        """,
        [search.DIVISIONS_AREA_PATH, *params],
    ).fetchone()
    if row is None or row[1] is None:
        return None
    return AreaFilter(
        label=label, id=row[0], country=row[1], region=row[2], bbox=tuple(row[3:7])
    )


def resolve_qualifier(con: duckdb.DuckDBPyConnection, text: str) -> AreaFilter | None:
    """Resolve a qualifier ("Texas", "NC", "Italy", "USA") to an admin area."""
    label = text.strip()
    key = label.lower()
    if key in COUNTRY_ALIASES:
        return _lookup_area(
            con, "subtype = 'country' AND country = ?", [COUNTRY_ALIASES[key]], label
        )
    if len(label) == 2 and label.upper() in US_STATE_CODES:
        return _lookup_area(
            con, "subtype = 'region' AND region = ?", [f"US-{label.upper()}"], label
        )
    return _lookup_area(
        con,
        "lower(names.primary) = ? OR lower(names.common.en) = ?",
        [key, key],
        label,
    )


def parse_qualified_query(
    con: duckdb.DuckDBPyConnection, q: str
) -> tuple[str, AreaFilter | None]:
    """Split "Name, Qualifier, ..." into the name and the area it names.

    The most specific resolved qualifier wins ("Texas" over "United States").
    A query whose qualifiers resolve to nothing is returned whole, so a name
    that itself contains a comma still matches as before.
    """
    parts = [p.strip() for p in q.split(",") if p.strip()]
    if len(parts) < 2:
        return q, None
    areas = [a for a in (resolve_qualifier(con, p) for p in parts[1:]) if a]
    if not areas:
        return q, None
    return parts[0], max(areas, key=lambda a: a.region is not None)


def strip_directional(name: str) -> str | None:
    """"Southern Brazil" -> "Brazil"; None when there is no leading direction."""
    words = name.split()
    if len(words) > 1 and words[0].lower() in DIRECTIONAL_WORDS:
        return " ".join(words[1:])
    return None


def _tokens(text: str) -> list[str]:
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.findall(r"[a-z0-9]+", folded.lower())


def jaro_winkler(a: str, b: str) -> float:
    """Jaro-Winkler similarity, matching DuckDB's ``jaro_winkler_similarity``."""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    window = max(max(len(a), len(b)) // 2 - 1, 0)
    a_hit = [False] * len(a)
    b_hit = [False] * len(b)
    matches = 0
    for i, ch in enumerate(a):
        for j in range(max(0, i - window), min(len(b), i + window + 1)):
            if not b_hit[j] and b[j] == ch:
                a_hit[i] = b_hit[j] = True
                matches += 1
                break
    if matches == 0:
        return 0.0
    a_seq = [ch for ch, hit in zip(a, a_hit) if hit]
    b_seq = [ch for ch, hit in zip(b, b_hit) if hit]
    transpositions = sum(x != y for x, y in zip(a_seq, b_seq)) / 2
    jaro = (
        matches / len(a) + matches / len(b) + (matches - transpositions) / matches
    ) / 3
    prefix = 0
    for x, y in zip(a[:4], b[:4]):
        if x != y:
            break
        prefix += 1
    return jaro + prefix * 0.1 * (1 - jaro)


def token_score(query: str, *names: str | None, drop_generic: bool = False) -> float:
    """The worst-matched query word's best Jaro-Winkler against any name word.

    Accents are folded first, so "Sao Paulo" matches "São Paulo" word for word.
    With ``drop_generic``, generic feature words in the query need not match:
    Natural Earth leaves them out ("Titicaca" for Lake Titicaca). Admin names
    keep them, and there they tell places apart ("Gulf of Mexico" is not the
    State of Mexico), so they are only dropped for natural features.
    """
    q_tokens = _tokens(query)
    if drop_generic:
        q_tokens = [t for t in q_tokens if t not in GENERIC_WORDS] or q_tokens
    n_tokens = [t for n in names if n for t in _tokens(n)]
    if not q_tokens or not n_tokens:
        return 0.0
    return min(max(jaro_winkler(q, n) for n in n_tokens) for q in q_tokens)
