"""
Generate synthetic training samples for text-to-SQL task.

This script:
1. Loads relation tables and entity inventories
2. For each SQL template, samples valid anchors
3. Renders and executes SQL to verify it works
4. Builds candidate lists with controlled distractors
5. Generates natural language questions using LLM
6. Saves complete training samples

Output:
- output/samples/sample_*.json (individual samples)
- output/dataset_raw.jsonl (all samples)
"""

import json
import random
import re
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import duckdb
import pandas as pd
from pydantic import BaseModel

# Suppress warnings
warnings.filterwarnings('ignore')

from gazet.config import DIVISIONS_AREA_PATH, NATURAL_EARTH_PATH

# Fixed paths embedded in every training SQL string.
# The model learns these short, stable strings rather than machine-specific
# local paths.  At inference, sql.py's _rewrite_data_paths substitutes them
# with the actual runtime paths from gazet.config.
_DIVISIONS_SQL_PATH = 'divisions_area'
_NATURAL_EARTH_SQL_PATH = 'natural_earth'


def _for_execution(sql: str) -> str:
    """Replace symbolic placeholder paths with actual local paths for verification."""
    return (
        sql
        .replace("read_parquet('divisions_area')", f"read_parquet('{DIVISIONS_AREA_PATH}')")
        .replace("read_parquet('natural_earth')", f"read_parquet('{NATURAL_EARTH_PATH}')")
    )

# Configurable parameters (can be overridden by CLI)
TARGET_COUNTS = None  # Will be set in main() or by CLI
MAX_WORKERS = 8
RETRY_MULTIPLIER = 2
APPEND_MODE = False


_GENERIC_SURFACE_RULES = [
    ("spelling_neighboring", r"\bneighbouring\b", ["neighboring"]),
    ("spelling_neighbors", r"\bneighbours\b", ["neighbors"]),
    ("expand_whats", r"\bwhat's\b", ["what is"]),
    ("show_me", r"\bshow me\b", ["show", "display"]),
    ("give_me", r"\bgive me\b", ["show", "list"]),
    ("pull_up", r"\bpull up\b", ["show", "display"]),
    ("find_to_show", r"\bfind\b", ["show", "locate"]),
    ("kilometers_variant", r"\bkilometers\b", ["km"]),
    ("metres_variant", r"\bmetres\b", ["meters"]),
    ("recognised_variant", r"\brecognised\b", ["recognized"]),
]

_FAMILY_SURFACE_RULES = {
    "adjacency": [
        ("which_border_to_next_to", r"\bwhich (.+?) border (.+)\?", [r"which \1 are next to \2?", r"which \1 are adjacent to \2?"]),
        ("bordering_to_next_to", r"\bbordering (.+)", [r"next to \1", r"adjacent to \1"]),
        ("touching_to_next_to", r"\btouching (.+)", [r"next to \1"]),
        ("share_border_to_adjacent", r"share a border with", ["are adjacent to", "are next to"]),
        ("adjacent_to_next_to", r"adjacent to", ["next to"]),
    ],
    "multi_adjacency": [
        ("which_border_both_to_next_to", r"\bwhich (.+?) border both (.+)\?", [r"which \1 are next to both \2?", r"which \1 are adjacent to both \2?"]),
        ("touch_both_to_next_to", r"touch both", ["are next to both"]),
        ("adjacent_both_to_next_to", r"adjacent to both", ["next to both"]),
    ],
    "containment": [
        ("within_to_inside", r"\bwithin\b", ["inside", "in"]),
        ("inside_to_in", r"\binside\b", ["in"]),
        ("belonging_to_in", r"belonging to", ["in"]),
        ("contain_to_have", r"\bcontain\b", ["have"]),
    ],
    "intersection": [
        ("which_intersect_to_overlap", r"\bwhich (.+?) intersect (.+)\?", [r"which \1 overlap \2?"]),
        ("overlap_with_to_intersect", r"overlap with", ["intersect"]),
        ("crossing_to_overlapping", r"crossing into", ["overlapping"]),
        ("partly_in_to_overlap", r"partly in", ["overlapping"]),
    ],
    "buffer": [
        ("within_distance_to_from", r"within ([0-9]+\s*(?:km|m)) of", [r"up to \1 from", r"at a distance of \1 from"]),
        ("buffer_to_radius", r"\bbuffer\b", ["radius", "zone"]),
        ("close_to_near", r"close to", ["near"]),
        ("around_to_near", r"what is around", ["what is near"]),
    ],
    "chained": [
        ("coastal_to_seaside", r"\bcoastal\b", ["seaside", "maritime"]),
        ("landlocked_to_inland", r"\blandlocked\b", ["inland"]),
        ("sea_access_to_coast", r"sea access", ["a coastline"]),
    ],
    "difference": [
        ("part_to_portion", r"\bpart of\b", ["portion of", "section of"]),
        ("outside_to_excluding", r"\boutside\b", ["excluding"]),
    ],
    "border_corridor": [
        ("zone_to_buffer", r"\bzone\b", ["buffer", "corridor"]),
        ("within_distance_to_along", r"within ([0-9]+ km) of the", [r"along the", r"up to \1 from the"]),
    ],
    "set_operations": [
        ("combined_to_merged", r"combined", ["merged"]),
        ("union_of_to_merged_area", r"\bunion of\b", ["merged area of", "combined area of"]),
        ("merge_to_combine", r"\bmerge\b", ["combine"]),
        ("together_to_combined", r"\btogether\b", ["combined"]),
    ],
    "partial_selection": [
        ("part_to_portion", r"\bpart of\b", ["portion of", "section of"]),
        ("half_to_side", r"\bhalf\b", ["side"]),
    ],
    "aggregation": [
        ("largest_to_biggest", r"\blargest\b", ["biggest"]),
        ("smallest_to_tiniest", r"\bsmallest\b", ["tiniest"]),
    ],
    "window_function": [
        ("largest_to_biggest", r"\blargest\b", ["biggest"]),
        ("smallest_to_tiniest", r"\bsmallest\b", ["tiniest"]),
    ],
    "attribute_filter": [
        ("official_to_recognized", r"\bofficial\b", ["recognized", "recognized territorial"]),
        ("land_based_to_on_land", r"land-based", ["on-land", "on land"]),
        ("sovereign_to_official", r"\bsovereign\b", ["official"]),
    ],
    "direct_lookup": [
        ("where_is_to_show", r"\bwhere is\b", ["show", "locate"]),
        ("map_of_to_outline", r"\bmap of\b", ["outline of"]),
    ],
    "disambiguation": [
        ("show_me_to_find", r"\bshow me\b", ["find", "show"]),
        ("pull_up_to_find", r"\bpull up\b", ["find", "show"]),
    ],
}


def _diversify_question_surface(question: str, family: str) -> tuple[str, List[str]]:
    """Apply light family-aware paraphrasing to reduce template memorization.

    Rewrites are intentionally shallow and lexically local so the generated
    question stays aligned with the underlying SQL intent.
    """
    if not question or random.random() < 0.35:
        return question, []

    rules = _GENERIC_SURFACE_RULES + _FAMILY_SURFACE_RULES.get(family, [])
    rewritten = question
    applied: List[str] = []
    max_rewrites = 2 if random.random() < 0.5 else 1

    for _ in range(max_rewrites):
        matches = []
        for label, pattern, replacements in rules:
            if re.search(pattern, rewritten, flags=re.IGNORECASE):
                for replacement in replacements:
                    matches.append((label, pattern, replacement))
        if not matches:
            break

        label, pattern, replacement = random.choice(matches)
        updated = re.sub(pattern, replacement, rewritten, count=1, flags=re.IGNORECASE)
        if updated == rewritten:
            continue
        rewritten = re.sub(r"\s+", " ", updated).strip()
        applied.append(f"{family}:{label}")

    return rewritten, applied


# Import templates from same directory
from . import sql_templates
TEMPLATES = sql_templates.TEMPLATES
SQLTemplate = sql_templates.SQLTemplate
get_templates_by_family = sql_templates.get_templates_by_family


_NE_NAMED_LOOKUP_SUBTYPES = {
    'sea', 'ocean', 'lake', 'river', 'basin', 'gulf', 'bay',
    'island group', 'peninsula', 'strait', 'range/mtn', 'depression',
}

_NE_TEMPLATE_SUBTYPES = {
    'lookup_02': {'sea', 'ocean', 'lake', 'river', 'basin', 'gulf', 'bay', 'island group', 'peninsula', 'strait', 'range/mtn', 'depression'},
    'adj_03': {'sea', 'ocean'},
    'adj_09': {'river', 'lake', 'basin'},
    'adj_10': {'range/mtn', 'peninsula', 'depression'},
    'adj_06': {'sea', 'ocean', 'lake', 'river', 'basin', 'gulf', 'bay', 'strait', 'range/mtn', 'peninsula', 'depression', 'plateau', 'plain', 'lowland', 'valley', 'gorge'},
    'adj_07': {'sea', 'ocean', 'lake', 'river', 'basin', 'gulf', 'bay', 'strait', 'range/mtn', 'peninsula', 'depression', 'plateau', 'plain', 'lowland', 'valley', 'gorge'},
    'adj_08': {'sea', 'ocean', 'lake', 'river', 'basin', 'gulf', 'bay', 'strait', 'range/mtn', 'peninsula', 'depression', 'plateau', 'plain', 'lowland', 'valley', 'gorge'},
    'contain_04': {'sea', 'ocean', 'gulf', 'bay', 'basin', 'island group', 'peninsula', 'range/mtn', 'depression'},
    'contain_05': {'sea', 'ocean', 'gulf', 'bay', 'strait'},
    'intersect_03': {'river', 'lake', 'basin', 'gulf', 'bay', 'strait', 'range/mtn', 'peninsula', 'depression'},
    'intersect_04': {'river', 'lake', 'basin', 'gulf', 'bay', 'strait', 'range/mtn', 'peninsula', 'depression'},
    'intersect_06': {'river', 'lake', 'basin', 'gulf', 'bay', 'strait', 'range/mtn', 'peninsula', 'depression'},
    'buffer_02': {'sea', 'ocean', 'lake', 'river', 'basin', 'gulf', 'bay', 'island group', 'peninsula', 'strait', 'range/mtn', 'depression'},
    'buffer_11': {'sea', 'ocean', 'lake', 'river', 'basin', 'gulf', 'bay', 'island group', 'peninsula', 'strait', 'range/mtn', 'depression'},
    'chained_03': {'island group', 'peninsula', 'range/mtn', 'depression'},
    'chained_04': {'river', 'lake', 'basin'},
    'chained_05': {'range/mtn', 'depression'},
    'chained_08': {'river', 'lake', 'basin'},
    'chained_09': {'range/mtn', 'depression'},
    'partial_05': {'sea', 'ocean', 'lake', 'river', 'basin', 'gulf', 'bay', 'island group', 'peninsula', 'strait', 'range/mtn', 'depression'},
    'diff_02': {'sea', 'ocean', 'lake', 'river', 'basin', 'gulf', 'bay', 'island group', 'peninsula', 'strait', 'range/mtn', 'depression'},
}


class Candidate(BaseModel):
    """Candidate entity for grounding."""
    candidate_id: str
    source: str
    id: str
    name: str
    subtype: Optional[str] = None
    country: Optional[str] = None
    region: Optional[str] = None
    admin_level: Optional[int] = None
    similarity: float = 0.0


class TrainingSample(BaseModel):
    """Complete training sample."""
    id: str
    question: str
    candidates: List[Candidate]
    target: Dict[str, Any]
    metadata: Dict[str, Any]


def load_relation_tables(intermediate_dir: Path, quiet: bool = False) -> Dict[str, pd.DataFrame]:
    """Load all precomputed relation tables."""
    tables = {}
    
    for file in intermediate_dir.glob("*.parquet"):
        name = file.stem
        tables[name] = pd.read_parquet(file)
        if not quiet:
            print(f"  {name}: {len(tables[name])} rows")
    
    return tables


def sample_adjacency_anchor(
    adjacency_df: pd.DataFrame,
    target_subtype: Optional[str] = None,
    anchor_subtypes: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Sample a random adjacency pair, optionally filtered by subtypes.

    When ``target_subtype`` is provided, only rows whose neighbouring feature
    matches that subtype are considered. When ``anchor_subtypes`` is provided,
    only rows whose anchor feature is one of those subtypes are considered.
    Both filters are applied together so sampled pairs are geographically
    coherent with the template intent (e.g. country anchor → country result).
    """
    if adjacency_df.empty:
        return None

    df = adjacency_df
    if target_subtype is not None:
        df = df[df['target_subtype'] == target_subtype]
        if df.empty:
            return None
    if anchor_subtypes is not None:
        filtered = df[df['anchor_subtype'].isin(anchor_subtypes)]
        if not filtered.empty:
            df = filtered

    row = df.sample(n=1).iloc[0]
    return {
        'anchor_id': row['anchor_id'],
        'anchor_name': row['anchor_name'],
        'anchor_subtype': row['anchor_subtype'],
        'anchor_country': row.get('anchor_country'),  # May not exist in all tables
        'target_id': row.get('target_id'),
        'target_name': row.get('target_name'),
        'target_subtype': row.get('target_subtype')
    }


def sample_intersection_anchor(intersection_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Sample a random intersection pair."""
    if intersection_df.empty:
        return None
    
    row = intersection_df.sample(n=1).iloc[0]
    return {
        'anchor_id': row['anchor_id'],
        'anchor_name': row['anchor_name'],
        'anchor_subtype': row['anchor_subtype'],
        'target_id': row.get('target_id'),
        'target_name': row.get('target_name'),
        'target_subtype': row.get('target_subtype')
    }


def sample_containment_anchor(containment_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Sample a random containment pair.

    Returns both ends of the pair so callers that need the contained entity
    (e.g. difference templates that clip container by contained) can use it
    directly without a second random draw.
    """
    if containment_df.empty:
        return None

    row = containment_df.sample(n=1).iloc[0]
    return {
        'container_id': row['container_id'],
        'container_name': row['container_name'],
        'container_subtype': row['container_subtype'],
        'contained_id': row['contained_id'],
        'contained_name': row['contained_name'],
        'contained_subtype': row['contained_subtype'],
    }


def sample_disambiguation_anchor(
    containment_df: pd.DataFrame,
    contained_subtypes: List[str],
    container_subtypes: List[str],
) -> Optional[Dict[str, Any]]:
    """Sample a (contained, container) pair from containment_pairs.

    Used by disambiguation templates like "Puri, Odisha" where the contained
    entity is the target and the container provides disambiguation context.
    """
    if containment_df.empty:
        return None

    df = containment_df[
        containment_df['contained_subtype'].isin(contained_subtypes)
        & containment_df['container_subtype'].isin(container_subtypes)
    ]
    if df.empty:
        return None

    row = df.sample(n=1).iloc[0]
    return {
        'contained_id': row['contained_id'],
        'contained_name': row['contained_name'],
        'contained_subtype': row['contained_subtype'],
        'container_id': row['container_id'],
        'container_name': row['container_name'],
        'container_subtype': row['container_subtype'],
    }


def sample_cross_source_anchor(
    cross_source_df: pd.DataFrame,
    natural_subtypes: Optional[set[str]] = None,
    relation_types: Optional[set[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Sample a random cross-source relation with optional subtype filters."""
    if cross_source_df.empty:
        return None

    df = cross_source_df
    if natural_subtypes is not None:
        df = df[df['natural_subtype'].isin(natural_subtypes)]
    if relation_types is not None:
        df = df[df['relation_type'].isin(relation_types)]
    if df.empty:
        return None

    row = df.sample(n=1).iloc[0]
    return {
        'division_id': row['division_id'],
        'division_name': row['division_name'],
        'division_subtype': row['division_subtype'],
        'natural_id': row['natural_id'],
        'natural_name': row['natural_name'],
        'natural_subtype': row['natural_subtype'],
        'relation_type': row['relation_type']
    }


def _merge_candidate_lists(
    *lists: List[Candidate],
    max_total: int = 10,
) -> List[Candidate]:
    """Merge N candidate lists, deduplicate by id, reassign candidate_ids.

    Interleaves the lists so each anchor is represented before any anchor
    gets a second candidate — matching the grouped-then-interleaved order
    that inference produces.
    """
    from itertools import zip_longest

    seen: set = set()
    merged: List[Candidate] = []
    for row in zip_longest(*lists):
        for c in row:
            if c is None:
                continue
            if c.id not in seen:
                merged.append(c)
                seen.add(c.id)
            if len(merged) >= max_total:
                break
        if len(merged) >= max_total:
            break
    for i, c in enumerate(merged, 1):
        c.candidate_id = f"c{i}"
    return merged


def build_candidate_list(
    con: duckdb.DuckDBPyConnection,
    anchor_id: str,
    anchor_name: str,
    anchor_source: str,
    num_candidates: int = 10,
    difficulty: str = "medium"
) -> List[Candidate]:
    """Build candidate list with true anchor + distractors."""

    # Helper to convert pandas NA to None
    def safe_get(row, key, default=None):
        val = row.get(key, default)
        return None if pd.isna(val) else val

    # Get the true anchor
    if anchor_source == "divisions_area":
        query = """
        SELECT
            id,
            names."primary" AS name,
            subtype,
            country,
            region,
            admin_level
        FROM read_parquet(?)
        WHERE id = ?
        """
        anchor_row = con.execute(query, [DIVISIONS_AREA_PATH, anchor_id]).fetchdf().iloc[0]
    else:
        query = """
        SELECT
            id,
            names."primary" AS name,
            subtype
        FROM read_parquet(?)
        WHERE id = ?
        """
        anchor_row = con.execute(query, [NATURAL_EARTH_PATH, anchor_id]).fetchdf().iloc[0]

    true_candidate = Candidate(
        candidate_id="c1",
        source=anchor_source,
        id=anchor_id,
        name=safe_get(anchor_row, 'name'),
        subtype=safe_get(anchor_row, 'subtype'),
        country=safe_get(anchor_row, 'country'),
        region=safe_get(anchor_row, 'region'),
        admin_level=safe_get(anchor_row, 'admin_level'),
        similarity=1.0,
    )

    distractors = build_distractors(
        con,
        anchor_name,
        anchor_source,
        anchor_id,
        num_candidates - 1,
        difficulty,
    )

    # Deduplicate by underlying entity id while preserving order.
    # Some parquet sources contain repeated rows for the same feature id,
    # which can otherwise leak duplicate candidates into the dataset.
    candidates: List[Candidate] = []
    seen_ids: set[str] = set()
    for cand in [true_candidate] + distractors:
        if cand.id in seen_ids:
            continue
        candidates.append(cand)
        seen_ids.add(cand.id)
        if len(candidates) >= num_candidates:
            break

    for i, cand in enumerate(candidates, 1):
        cand.candidate_id = f"c{i}"

    return candidates


def build_distractors(
    con: duckdb.DuckDBPyConnection,
    anchor_name: str,
    anchor_source: str,
    exclude_id: str,
    num_distractors: int,
    difficulty: str,
    cross_source_ratio: float = 0.5,
) -> List[Candidate]:
    """Build distractor candidates using fuzzy search.

    Always includes candidates from both sources so the model sees mixed
    ``source`` values in every training example — matching the inference
    behaviour where search.py queries divisions_area AND natural_earth equally
    (5 results each per place).

    Args:
        cross_source_ratio: Fraction of distractors drawn from the *other*
            source.  Defaults to 0.5 (50/50 split) to match inference exactly.
    """

    def safe_get(row, key, default=None):
        val = row.get(key, default)
        return None if pd.isna(val) else val

    def _query_source(path: str, src_name: str, n: int, excl_id: str) -> List[Candidate]:
        query = """
        WITH ranked AS (
            SELECT
                id,
                names."primary" AS name,
                subtype,
                country,
                region,
                admin_level,
                jaro_winkler_similarity(lower(names."primary"), lower(?)) AS similarity,
                ROW_NUMBER() OVER (
                    PARTITION BY id
                    ORDER BY jaro_winkler_similarity(lower(names."primary"), lower(?)) DESC
                ) AS rn
            FROM read_parquet(?)
            WHERE id != ?
              AND names."primary" IS NOT NULL
              AND trim(names."primary") != ''
              AND geometry IS NOT NULL
        )
        SELECT
            id,
            name,
            subtype,
            country,
            region,
            admin_level,
            similarity
        FROM ranked
        WHERE rn = 1
        ORDER BY similarity DESC
        LIMIT ?
        """
        df = con.execute(query, [anchor_name, anchor_name, path, excl_id, n]).fetchdf()
        results = []
        for _, row in df.iterrows():
            results.append(Candidate(
                candidate_id="temp",
                source=src_name,
                id=row["id"],
                name=safe_get(row, "name"),
                subtype=safe_get(row, "subtype"),
                country=safe_get(row, "country"),
                region=safe_get(row, "region"),
                admin_level=safe_get(row, "admin_level"),
                similarity=float(row["similarity"]),
            ))
        return results

    cross_n = max(1, round(num_distractors * cross_source_ratio))
    same_n = num_distractors - cross_n

    if anchor_source == "divisions_area":
        same = _query_source(DIVISIONS_AREA_PATH, "divisions_area", same_n, exclude_id)
        cross = _query_source(NATURAL_EARTH_PATH, "natural_earth", cross_n, "")
    else:
        same = _query_source(NATURAL_EARTH_PATH, "natural_earth", same_n, exclude_id)
        cross = _query_source(DIVISIONS_AREA_PATH, "divisions_area", cross_n, "")

    return same + cross


def sample_random_entity(
    con: duckdb.DuckDBPyConnection,
    inventory_df: pd.DataFrame,
    source: str,
    subtypes: Optional[set[str]] = None,
    countries: Optional[set[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Sample a random entity from inventory with optional filters."""
    if inventory_df.empty:
        return None

    df = inventory_df
    if subtypes is not None:
        df = df[df['subtype'].isin(subtypes)]
    if countries is not None and 'country' in df.columns:
        df = df[df['country'].isin(countries)]
    if df.empty:
        return None

    row = df.sample(n=1).iloc[0]
    return {
        'id': row['id'],
        'name': row['name'],
        'subtype': row.get('subtype'),
        'country': row.get('country'),
        'source': source
    }


def generate_template_based_sample(
    con: duckdb.DuckDBPyConnection,
    template: SQLTemplate,
    tables: Dict[str, pd.DataFrame],
    sample_id: str
) -> Optional[TrainingSample]:
    """Generate a sample based on a SQL template."""
    
    # Sample anchor based on template requirements
    if template.family == "direct_lookup":
        # Just pick a random entity
        if template.anchor_source == "divisions_area":
            anchor = sample_random_entity(con, tables['divisions_area_inventory'], 'divisions_area')
        else:
            anchor = sample_random_entity(
                con,
                tables['natural_earth_inventory'],
                'natural_earth',
                subtypes=_NE_TEMPLATE_SUBTYPES.get(template.template_id, _NE_NAMED_LOOKUP_SUBTYPES),
            )
        
        if not anchor:
            return None
        
        # Render SQL
        sql = template.sql_template.format(
            anchor_id=anchor['id']
        )
        
        # Build candidates
        candidates = build_candidate_list(
            con, anchor['id'], anchor['name'], anchor['source'],
            num_candidates=10, difficulty="easy"
        )
        
        # Question
        question = random.choice(template.question_hints).format(anchor_name=anchor['name'])

    elif template.family == "disambiguation":
        # "Puri, Odisha" style: pick a (contained, container) pair whose
        # subtypes match the template, build candidates that include the
        # container + same-name distractors so the model must read the CSV
        # to pick the right entry.
        _disambig_subtypes = {
            "disambiguate_01": (["county"], ["region", "country"]),
            "disambiguate_02": (["county"], ["country"]),
            "disambiguate_03": (["region"], ["country"]),
        }
        contained_sts, container_sts = _disambig_subtypes.get(
            template.template_id, (["county"], ["country"])
        )

        pair = sample_disambiguation_anchor(
            tables["containment_pairs"], contained_sts, container_sts
        )
        if not pair:
            return None

        candidates = build_candidate_list(
            con, pair["contained_id"], pair["contained_name"], "divisions_area",
            num_candidates=10, difficulty="hard"
        )

        # Ensure the container is among the candidates so the model can
        # ground the disambiguation context (e.g. "Odisha").
        if not any(c.id == pair["container_id"] for c in candidates):
            container_rows = con.execute(
                'SELECT id, names."primary" AS name, subtype, country, region, admin_level '
                'FROM read_parquet(?) WHERE id = ? LIMIT 1',
                [DIVISIONS_AREA_PATH, pair["container_id"]]
            ).fetchdf()
            if container_rows.empty:
                return None
            crow = container_rows.iloc[0]

            def _nn(v):
                return None if pd.isna(v) else v

            container_cand = Candidate(
                candidate_id="temp",
                source="divisions_area",
                id=pair["container_id"],
                name=_nn(crow["name"]),
                subtype=_nn(crow["subtype"]),
                country=_nn(crow["country"]),
                region=_nn(crow["region"]),
                admin_level=_nn(crow["admin_level"]),
                similarity=0.95,
            )
            # Insert the container right after the true target and drop the
            # last filler distractor so the total stays at 10.
            candidates = [candidates[0], container_cand] + candidates[1:-1]
            for i, c in enumerate(candidates, 1):
                c.candidate_id = f"c{i}"

        sql = template.sql_template.format(anchor_id=pair["contained_id"])

        question = random.choice(template.question_hints).format(
            anchor_name=pair["contained_name"],
            container_name=pair["container_name"],
        )

        # Only the contained entity is the query target — the container is
        # disambiguation context and stays in candidates but NOT in
        # selected_candidates. The model learns to use the container row of
        # the CSV (via country/region columns) to pick the right same-name
        # county or region.
        anchor = {"id": pair["contained_id"], "name": pair["contained_name"]}

    elif template.family == "adjacency":
        # adj_03/09/10/11/12: division anchor -> natural_earth targets.
        # adj_06/07/08: natural_earth anchor -> admin targets.
        # Use cross_source_relations so anchors are guaranteed to intersect.
        _NE_TARGET_ADJ_SUBTYPES = {
            "adj_03": ("ocean", "sea"),
            "adj_09": ("river", "lake", "basin"),
            "adj_10": ("range/mtn",),
            "adj_11": ("plateau",),
            "adj_12": ("plain", "lowland", "basin", "valley", "depression", "gorge"),
        }
        if template.template_id in _NE_TARGET_ADJ_SUBTYPES:
            cs_df = tables.get('cross_source_relations', pd.DataFrame())
            if cs_df.empty:
                return None
            ne_types = _NE_TARGET_ADJ_SUBTYPES[template.template_id]
            filtered = cs_df[cs_df['natural_subtype'].isin(ne_types)]
            if filtered.empty:
                return None
            row = filtered.sample(n=1).iloc[0]
            anchor = {
                'anchor_id': row['division_id'],
                'anchor_name': row['division_name'],
                'anchor_subtype': row['division_subtype'],
                'target_subtype': row['natural_subtype'],
                'anchor_source': 'divisions_area',
            }
        elif template.anchor_source == "natural_earth":
            cs_anchor = sample_cross_source_anchor(
                tables.get('cross_source_relations', pd.DataFrame()),
                natural_subtypes=_NE_TEMPLATE_SUBTYPES.get(template.template_id),
            )
            if not cs_anchor:
                return None
            anchor = {
                'anchor_id': cs_anchor['natural_id'],
                'anchor_name': cs_anchor['natural_name'],
                'target_subtype': template.target_subtype,
                'anchor_source': 'natural_earth',
            }
        else:
            # divisions_area self-join adjacency.
            _ADJ_ANCHOR_SUBTYPES = {
                "adj_02": ["country", "region"],
                "adj_04": ["region"],
                "adj_05": ["country"],
            }
            filter_subtype = (
                template.target_subtype
                if '{target_subtype}' in template.sql_template
                else None
            )
            anchor = sample_adjacency_anchor(
                tables['adjacency_pairs'],
                target_subtype=filter_subtype,
                anchor_subtypes=_ADJ_ANCHOR_SUBTYPES.get(template.template_id),
            )
            if anchor:
                anchor['anchor_source'] = 'divisions_area'
        if not anchor:
            return None

        sql = template.sql_template.format(
            anchor_id=anchor['anchor_id'],
            target_subtype=anchor.get('target_subtype', ''),
        )

        candidates = build_candidate_list(
            con, anchor['anchor_id'], anchor['anchor_name'], anchor.get('anchor_source', 'divisions_area'),
            num_candidates=10, difficulty="medium"
        )

        question = random.choice(template.question_hints).format(
            anchor_name=anchor['anchor_name'],
            target_subtype=anchor.get('target_subtype', ''),
        )
        
    elif template.family == "containment":
        if template.anchor_source == "natural_earth":
            # contain_04 / contain_05: NE anchor (sea, desert, etc.).
            # Use cross_source_relations so the anchor exists in natural_earth
            # and is guaranteed to intersect divisions_area features.
            cs_anchor = sample_cross_source_anchor(
                tables.get('cross_source_relations', pd.DataFrame()),
                natural_subtypes=_NE_TEMPLATE_SUBTYPES.get(template.template_id),
            )
            if not cs_anchor:
                return None
            anchor_id = cs_anchor['natural_id']
            anchor_name = cs_anchor['natural_name']
            target_subtype = template.target_subtype or 'country'

            sql = template.sql_template.format(
                anchor_id=anchor_id,
                target_subtype=target_subtype,
            )
            candidates = build_candidate_list(
                con, anchor_id, anchor_name, 'natural_earth',
                num_candidates=10, difficulty="medium"
            )
            question = random.choice(template.question_hints).format(
                anchor_name=anchor_name,
                target_subtype=target_subtype,
            )
            anchor = {'id': anchor_id, 'name': anchor_name}

        elif template.template_id == "contain_02":
            # "What country contains X?" - anchor is the CONTAINED entity;
            # result is the country that ST_Contains it.
            # Guard against stale relation tables by only allowing contained
            # subtypes that exist in the simplified admin schema.
            df = tables['containment_pairs']
            df = df[
                (df['container_subtype'] == 'country')
                & (df['contained_subtype'].isin(['region', 'county']))
            ]
            pair = sample_containment_anchor(df)
            if not pair:
                return None

            sql = template.sql_template.format(
                anchor_id=pair['contained_id'],
                target_subtype='country',
            )
            candidates = build_candidate_list(
                con, pair['contained_id'], pair['contained_name'], 'divisions_area',
                num_candidates=10, difficulty="medium"
            )
            question = random.choice(template.question_hints).format(
                anchor_name=pair['contained_name'],
                target_subtype='country',
            )
            anchor = {'id': pair['contained_id'], 'name': pair['contained_name']}

        elif template.template_id == "contain_03":
            # "What regions are in country X?" - anchor is a country, target is regions.
            df = tables['containment_pairs']
            df = df[
                (df['container_subtype'] == 'country')
                & (df['contained_subtype'] == 'region')
            ]
            pair = sample_containment_anchor(df)
            if not pair:
                return None

            sql = template.sql_template.format(
                anchor_id=pair['container_id'],
                target_subtype='region',
            )
            candidates = build_candidate_list(
                con, pair['container_id'], pair['container_name'], 'divisions_area',
                num_candidates=10, difficulty="medium"
            )
            question = random.choice(template.question_hints).format(
                anchor_name=pair['container_name'],
                target_subtype='region',
            )
            anchor = {'id': pair['container_id'], 'name': pair['container_name']}

        else:
            # contain_01: standard containment.
            # Enforce hierarchy: county must be inside region or country, never
            # inside another county. Filter container_subtype accordingly.
            # Also filter contained_subtype to match template.target_subtype so
            # hardcoded vocab hints (e.g. "districts") always align with the SQL.
            _VALID_CONTAINERS = {
                "county":  ["region", "country"],
                "region":  ["country"],
            }
            df = tables['containment_pairs']
            if template.target_subtype:
                filtered = df[df['contained_subtype'] == template.target_subtype]
                if not filtered.empty:
                    df = filtered
            valid_containers = _VALID_CONTAINERS.get(template.target_subtype)
            if valid_containers:
                filtered = df[df['container_subtype'].isin(valid_containers)]
                if not filtered.empty:
                    df = filtered
            anchor = sample_containment_anchor(df)
            if not anchor:
                return None

            target_subtype = template.target_subtype or anchor['contained_subtype']

            sql = template.sql_template.format(
                anchor_id=anchor['container_id'],
                target_subtype=target_subtype,
            )
            candidates = build_candidate_list(
                con, anchor['container_id'], anchor['container_name'], 'divisions_area',
                num_candidates=10, difficulty="medium"
            )
            question = random.choice(template.question_hints).format(
                anchor_name=anchor['container_name'],
                target_subtype=target_subtype,
            )
        
    elif template.family == "intersection":
        if template.anchor_source == "natural_earth":
            anchor = sample_cross_source_anchor(
                tables['cross_source_relations'],
                natural_subtypes=_NE_TEMPLATE_SUBTYPES.get(template.template_id),
            )
            if not anchor:
                return None

            target_subtype = template.target_subtype or 'country'

            sql = template.sql_template.format(
                        anchor_id=anchor['natural_id'],
                target_subtype=target_subtype,
            )

            candidates = build_candidate_list(
                con, anchor['natural_id'], anchor['natural_name'], 'natural_earth',
                num_candidates=10, difficulty="medium"
            )

            question = random.choice(template.question_hints).format(
                anchor_name=anchor['natural_name'],
                target_subtype=target_subtype,
            )
        else:
            # Same-source intersection.
            # If the template pins a target_subtype (e.g. intersect_02 targets county),
            # filter intersection_pairs so the sampled pair is guaranteed to match.
            idf = tables['intersection_pairs']
            if template.target_subtype and not idf.empty:
                filtered = idf[idf['target_subtype'] == template.target_subtype]
                if filtered.empty:
                    return None
                idf = filtered
            anchor = sample_intersection_anchor(idf)
            if not anchor:
                return None

            target_subtype = template.target_subtype or anchor.get('target_subtype') or 'region'

            sql = template.sql_template.format(
                        anchor_id=anchor['anchor_id'],
                target_subtype=target_subtype
            )

            candidates = build_candidate_list(
                con, anchor['anchor_id'], anchor['anchor_name'], 'divisions_area',
                num_candidates=10, difficulty="medium"
            )

            question = random.choice(template.question_hints).format(
                anchor_name=anchor['anchor_name'],
                target_subtype=target_subtype
            )
    
    elif template.family == "set_operations":
        if template.template_id == "union_03":
            # 3-anchor union by ID — candidates: 3 per anchor (9 total)
            anchors = [
                sample_random_entity(con, tables['divisions_area_inventory'], 'divisions_area')
                for _ in range(3)
            ]
            if any(a is None for a in anchors):
                return None
            anchor1, anchor2, anchor3 = anchors

            sql = template.sql_template.format(
                    anchor_id_1=anchor1['id'],
                anchor_id_2=anchor2['id'],
                anchor_id_3=anchor3['id'],
            )

            per_anchor = 3
            cands = [
                build_candidate_list(con, a['id'], a['name'], 'divisions_area',
                                     num_candidates=per_anchor, difficulty="medium")
                for a in anchors
            ]
            candidates = _merge_candidate_lists(*cands, max_total=9)

            question = random.choice(template.question_hints).format(
                anchor_1_name=anchor1['name'],
                anchor_2_name=anchor2['name'],
                anchor_3_name=anchor3['name'],
            )

        elif template.template_id in ("contain_multi_01", "contain_multi_02", "contain_multi_03"):
            # country IN clause — 2 or 3 anchors, each contributes its country code
            num_a = 3 if template.template_id == "contain_multi_02" else 2
            anchors = [
                sample_random_entity(
                    con,
                    tables['divisions_area_inventory'],
                    'divisions_area',
                    subtypes={'country'},
                )
                for _ in range(num_a)
            ]
            if any(a is None for a in anchors):
                return None

            countries = [a.get('country') or 'US' for a in anchors]
            target_subtype = template.target_subtype or 'region'
            per_anchor = 3 if num_a == 3 else 4

            fmt_kwargs = dict(
                    target_subtype=target_subtype,
            )
            for i, c in enumerate(countries, 1):
                fmt_kwargs[f'country_{i}'] = c

            sql = template.sql_template.format(**fmt_kwargs)

            cands = [
                build_candidate_list(con, a['id'], a['name'], 'divisions_area',
                                     num_candidates=per_anchor, difficulty="medium")
                for a in anchors
            ]
            candidates = _merge_candidate_lists(*cands, max_total=num_a * per_anchor)

            q_kwargs = dict(target_subtype=target_subtype)
            for i, a in enumerate(anchors, 1):
                q_kwargs[f'anchor_{i}_name'] = a['name']

            question = random.choice(template.question_hints).format(**q_kwargs)

        elif template.template_id == "union_02":
            # Filtered union: ST_Union_Agg of contained sub-features.
            # Pin to template.target_subtype so hardcoded vocabulary hints
            # (e.g. "districts") always match the SQL subtype.
            df = tables['containment_pairs']
            if template.target_subtype:
                filtered = df[df['contained_subtype'] == template.target_subtype]
                if not filtered.empty:
                    df = filtered
            pair = sample_containment_anchor(df)
            if not pair:
                return None

            target_subtype = template.target_subtype or pair.get('contained_subtype', 'county')
            sql = template.sql_template.format(
                    anchor_id=pair['container_id'],
                target_subtype=target_subtype,
            )

            candidates = build_candidate_list(
                con, pair['container_id'], pair['container_name'], 'divisions_area',
                num_candidates=10, difficulty="medium"
            )

            question = random.choice(template.question_hints).format(
                anchor_name=pair['container_name'],
                target_subtype=target_subtype,
            )

        else:
            # union_01: 2-anchor union by ID — candidates: 5 per anchor
            anchor1 = sample_random_entity(con, tables['divisions_area_inventory'], 'divisions_area')
            anchor2 = sample_random_entity(con, tables['divisions_area_inventory'], 'divisions_area')
            if not anchor1 or not anchor2:
                return None

            sql = template.sql_template.format(
                    anchor_id_1=anchor1['id'],
                anchor_id_2=anchor2['id'],
            )

            cands1 = build_candidate_list(
                con, anchor1['id'], anchor1['name'], 'divisions_area',
                num_candidates=5, difficulty="medium"
            )
            cands2 = build_candidate_list(
                con, anchor2['id'], anchor2['name'], 'divisions_area',
                num_candidates=5, difficulty="medium"
            )
            candidates = _merge_candidate_lists(cands1, cands2, max_total=10)

            question = random.choice(template.question_hints).format(
                anchor_1_name=anchor1['name'],
                anchor_2_name=anchor2['name'],
            )
    
    elif template.family == "buffer":
        # Buffer operations use metre distances in SQL and a human-readable
        # buffer_label in questions, e.g. (1000, "1 km") or (250, "250 m").
        # The template SQL divides by 111 320 to approximate metres in degrees.
        _buffer_choices = [
            (100, "100 m"),
            (250, "250 m"),
            (500, "500 m"),
            (1000, "1 km"),
            (2000, "2 km"),
            (5000, "5 km"),
            (10000, "10 km"),
            (25000, "25 km"),
            (50000, "50 km"),
            (100000, "100 km"),
            (200000, "200 km"),
        ]

        if template.num_anchors == 1:
            if template.anchor_source == "natural_earth":
                anchor = sample_random_entity(
                    con,
                    tables['natural_earth_inventory'],
                    'natural_earth',
                    subtypes=_NE_TEMPLATE_SUBTYPES.get(template.template_id, _NE_NAMED_LOOKUP_SUBTYPES),
                )
            else:
                anchor = sample_random_entity(con, tables['divisions_area_inventory'], 'divisions_area')
            if not anchor:
                return None

            buffer_m, buffer_label = random.choice(_buffer_choices)
            fmt_kwargs = dict(anchor_id=anchor['id'], buffer_m=buffer_m)
            q_kwargs = dict(anchor_name=anchor['name'], buffer_label=buffer_label)

            if template.target_subtype:
                fmt_kwargs['target_subtype'] = template.target_subtype
                q_kwargs['target_subtype'] = template.target_subtype

            sql = template.sql_template.format(**fmt_kwargs)

            candidates = build_candidate_list(
                con, anchor['id'], anchor['name'], anchor['source'],
                num_candidates=10, difficulty="medium"
            )

            question = random.choice(template.question_hints).format(**q_kwargs)
        else:
            # Multi-anchor buffer (2–5 places): union of individual buffers.
            num_a = template.num_anchors
            anchors = []
            for _ in range(num_a):
                a = sample_random_entity(con, tables['divisions_area_inventory'], 'divisions_area')
                if not a:
                    return None
                anchors.append(a)

            buffer_m, buffer_label = random.choice(_buffer_choices[:7])

            fmt_kwargs = {f'anchor_id_{i+1}': a['id'] for i, a in enumerate(anchors)}
            fmt_kwargs['buffer_m'] = buffer_m
            if template.target_subtype:
                fmt_kwargs['target_subtype'] = template.target_subtype

            sql = template.sql_template.format(**fmt_kwargs)

            # Build one candidate list per anchor then merge.
            per_anchor_n = max(2, 10 // num_a)
            cand_lists = [
                build_candidate_list(
                    con, a['id'], a['name'], 'divisions_area',
                    num_candidates=per_anchor_n, difficulty="medium",
                )
                for a in anchors
            ]
            candidates = _merge_candidate_lists(*cand_lists)

            q_kwargs = {f'anchor_{i+1}_name': a['name'] for i, a in enumerate(anchors)}
            q_kwargs['buffer_label'] = buffer_label
            if template.target_subtype:
                q_kwargs['target_subtype'] = template.target_subtype

            question = random.choice(template.question_hints).format(**q_kwargs)
    
    elif template.family == "partial_selection":
        # Partial selection (northern half, clipping, etc.)
        anchor = sample_random_entity(con, tables['divisions_area_inventory'], 'divisions_area')
        if not anchor:
            return None
        
        if template.num_anchors == 1:
            sql = template.sql_template.format(
                        anchor_id=anchor['id'],
            )
            question = random.choice(template.question_hints).format(
                anchor_name=anchor['name'],
            )
            candidates = build_candidate_list(
                con, anchor['id'], anchor['name'], 'divisions_area',
                num_candidates=10, difficulty="hard",
            )
        else:
            # Mixed-source clip: division intersected with a natural_earth feature.
            # Use cross_source_relations so the pair is guaranteed to intersect —
            # random sampling almost never produces an intersecting pair.
            cs_anchor = sample_cross_source_anchor(
                tables.get('cross_source_relations', pd.DataFrame()),
                natural_subtypes=_NE_TEMPLATE_SUBTYPES.get(template.template_id),
            )
            if not cs_anchor:
                return None
            clip_feature = {
                'id':   cs_anchor['natural_id'],
                'name': cs_anchor['natural_name'],
                'source': 'natural_earth',
            }
            # Override the division anchor with the paired division so the
            # ST_Intersects check in the SQL is guaranteed to pass.
            anchor = {
                'id':   cs_anchor['division_id'],
                'name': cs_anchor['division_name'],
                'source': 'divisions_area',
            }

            sql = template.sql_template.format(
                        anchor_id=anchor['id'],
                clip_feature_id=clip_feature['id'],
            )
            question = random.choice(template.question_hints).format(
                anchor_name=anchor['name'],
                clip_feature_name=clip_feature['name'],
            )
            # Build candidates for BOTH anchors so the model sees both IDs
            # in context and learns to pick the right one for each placeholder.
            div_cands = build_candidate_list(
                con, anchor['id'], anchor['name'], 'divisions_area',
                num_candidates=5, difficulty="hard",
            )
            ne_cands = build_candidate_list(
                con, clip_feature['id'], clip_feature['name'], 'natural_earth',
                num_candidates=5, difficulty="hard",
            )
            candidates = _merge_candidate_lists(div_cands, ne_cands, max_total=10)
    
    elif template.family == "aggregation":
        # Teach the model to distinguish singular superlatives ("the largest")
        # from explicit top-N requests ("top 5 largest").
        top_n = random.choice([1, 3, 5, 10])
        target_subtype = random.choice(['county', 'region'])
        singular_hints = [h for h in template.question_hints if '{top_n}' not in h]
        plural_hints = [h for h in template.question_hints if '{top_n}' in h]
        question_hint_pool = singular_hints if top_n == 1 and singular_hints else plural_hints or template.question_hints

        if template.template_id in ['agg_03', 'agg_04']:
            # Country-level aggregation: SQL uses country code, so the anchor
            # in the question must also be a country.
            anchor = sample_random_entity(
                con,
                tables['divisions_area_inventory'],
                'divisions_area',
                subtypes={'country'},
            )
            if not anchor:
                return None

            country = anchor.get('country') or 'US'

            sql = template.sql_template.format(
                    country=country,
                target_subtype=target_subtype,
                top_n=top_n,
            )

            candidates = build_candidate_list(
                con, anchor['id'], anchor['name'], 'divisions_area',
                num_candidates=10, difficulty="hard"
            )

            question = random.choice(question_hint_pool).format(
                top_n=top_n,
                target_subtype=target_subtype,
                anchor_name=anchor['name'],
            )
        else:
            # Containment-based aggregation: anchor is the container region.
            anchor = sample_containment_anchor(tables['containment_pairs'])
            if not anchor:
                return None

            sql = template.sql_template.format(
                    anchor_id=anchor['container_id'],
                target_subtype=target_subtype,
                top_n=top_n,
            )

            candidates = build_candidate_list(
                con, anchor['container_id'], anchor['container_name'], 'divisions_area',
                num_candidates=10, difficulty="hard"
            )

            question = random.choice(question_hint_pool).format(
                top_n=top_n,
                target_subtype=target_subtype,
                anchor_name=anchor['container_name'],
            )
        
    elif template.family == "chained":
        # chained_12/13: country-level coastal/landlocked via adjacency.
        # The SQL uses ST_Touches (not containment), so bypass the containment
        # pair sampling and use adjacency_pairs with country-level anchors.
        if template.template_id in {"chained_12", "chained_13"}:
            adj_df = tables.get('adjacency_pairs', pd.DataFrame())
            country_adj = (
                adj_df[
                    (adj_df['anchor_subtype'] == 'country')
                    & (adj_df['target_subtype'] == 'country')
                ]
                if not adj_df.empty else pd.DataFrame()
            )
            if country_adj.empty:
                return None
            pair = sample_adjacency_anchor(country_adj)
            if not pair:
                return None

            sql = template.sql_template.format(anchor_id=pair['anchor_id'])
            candidates = build_candidate_list(
                con, pair['anchor_id'], pair['anchor_name'], 'divisions_area',
                num_candidates=10, difficulty="hard"
            )
            question = random.choice(template.question_hints).format(
                anchor_name=pair['anchor_name']
            )
            anchor = {'id': pair['anchor_id'], 'name': pair['anchor_name']}

        else:
            # Use pre-filtered coastal/landlocked containment pairs so the SQL
            # verification step doesn't constantly return empty results.
            _COASTAL_CHAINED = {"chained_01", "chained_06", "chained_10"}
            _LANDLOCKED_CHAINED = {"chained_02", "chained_07", "chained_11"}
            if template.template_id in _COASTAL_CHAINED:
                table_key = 'coastal_containment_pairs'
            elif template.template_id in _LANDLOCKED_CHAINED:
                table_key = 'landlocked_containment_pairs'
            else:
                table_key = 'containment_pairs'

            df = tables.get(table_key, tables['containment_pairs'])

            # When the template pins a target_subtype (e.g. chained_06 wants
            # counties), only consider pairs whose contained entity already
            # matches — guarantees the sampled container holds at least one
            # entity of the right subtype so the SQL filter returns rows.
            if template.target_subtype:
                df = df[df['contained_subtype'] == template.target_subtype]

            # chained_10/11 additionally need a country-level container so
            # phrasings like "coastal states of India" line up.
            if template.template_id in {"chained_10", "chained_11"}:
                df = df[df['container_subtype'] == 'country']

            anchor = sample_containment_anchor(df)
            if not anchor:
                return None

            target_subtype = template.target_subtype or anchor.get('contained_subtype', 'county')

            sql = template.sql_template.format(
                anchor_id=anchor['container_id'],
                target_subtype=target_subtype,
            )

            candidates = build_candidate_list(
                con, anchor['container_id'], anchor['container_name'], 'divisions_area',
                num_candidates=10, difficulty="hard"
            )

            question = random.choice(template.question_hints).format(
                anchor_name=anchor['container_name'],
                target_subtype=target_subtype,
            )

    elif template.family == "multi_adjacency":
        # Use common_neighbor_pairs so anchor1 and anchor2 are guaranteed to
        # share at least one touching neighbour — SQL will return non-empty.
        # Filter by both anchor subtypes AND shared_neighbor_subtype so the
        # sampled pair is geographically coherent with the template intent:
        #   multi_adj_01: region anchors → region result
        #   multi_adj_02: country anchors → country result
        #   multi_adj_03: region anchors → county result
        _MULTI_ADJ_ANCHOR_SUBTYPES = {
            "multi_adj_01": ("region", "region"),
            "multi_adj_02": ("country", "country"),
            "multi_adj_03": ("region", "region"),
        }
        cn_df = tables.get('common_neighbor_pairs', pd.DataFrame())
        if cn_df.empty:
            return None
        if template.target_subtype and 'shared_neighbor_subtype' in cn_df.columns:
            filtered = cn_df[cn_df['shared_neighbor_subtype'] == template.target_subtype]
            if not filtered.empty:
                cn_df = filtered
        if template.template_id in _MULTI_ADJ_ANCHOR_SUBTYPES and 'anchor_subtype_1' in cn_df.columns:
            a1_st, a2_st = _MULTI_ADJ_ANCHOR_SUBTYPES[template.template_id]
            filtered = cn_df[
                (cn_df['anchor_subtype_1'] == a1_st) &
                (cn_df['anchor_subtype_2'] == a2_st)
            ]
            if not filtered.empty:
                cn_df = filtered
        row = cn_df.sample(n=1).iloc[0]
        anchor1 = {'id': row['anchor_id_1'], 'name': row['anchor_name_1'], 'source': 'divisions_area'}
        anchor2 = {'id': row['anchor_id_2'], 'name': row['anchor_name_2'], 'source': 'divisions_area'}

        target_subtype = template.target_subtype or row.get('shared_neighbor_subtype', 'region')

        sql = template.sql_template.format(
            anchor_id_1=anchor1['id'],
            anchor_id_2=anchor2['id'],
            target_subtype=target_subtype,
        )

        candidates1 = build_candidate_list(
            con, anchor1['id'], anchor1['name'], 'divisions_area',
            num_candidates=5, difficulty="medium"
        )
        candidates2 = build_candidate_list(
            con, anchor2['id'], anchor2['name'], 'divisions_area',
            num_candidates=5, difficulty="medium"
        )
        candidates = _merge_candidate_lists(candidates1, candidates2)

        question = random.choice(template.question_hints).format(
            anchor_1_name=anchor1['name'],
            anchor_2_name=anchor2['name'],
            target_subtype=target_subtype,
        )

    elif template.family == "difference":
        if template.anchor_source == "mixed":
            # divisions_area anchor differenced against a natural_earth feature.
            # Use cross_source_relations so the pair is guaranteed to intersect
            # (ST_Difference on non-intersecting geometries is always equal to
            # the original geometry — a trivial and uninformative sample).
            cs_anchor = sample_cross_source_anchor(
                tables.get('cross_source_relations', pd.DataFrame()),
                natural_subtypes=_NE_TEMPLATE_SUBTYPES.get(template.template_id),
            )
            if not cs_anchor:
                return None
            anchor = {
                'id':   cs_anchor['division_id'],
                'name': cs_anchor['division_name'],
                'source': 'divisions_area',
            }
            clip_feature = {
                'id':   cs_anchor['natural_id'],
                'name': cs_anchor['natural_name'],
                'source': 'natural_earth',
            }

            sql = template.sql_template.format(
                        anchor_id=anchor['id'],
                clip_feature_id=clip_feature['id'],
            )
            question = random.choice(template.question_hints).format(
                anchor_name=anchor['name'],
                clip_feature_name=clip_feature['name'],
            )
            # Build candidates for BOTH anchors — model must see both IDs
            # to correctly assign anchor_id vs clip_feature_id in the SQL.
            div_cands = build_candidate_list(
                con, anchor['id'], anchor['name'], 'divisions_area',
                num_candidates=5, difficulty="hard",
            )
            ne_cands = build_candidate_list(
                con, clip_feature['id'], clip_feature['name'], 'natural_earth',
                num_candidates=5, difficulty="hard",
            )
            candidates = _merge_candidate_lists(div_cands, ne_cands, max_total=10)

        else:
            # Two divisions_area anchors: use both ends of a containment
            # pair so the contained entity is guaranteed to intersect the
            # container. ST_Difference(container, contained) yields the
            # portion of the container outside the contained piece.
            pair = sample_containment_anchor(tables['containment_pairs'])
            if not pair:
                return None

            anchor1 = {'id': pair['container_id'], 'name': pair['container_name']}
            anchor2 = {'id': pair['contained_id'], 'name': pair['contained_name']}

            sql = template.sql_template.format(
                anchor_id_1=anchor1['id'],
                anchor_id_2=anchor2['id'],
            )

            candidates1 = build_candidate_list(
                con, anchor1['id'], anchor1['name'], 'divisions_area',
                num_candidates=5, difficulty="medium"
            )
            candidates2 = build_candidate_list(
                con, anchor2['id'], anchor2['name'], 'divisions_area',
                num_candidates=5, difficulty="medium"
            )
            candidates = _merge_candidate_lists(candidates1, candidates2)

            question = random.choice(template.question_hints).format(
                anchor_1_name=anchor1['name'],
                anchor_2_name=anchor2['name'],
            )

    elif template.family == "border_corridor":
        # Buffered border zone — needs two anchors that actually touch.
        pair = sample_adjacency_anchor(tables['adjacency_pairs'])
        if not pair:
            return None

        anchor1 = {'id': pair['anchor_id'], 'name': pair['anchor_name']}
        anchor2 = {'id': pair['target_id'], 'name': pair['target_name']}

        buffer_val = random.choice([5, 10, 25, 50])

        sql = template.sql_template.format(
            anchor_id_1=anchor1['id'],
            anchor_id_2=anchor2['id'],
            buffer_km=buffer_val,
        )

        candidates1 = build_candidate_list(
            con, anchor1['id'], anchor1['name'], 'divisions_area',
            num_candidates=5, difficulty="medium"
        )
        candidates2 = build_candidate_list(
            con, anchor2['id'], anchor2['name'], 'divisions_area',
            num_candidates=5, difficulty="medium"
        )
        candidates = _merge_candidate_lists(candidates1, candidates2)

        question = random.choice(template.question_hints).format(
            anchor_1_name=anchor1['name'],
            anchor_2_name=anchor2['name'],
            buffer_km=buffer_val,
        )

    elif template.family == "window_function":
        anchor = sample_random_entity(
            con,
            tables['divisions_area_inventory'],
            'divisions_area',
            subtypes={'country'},
        )
        if not anchor:
            return None

        country = anchor.get('country') or 'US'
        target_subtype = template.target_subtype or 'county'

        sql = template.sql_template.format(
            country=country,
            target_subtype=target_subtype,
        )

        candidates = build_candidate_list(
            con, anchor['id'], anchor['name'], 'divisions_area',
            num_candidates=10, difficulty="hard"
        )

        question = random.choice(template.question_hints).format(
            anchor_name=anchor['name'],
            target_subtype=target_subtype,
        )

    elif template.family == "attribute_filter":
        anchor = sample_random_entity(
            con,
            tables['divisions_area_inventory'],
            'divisions_area',
            subtypes={'country'},
        )
        if not anchor:
            return None

        country = anchor.get('country') or 'US'
        target_subtype = template.target_subtype or 'region'

        sql = template.sql_template.format(
            country=country,
            target_subtype=target_subtype,
        )

        candidates = build_candidate_list(
            con, anchor['id'], anchor['name'], 'divisions_area',
            num_candidates=10, difficulty="medium"
        )

        question = random.choice(template.question_hints).format(
            anchor_name=anchor['name'],
            target_subtype=target_subtype,
            country=country,
        )

    else:
        # Skip unsupported families
        return None

    # Execute SQL to verify
    try:
        result = con.execute(_for_execution(sql)).fetchdf()
        if result.empty:
            return None
    except Exception as e:
        # Errors are tracked in worker return, no need to print
        return None

    # Collect every anchor ID that appears in the generated SQL so we can
    # mark them as the "selected" candidates in the training sample.
    _multi_anchor_families = {"set_operations", "multi_adjacency", "difference", "border_corridor", "buffer"}

    # Mixed partial_selection (partial_05) and mixed difference (diff_02) each
    # have two anchors from different sources — both must be marked selected.
    _is_mixed_two_anchor = (
        template.anchor_source == "mixed" and template.num_anchors == 2
    )

    if template.family in _multi_anchor_families and template.num_anchors >= 2:
        anchor_ids: set = set()
        for var in ("anchor1", "anchor2", "anchor3"):
            obj = locals().get(var)
            if obj:
                anchor_ids.add(obj.get("id", ""))
        if "anchors" in locals():
            for a in locals()["anchors"]:
                if a:
                    anchor_ids.add(a.get("id", ""))
        selected_candidate_ids = [c.candidate_id for c in candidates if c.id in anchor_ids]

    elif _is_mixed_two_anchor:
        # partial_05 / diff_02: anchor (division) + clip_feature (natural_earth)
        mixed_ids = {anchor.get("id", ""), clip_feature.get("id", "")}
        selected_candidate_ids = [c.candidate_id for c in candidates if c.id in mixed_ids]

    else:
        anchor_id_to_find = (
            anchor.get('anchor_id')
            or anchor.get('container_id')
            or anchor.get('natural_id')
            or anchor.get('id')
        )
        selected_candidate_ids = [c.candidate_id for c in candidates if c.id == anchor_id_to_find]

    question, surface_variants = _diversify_question_surface(question, template.family)

    return TrainingSample(
        id=sample_id,
        question=question,
        candidates=candidates,
        target={
            "selected_candidates": selected_candidate_ids,
            "sql": sql,
        },
        metadata={
            "task_family": template.family,
            "sql_difficulty": template.sql_difficulty,
            "grounding_difficulty": "medium",
            "template_id": template.template_id,
            "num_candidates": len(candidates),
            "anchor_source": template.anchor_source,
            "sql_verified": True,
            "surface_variants": surface_variants,
        }
    )


def generate_sample_batch_worker(args):
    """Worker function that processes a batch of work items with a single DuckDB connection.
    
    Initializes DuckDB, spatial extension, templates module, and relation tables
    ONCE per batch, then processes all items sequentially.
    """
    from pathlib import Path
    
    work_items, intermediate_dir_str = args
    
    # Convert string back to Path
    intermediate_dir = Path(intermediate_dir_str)
    
    # Initialize DuckDB ONCE for the entire batch
    con = duckdb.connect()
    con.execute("SET enable_progress_bar=false")
    con.execute("INSTALL spatial")
    con.execute("LOAD spatial")
    
    # Load relation tables ONCE
    tables = load_relation_tables(intermediate_dir, quiet=True)
    
    # Process all items in batch
    results = []
    for family, template_dict, sample_id, _ in work_items:
        # Reconstruct template from dict (sql_templates is already imported at module level)
        template = sql_templates.SQLTemplate(**template_dict)
        try:
            sample = generate_template_based_sample(con, template, tables, sample_id)
            if sample:
                results.append((sample, family, template.template_id, None))
            else:
                results.append((None, family, template.template_id, "Empty result"))
        except Exception as e:
            results.append((None, family, template_dict.get('template_id', 'unknown'), str(e)))
    
    con.close()
    return results


def generate_batch_core(
    work_items: List[tuple],
    intermediate_dir: str,
) -> List[Dict[str, Any]]:
    """Standalone batch worker usable from Modal or any remote context.
    
    Data paths are resolved via GAZET_DATA_DIR env var (set in Modal image).
    
    Args:
        work_items: List of (family, template_dict, sample_id, _) tuples
        intermediate_dir: Path to intermediate dir with relation parquets
        
    Returns:
        List of dicts with keys: sample (dict or None), family, template_id, error
    """
    from pathlib import Path as _Path
    intermediate = _Path(intermediate_dir)
    
    con = duckdb.connect()
    con.execute("SET enable_progress_bar=false")
    con.execute("INSTALL spatial")
    con.execute("LOAD spatial")
    
    tables = load_relation_tables(intermediate, quiet=True)
    
    results = []
    for family, template_dict, sample_id, _ in work_items:
        template = sql_templates.SQLTemplate(**template_dict)
        try:
            sample = generate_template_based_sample(con, template, tables, sample_id)
            if sample:
                results.append({
                    "sample": sample.model_dump(),
                    "family": family,
                    "template_id": template.template_id,
                    "error": None,
                })
            else:
                results.append({
                    "sample": None,
                    "family": family,
                    "template_id": template.template_id,
                    "error": "Empty result",
                })
        except Exception as e:
            results.append({
                "sample": None,
                "family": family,
                "template_id": template_dict.get('template_id', 'unknown'),
                "error": str(e),
            })
    
    con.close()
    return results


def prepare_work_items(
    target_counts: Dict[str, int],
    retry_multiplier: int = 2,
    start_counter: int = 1,
    intermediate_dir_str: str = "",
) -> List[tuple]:
    """Prepare shuffled work items for sample generation.
    
    Returns list of (family, template_dict, sample_id, intermediate_dir_str) tuples.
    Reusable by both local main() and Modal orchestrator.
    """
    work_items = []
    sample_counter = start_counter

    for family, target_count in target_counts.items():
        if target_count == 0:
            continue

        family_templates = [t for t in TEMPLATES if t.family == family]
        if not family_templates:
            print(f"No templates found for {family}, skipping...")
            continue

        # Distribute target evenly across templates so every template_id gets
        # a guaranteed share. Uniform random choice previously let rare
        # variants like partial_05 / diff_02 get under-represented or dropped
        # entirely when their mixed-source branch hit transient failures.
        n_tpl = len(family_templates)
        per_tpl = target_count // n_tpl
        remainder = target_count % n_tpl

        for i, template in enumerate(family_templates):
            count = per_tpl + (1 if i < remainder else 0)
            template_dict = {
                'template_id': template.template_id,
                'family': template.family,
                'sql_difficulty': template.sql_difficulty,
                'anchor_source': template.anchor_source,
                'num_anchors': template.num_anchors,
                'sql_template': template.sql_template,
                'question_hints': template.question_hints,
                'target_subtype': template.target_subtype,
                'requires_buffer': template.requires_buffer,
                'requires_aggregation': template.requires_aggregation
            }
            for _ in range(count * retry_multiplier):
                work_items.append((
                    family,
                    template_dict,
                    f"sample_{sample_counter:06d}",
                    intermediate_dir_str,
                ))
                sample_counter += 1

    random.shuffle(work_items)
    return work_items


def main():
    """Generate training samples."""
    global TARGET_COUNTS, MAX_WORKERS, RETRY_MULTIPLIER, APPEND_MODE
    
    # Setup paths
    script_dir = Path(__file__).parent
    intermediate_dir = script_dir.parent / "intermediate"
    output_dir = script_dir.parent / "output"
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load relation tables once to check availability
    print("Loading relation tables...")
    tables = load_relation_tables(intermediate_dir, quiet=False)
    
    # Use configured target counts or defaults
    if TARGET_COUNTS is None:
        target_counts = {
            'direct_lookup':    100,
            'adjacency':        150,
            'multi_adjacency':   75,
            'containment':      100,
            'intersection':     100,
            'buffer':           100,
            'chained':          150,
            'difference':        75,
            'border_corridor':   75,
            'set_operations':   150,
            'partial_selection': 75,
            'aggregation':      100,
            'window_function':   75,
            'attribute_filter':  75,
        }
    else:
        target_counts = TARGET_COUNTS
    
    # Load existing samples if in append mode
    existing_samples = []
    existing_sample_ids = set()
    jsonl_file = output_dir / "dataset_raw.jsonl"
    
    if APPEND_MODE and jsonl_file.exists():
        print(f"\nAppend mode: Loading existing samples from {jsonl_file}")
        with open(jsonl_file, 'r') as f:
            for line in f:
                if line.strip():
                    sample_data = json.loads(line)
                    existing_samples.append(sample_data)
                    existing_sample_ids.add(sample_data['id'])
        print(f"  Found {len(existing_samples)} existing samples")
        
        # Determine starting sample counter
        max_existing_id = max([int(s['id'].split('_')[1]) for s in existing_samples if s['id'].startswith('sample_')], default=0)
        sample_counter = max_existing_id + 1
    else:
        sample_counter = 1
    
    # Prepare work items using shared helper
    work_items = prepare_work_items(
        target_counts=target_counts,
        retry_multiplier=RETRY_MULTIPLIER,
        start_counter=sample_counter,
        intermediate_dir_str=str(intermediate_dir),
    )
    starting_sample_counter = sample_counter
    
    # Partition work items into batches (one per worker)
    num_workers = min(MAX_WORKERS, len(work_items))
    if num_workers == 0:
        print("No work items to process")
        return
    batch_size = (len(work_items) + num_workers - 1) // num_workers
    batches = []
    for i in range(0, len(work_items), batch_size):
        batch = work_items[i:i + batch_size]
        batches.append((batch, str(intermediate_dir)))
    
    # Generate samples in parallel (one batch per worker)
    active_families = len([f for f in target_counts.values() if f > 0])
    print(f"\nGenerating {len(work_items)} samples across {active_families} families...")
    print(f"  Split into {len(batches)} batches of ~{batch_size} items (1 DuckDB init per batch)")
    if APPEND_MODE and existing_samples:
        print(f"Appending: starting from sample_{starting_sample_counter:03d}")
    
    all_samples = []
    family_progress = {f: {'success': 0, 'failed': 0} for f in target_counts.keys() if target_counts[f] > 0}
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit one batch per worker
        futures = {executor.submit(generate_sample_batch_worker, batch): i for i, batch in enumerate(batches)}
        
        # Collect results as batches complete
        batches_done = 0
        for future in as_completed(futures):
            try:
                batch_results = future.result()
                for sample, family, template_id, error in batch_results:
                    if sample:
                        all_samples.append(sample)
                        family_progress[family]['success'] += 1
                    else:
                        family_progress[family]['failed'] += 1
            except Exception as e:
                print(f"\n  Batch failed: {e}")
            
            batches_done += 1
            total_done = sum(p['success'] + p['failed'] for p in family_progress.values())
            print(f"\r  Progress: {total_done}/{len(work_items)} samples ({batches_done}/{len(batches)} batches) ", end='', flush=True)
        
        print()  # New line after progress
    
    # Show distribution (keep all samples, no filtering)
    print("\nResults by family:")
    for family in sorted(family_progress.keys()):
        success = family_progress[family]['success']
        failed = family_progress[family]['failed']
        target = target_counts.get(family, 0)
        total = success + failed
        success_rate = (success / total * 100) if total > 0 else 0
        print(f"  {family:20s}: {success:3d} success / {failed:3d} failed ({success_rate:5.1f}% success rate, target: {target})")
    
    # Save combined JSONL (skip individual JSON files for speed at scale)
    print(f"\nSaving {len(all_samples)} new samples...")
    if APPEND_MODE and existing_samples:
        # Append to existing dataset
        print(f"Appending to existing dataset ({len(existing_samples)} existing samples)")
        with open(jsonl_file, 'a') as f:
            for sample in all_samples:
                f.write(json.dumps(sample.model_dump()) + '\n')
        total_samples = len(existing_samples) + len(all_samples)
    else:
        # Overwrite dataset
        with open(jsonl_file, 'w') as f:
            for sample in all_samples:
                f.write(json.dumps(sample.model_dump()) + '\n')
        total_samples = len(all_samples)
    
    print(f"\nGenerated {len(all_samples)} new samples")
    print(f"Total dataset size: {total_samples} samples")
    print(f"  Dataset: {jsonl_file}")


if __name__ == "__main__":
    main()
