"""Value hints: real column values that match words in the question, looked up without a model.

The schema card shows only a few top values per column, so a question about "Fukushima" cannot tell the
student that the stored value is 'Fukushima Ken'. A hint line gives it:  jp_prefectures.nam: 'Fukushima Ken'

Usage:
    python hints.py show TABLE[,TABLE] "question"     # print the hint lines
    python hints.py coverage FILE.jsonl               # how often hints supply the SQL's filter values
"""
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import harness

MAX_DISTINCT = 200_000   # skip columns with more distinct values than this (free text, ids of huge tables)
MAX_VALUE_LEN = 60       # skip long values (descriptions, comma-separated lists)
MAX_CATEGORY = 1000      # columns with more distinct values hold names; fewer, categories
MAX_PARTIAL = 6          # an n-gram matching more values than this in a column is too vague to be a hint
PER_GRAM, PER_COLUMN, MAX_LINES = 3, 6, 8  # partial hits per n-gram, values per line, lines per prompt
STOP = set("""a an and are around as at between by city cities count each every for from give have how in inside into is
km list many map me metres meters name named near nearest number of on or over per place places show than that the their
them there these to top under what where which with within""".split())


def norm(s, lower=True):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower() if lower else s
    return " ".join(re.sub(r"[_\-/,.;:()\[\]'\"`]", " ", s).split())


class TableIndex:
    """Distinct values of the text columns of one table, by normalised value and by token."""

    def __init__(self, c, table):
        self.cols = {}
        for name, typ, *_ in c.execute(f"DESCRIBE {table}").fetchall():
            if typ == "VARCHAR":
                expr = f'"{name}"'
            elif typ.startswith('STRUCT("primary" VARCHAR'):  # Overture names
                expr, name = f'"{name}"."primary"', f"{name}.primary"
            else:
                continue
            vals = [r[0] for r in c.execute(
                f"SELECT DISTINCT {expr} FROM {table} WHERE {expr} IS NOT NULL AND length({expr}) <= {MAX_VALUE_LEN} "
                f"LIMIT {MAX_DISTINCT + 1}").fetchall()]
            if not vals or len(vals) > MAX_DISTINCT or sum(v.startswith("http") for v in vals[:200]) > 20:
                continue
            exact, by_token = defaultdict(list), defaultdict(set)
            for v in vals:
                n = norm(v)
                if not n:
                    continue
                exact[n].append(v)
                for tok in set(n.split()):
                    by_token[tok].add(n)
            self.cols[name] = (exact, by_token, len(vals) <= MAX_CATEGORY)

    def match(self, grams):
        """-> {column: [(score, value)]}; an exact value match beats a value that merely contains the n-gram."""
        out = {}
        for col, (exact, by_token, is_category) in self.cols.items():
            found, used = {}, set()
            for g, toks, span, proper in grams:  # longest n-grams first
                if span & used:
                    continue  # already explained by a longer match in this column
                if len(toks) == 1 and not proper and not is_category:
                    continue  # a common word that happens to be someone's name ('South', 'Total')
                cands = []
                for form in (g, g[:-1]) if g.endswith("s") else (g,):  # 'travel services' -> travel_service
                    if form in exact:
                        cands = [(10 + len(toks), form)]
                        break
                if not cands and proper:
                    sets = [by_token.get(t) for t in toks]
                    if all(sets):
                        hits = [n for n in set.intersection(*sets) if f" {g} " in f" {n} "]
                        if 0 < len(hits) <= MAX_PARTIAL:
                            cands = [(len(toks), n) for n in sorted(hits, key=len)[:PER_GRAM]]
                if cands:
                    used |= span
                    for score, n in cands:
                        for v in exact[n][:2]:
                            found[v] = max(found.get(v, 0), score)
            if found:
                out[col] = sorted(((s, v) for v, s in found.items()), key=lambda sv: (-sv[0], len(sv[1]), sv[1]))[:PER_COLUMN]
        return out


_INDEX = {}


def _grams(question, max_n=4):
    """N-grams of the question, longest first, as (text, tokens, positions, proper).

    `proper` gates the loose matches. An n-gram may match part of a value ('Fukushima' -> 'Fukushima Ken') only when it looks like a proper noun:
    it holds a capitalised word that is not the first word of the question. Otherwise 'square', 'region' or
    'power plants' pull in noise. A question written entirely in lower case gives no such signal, so there
    every n-gram may match partially.
    """
    cased = norm(question, lower=False).split()
    toks = [t.lower() for t in cased]
    no_case = not any(t[0].isupper() for t in cased)
    grams = []
    for n in range(max_n, 0, -1):
        for i in range(len(toks) - n + 1):
            part = toks[i:i + n]
            if n == 1 and (len(part[0]) < 4 or part[0] in STOP or part[0].replace(".", "").isdigit()):
                continue
            if part[0] in STOP or part[-1] in STOP or all(t.isdigit() for t in part):
                continue
            proper = no_case or any(cased[j][0].isupper() for j in range(max(i, 1), i + n))
            grams.append((" ".join(part), part, frozenset(range(i, i + n)), proper))
    return grams


def value_hints(c, tables, question, max_lines=MAX_LINES):
    """Hint lines `table.column: 'v1', 'v2'` for the tables of a prompt, best matches first."""
    grams, scored = _grams(question), []
    for t in tables:
        if t not in _INDEX:
            _INDEX[t] = TableIndex(c, t)
        for col, hits in _INDEX[t].match(grams).items():
            vals = ", ".join("'" + v.replace("'", "''") + "'" for _, v in hits)
            scored.append((-hits[0][0], f"{t}.{col}: {vals}"))
    return [line for _, line in sorted(scored)[:max_lines]]


def _filter_literals(sql):
    """String literals the SQL compares a column against (=, <>, IN, LIKE)."""
    out = []
    for lit in re.findall(r"'((?:[^']|'')*)'", sql):
        if re.search(r"(=|<>|!=|\bIN\s*\([^)]*|LIKE)\s*'" + re.escape(lit) + "'", sql, flags=re.IGNORECASE):
            out.append(lit.replace("''", "'").strip("%"))
    return [v for v in out if re.search(r"[^\W\d_]{2}", v)]


def coverage(path):
    c = harness.con()
    cards = {t: harness.schema_card(c, t) for t in harness.TABLE_DOCS}
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    n_hidden = n_solved = n_hinted = n_lines = 0
    missed = []
    for r in rows:
        hints = value_hints(c, r["tables"], r["question"])
        n_hinted += bool(hints)
        n_lines += len(hints)
        seen = norm(r["question"]) + "\n" + norm("\n".join(cards[t] for t in r["tables"]))
        hidden = [v for v in _filter_literals(r["sql"]) if norm(v) not in seen]
        if hidden:
            n_hidden += 1
            text = norm("\n".join(hints))
            ok = all(norm(v) in text for v in hidden)
            n_solved += ok
            if not ok and len(missed) < 25:
                missed.append((r["question"], hidden, hints))
    print(f"{len(rows)} pairs; {n_hinted} get hints ({n_lines / max(1, n_hinted):.1f} lines on average)")
    print(f"filter value in neither question nor card: {n_hidden}; all such values supplied by hints: {n_solved}")
    for q, hidden, hints in missed:
        print(f"\n  Q: {q}\n  missing: {hidden}\n  hints: {hints}")


if __name__ == "__main__":
    if sys.argv[1] == "show":
        print("\n".join(value_hints(harness.con(), sys.argv[2].split(","), sys.argv[3])) or "(no hints)")
    elif sys.argv[1] == "coverage":
        coverage(sys.argv[2])
