"""
Export validated dataset to train/val/test splits.

Produces two task datasets from the same source samples:

1. SQL generation  (prompt = question + candidates CSV, completion = SQL)
2. Place extraction (prompt = question only, completion = PlacesResult JSON)

Place extraction pairs are derived automatically: for each SQL sample the
selected_candidates give us the correct place names, subtypes, and country
codes that the extractor should return.

Output layout (all paths relative to dataset/):
    output/runs/{run_name}/sql/train.jsonl
    output/runs/{run_name}/sql/val.jsonl
    output/runs/{run_name}/sql/test.jsonl
    output/runs/{run_name}/places/train.jsonl
    output/runs/{run_name}/places/val.jsonl
    output/runs/{run_name}/places/test.jsonl
    output/runs/{run_name}/stats.json
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_samples(jsonl_path: Path) -> List[Dict[str, Any]]:
    samples = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def load_run_name(config_path: Optional[Path]) -> str:
    if config_path and config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("run_name", "default")
    return "default"


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def stratified_split(
    samples: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split stratified by task_family so every family is represented in each split."""
    random.seed(seed)
    by_family: Dict[str, List] = defaultdict(list)
    for s in samples:
        by_family[s["metadata"]["task_family"]].append(s)

    train, val, test = [], [], []
    for family_samples in by_family.values():
        random.shuffle(family_samples)
        n = len(family_samples)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train.extend(family_samples[:n_train])
        val.extend(family_samples[n_train : n_train + n_val])
        test.extend(family_samples[n_train + n_val :])

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    return train, val, test


# ---------------------------------------------------------------------------
# SQL generation format
# Conversational prompt-completion: model sees system + user, generates SQL.
# ---------------------------------------------------------------------------

_SQL_SYSTEM = (
    "You are a text to SQL query translator that helps in natural language geocoding."
)

_CANDIDATES_COLS = [
    "source", "id", "name", "subtype", "country", "region",
    "admin_level", "similarity",
]

_SCHEMA = """1. divisions_area  -- Overture polygon/multipolygon admin boundaries
   query: read_parquet('divisions_area')
   columns:
     id VARCHAR              -- unique feature id
     names STRUCT("primary" VARCHAR, ...)
     country VARCHAR         -- ISO 3166-1 alpha-2
     subtype VARCHAR         -- country | region | dependency | county | localadmin |
                               locality | macrohood | neighborhood | microhood
     class VARCHAR
     region VARCHAR
     admin_level INTEGER
     division_id VARCHAR
     is_land BOOLEAN
     is_territorial BOOLEAN
     geometry GEOMETRY       -- WGS-84 polygon/multipolygon (spatial ext loaded)

2. natural_earth  -- Natural Earth geography polygons (oceans, seas, rivers, terrain)
   query: read_parquet('natural_earth')
   columns:
     id VARCHAR              -- unique feature id prefixed 'ne_'
     names STRUCT("primary" VARCHAR, ...)
     country VARCHAR
     subtype VARCHAR         -- e.g. 'ocean', 'sea', 'bay', 'Terrain area', 'Island group'
     class VARCHAR
     region VARCHAR
     admin_level INTEGER
     is_land BOOLEAN
     is_territorial BOOLEAN
     geometry GEOMETRY       -- WGS-84 polygon/multipolygon (spatial ext loaded)

The candidates table has a 'source' column: 'divisions_area' or 'natural_earth'.
Use read_parquet('divisions_area') or read_parquet('natural_earth') accordingly.
Use ST_AsGeoJSON(geometry) for all geometry outputs."""


def _candidates_csv(candidates: List[Dict]) -> str:
    import io
    import csv
    rows = []
    for c in candidates:
        row = {col: c.get(col, "") for col in _CANDIDATES_COLS if col in c}
        rows.append(row)
    if not rows:
        return ""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=[k for k in _CANDIDATES_COLS if k in rows[0]])
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().strip()


def sample_to_sql_pair(sample: Dict[str, Any]) -> Optional[Dict]:
    """Convert a raw sample to a conversational prompt-completion pair for SQL generation."""
    sql = sample.get("target", {}).get("sql", "").strip()
    if not sql:
        return None

    user_content = (
        "GIVEN the <SCHEMA_DETAILS>, <CANDIDATES> and <USER_QUERY>, "
        "generate the corresponding SQL command to retrieve the desired geometry.\n\n"
        f"<SCHEMA_DETAILS>\n{_SCHEMA}\n</SCHEMA_DETAILS>\n\n"
        f"<CANDIDATES>\n{_candidates_csv(sample.get('candidates', []))}\n</CANDIDATES>\n\n"
        f"<USER_QUERY>\n{sample['question']}\n</USER_QUERY>"
    )

    return {
        "prompt": [
            {"role": "system", "content": _SQL_SYSTEM},
            {"role": "user",   "content": user_content},
        ],
        "completion": [
            {"role": "assistant", "content": sql},
        ],
        "metadata": sample.get("metadata", {}),
    }


# ---------------------------------------------------------------------------
# Place extraction format
# Derived from the same SQL samples: selected_candidates → PlacesResult JSON.
# ---------------------------------------------------------------------------

_PLACE_SYSTEM = (
    "You are a geographic entity extractor. "
    "Extract place names from the query and return valid JSON only."
)

# Overture division subtypes — used to filter out natural_earth candidates
# from the place extraction output (NE features don't have these subtypes).
_DIVISION_SUBTYPES = {
    "country", "region", "dependency", "county", "localadmin",
    "locality", "macrohood", "neighborhood", "microhood",
}


def _candidate_to_place(c: Dict) -> Optional[Dict]:
    """Convert a selected candidate to a Place dict for PlacesResult."""
    name = c.get("name", "").strip()
    if not name:
        return None

    place: Dict[str, Any] = {"place": name}

    subtype = c.get("subtype", "")
    if subtype in _DIVISION_SUBTYPES:
        place["subtype"] = subtype

    country = c.get("country", "")
    if country and len(country) == 2:
        place["country"] = country

    return place


def sample_to_place_pair(sample: Dict[str, Any]) -> Optional[Dict]:
    """Convert a raw sample to a conversational prompt-completion pair for place extraction.

    Uses selected_candidates to determine the correct PlacesResult output.
    Skips samples where no valid places can be derived.
    """
    selected_ids = set(sample.get("target", {}).get("selected_candidates", []))
    if not selected_ids:
        return None

    id_to_candidate = {c["candidate_id"]: c for c in sample.get("candidates", [])}
    places = []
    seen_names: set = set()

    for cid in selected_ids:
        c = id_to_candidate.get(cid)
        if not c:
            continue
        place = _candidate_to_place(c)
        if place and place["place"].lower() not in seen_names:
            places.append(place)
            seen_names.add(place["place"].lower())

    if not places:
        return None

    completion_json = json.dumps({"places": places}, ensure_ascii=False)

    return {
        "prompt": [
            {"role": "system", "content": _PLACE_SYSTEM},
            {"role": "user",   "content": sample["question"]},
        ],
        "completion": [
            {"role": "assistant", "content": completion_json},
        ],
        "metadata": sample.get("metadata", {}),
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_jsonl(records: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def split_stats(samples: List[Dict]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for s in samples:
        counts[s.get("metadata", {}).get("task_family", "unknown")] += 1
    return dict(sorted(counts.items()))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: Optional[Path] = None) -> None:
    script_dir = Path(__file__).parent
    dataset_dir = script_dir.parent
    output_dir = dataset_dir / "output"

    run_name = load_run_name(config_path or dataset_dir / "config.yaml")

    validated_file = output_dir / "dataset_validated.jsonl"
    if not validated_file.exists():
        print(f"Error: {validated_file} not found. Run validate first.")
        sys.exit(1)

    run_dir = output_dir / "runs" / run_name
    sql_dir = run_dir / "sql"
    places_dir = run_dir / "places"

    print(f"Run name   : {run_name}")
    print(f"Output dir : {run_dir}")

    # Load
    print("\nLoading validated samples...")
    samples = load_samples(validated_file)
    print(f"  {len(samples):,} samples loaded")

    # Split once, reuse for both tasks
    print("\nSplitting 80 / 10 / 10 (stratified by task family)...")
    train_raw, val_raw, test_raw = stratified_split(samples)
    print(f"  train={len(train_raw):,}  val={len(val_raw):,}  test={len(test_raw):,}")

    # --- SQL generation ---
    print("\nBuilding SQL generation splits...")
    sql_stats: Dict = {}
    for split_name, raw in [("train", train_raw), ("val", val_raw), ("test", test_raw)]:
        pairs = [p for s in raw if (p := sample_to_sql_pair(s)) is not None]
        save_jsonl(pairs, sql_dir / f"{split_name}.jsonl")
        sql_stats[split_name] = {"total": len(pairs), "by_family": split_stats(pairs)}
        print(f"  sql/{split_name}.jsonl  — {len(pairs):,} pairs")

    # --- Place extraction ---
    print("\nBuilding place extraction splits...")
    place_stats: Dict = {}
    for split_name, raw in [("train", train_raw), ("val", val_raw), ("test", test_raw)]:
        pairs = [p for s in raw if (p := sample_to_place_pair(s)) is not None]
        save_jsonl(pairs, places_dir / f"{split_name}.jsonl")
        place_stats[split_name] = {"total": len(pairs), "by_family": split_stats(pairs)}
        print(f"  places/{split_name}.jsonl  — {len(pairs):,} pairs")

    # --- Stats ---
    stats = {
        "run_name": run_name,
        "total_samples": len(samples),
        "sql_generation": sql_stats,
        "place_extraction": place_stats,
    }
    stats_path = run_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nStats written to {stats_path}")
    print("\nDone. Training-ready files:")
    print(f"  SQL generation  : {sql_dir}/{{train,val,test}}.jsonl")
    print(f"  Place extraction: {places_dir}/{{train,val,test}}.jsonl")


if __name__ == "__main__":
    main()
