"""The student model's input format: one place for the prompt, shared by evaluation and the training export.

No imports beyond the standard library, so the demo app can copy this file.

Order is schema cards -> value hints (optional) -> question, so llama.cpp can reuse a cached schema prefix.
"""
# For models that were not fine-tuned on this data (reference ceilings, zero-shot baselines): rules spelled out.
# Also the solver prompt of consistency.py.
SYSTEM_RULES = """You write DuckDB SQL. The spatial extension is loaded and `SET geometry_always_xy = true` is active (lon/lat order).
Geometries are EPSG:4326 lon/lat degrees. DuckDB spatial is not PostGIS: no SRID arguments, ST_Transform takes CRS strings
(ST_Transform(geom, 'EPSG:4326', 'EPSG:3763', always_xy := true)), no GEOGRAPHY type, ST_Collect takes a list.
For metres use ST_Distance_Sphere, ST_Area_Spheroid, ST_Length_Spheroid, or a local metric CRS; ST_Distance_Spheroid and
ST_DWithin_Spheroid need geometry::GEOMETRY casts. List lambdas are written lambda x: x + 1, not x -> x + 1. Return geometries with ST_AsGeoJSON only when the user wants shapes.
Answer with a single read-only SELECT statement and nothing else."""
# For the fine-tuned student: the rules are learned, so the prompt stays short.
SYSTEM_MINIMAL = "You write DuckDB spatial SQL. Answer with a single read-only SELECT statement and nothing else."
SYSTEMS = {"rules": SYSTEM_RULES, "minimal": SYSTEM_MINIMAL}

HINTS_HEADER = "VALUES matching the question:"


def user_prompt(cards, tables, question, hints=None):
    """`hints` is a list of 'table.column: 'v1', 'v2'' lines from hints.value_hints, or None / empty for no block."""
    parts = ["\n\n".join(cards[t] for t in tables)]
    if hints:
        parts.append(HINTS_HEADER + "\n" + "\n".join("  " + h for h in hints))
    parts.append(f"Question: {question}")
    return "\n\n".join(parts)
