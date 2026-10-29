"""Schema cards, scenario prompts, and execution validation for generative text-to-SQL data.

Usage:
    python harness.py cards                 # print schema cards
    python harness.py prompts N SEED        # write N scenario prompts to prompts/
    python harness.py validate OUT.jsonl    # execute generated pairs, write results
"""
import json
import random
import re
import sys
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb

HERE = Path(__file__).parent
LAKE = HERE / "lake"

TABLE_DOCS = {
    "countries": "Natural Earth 1:50m country polygons with population/GDP attributes. EPSG:4326.",
    "populated_places": "Natural Earth 1:50m populated places (points). EPSG:4326.",
    "rivers": "Natural Earth 1:50m river and lake centerlines (linestrings). EPSG:4326.",
    "lakes": "Natural Earth 1:50m lake polygons. EPSG:4326.",
    "airports": "Natural Earth 1:10m airports (points). EPSG:4326.",
    "ports": "Natural Earth 1:10m ports (points). EPSG:4326.",
    "nyc_taxi_zones": "NYC TLC taxi zones (polygons), reprojected to EPSG:4326.",
    "nyc_taxi_trips": "Sample of NYC yellow taxi trips, Jan 2024. PULocationID/DOLocationID reference nyc_taxi_zones.location_id. No geometry.",
    "lisbon_places": "Overture Maps places (POIs, points) in Lisbon. EPSG:4326.",
    "lisbon_buildings": "Overture Maps building footprints (polygons) in Lisbon. EPSG:4326. height in metres.",
    "lisbon_roads": "Overture Maps road segments (linestrings) in Lisbon. EPSG:4326.",
    # Regionally and institutionally diverse sources (build_lake_world.py).
    "uk_police_crimes": "UK street-level crimes within one mile of central London, Apr-Jun 2024 (data.police.uk). Points. month is 'YYYY-MM'.",
    "uk_naptan_stops": "UK public transport access nodes (bus stops, stations) in London from NaPTAN. Points; easting/northing are British National Grid (EPSG:27700).",
    "uk_local_authorities": "UK local authority district boundaries, December 2023 (ONS). Polygons; lad23cd is the official code.",
    "eu_nuts_regions": "Eurostat NUTS 2021 statistical regions at levels 0-3 (levl_code), 1:20M. Polygons; nuts_id prefixes encode the hierarchy.",
    "paris_trees": "Street and park trees managed by the City of Paris (column names in French). Points; hauteurenm = height in m, circonferenceencm = girth in cm.",
    "paris_velib_stations": "Velib' bike-share station locations in Paris (French column names). Points.",
    "nl_municipalities": "Dutch municipalities (gemeenten) 2023 with CBS statistics (Dutch column names). Polygons; water='JA' rows are water areas.",
    "br_fire_hotspots": "Satellite-detected fire hotspots in Brazil over 7 days (INPE Queimadas, Portuguese column names). Points; frp = fire radiative power; some numeric fields are stored as text.",
    "br_sp_municipalities": "Municipalities of Sao Paulo state, Brazil (IBGE 2022, Portuguese column names). Polygons; AREA_KM2 is official area.",
    "bike_share_stations": "Bike-share stations from GBFS feeds of Buenos Aires and Mexico City (system_id). Points.",
    "jp_prefectures": "The 47 prefectures of Japan with Japanese and English names. Polygons.",
    "in_districts": "Districts of India, Census 2011 boundaries (DataMeet). Polygons.",
    "sg_parks": "Parks and playgrounds in Singapore (data.gov.sg). Points; X/Y are SVY21 metres (EPSG:3414), geometry is lon/lat.",
    "zm_health_facilities": "Geo-located health facilities in Zambia (World Bank). Points.",
    "usgs_earthquakes": "Global earthquakes of the past 30 days (USGS feed). Points; depth_km below surface, event_time UTC.",
    "power_plants": "Global Power Plant Database (WRI): ~35k plants with capacity_mw, primary_fuel and generation by year. Points.",
    "ourairports": "Global airports, heliports and seaplane bases (OurAirports). Points; id joins ourairports_runways.airport_ref.",
    "ourairports_runways": "Runways of OurAirports airports. No geometry; airport_ref references ourairports.id; lengths in feet.",
    "geonames_cities": "Cities with population > 15000 (GeoNames). Points; alternatenames is a comma-separated list.",
    "tokyo_places": "Overture Maps places (POIs, points) in central Tokyo, names often in Japanese. EPSG:4326.",
}

# Never used for training data: reserved for the held-out gold eval set (new schemas, regions and publishers).
HELD_OUT = {"uk_police_crimes", "nl_municipalities", "br_fire_hotspots", "br_sp_municipalities", "sg_parks",
            "zm_health_facilities", "power_plants", "tokyo_places"}

EVAL_BUNDLES = [
    ["uk_police_crimes"], ["uk_police_crimes", "uk_local_authorities"], ["nl_municipalities"],
    ["nl_municipalities", "eu_nuts_regions"], ["br_fire_hotspots"], ["br_fire_hotspots", "br_sp_municipalities"],
    ["br_sp_municipalities"], ["sg_parks"], ["zm_health_facilities"], ["zm_health_facilities", "geonames_cities"],
    ["power_plants"], ["power_plants", "countries"], ["power_plants", "eu_nuts_regions"], ["tokyo_places"],
]

# Table bundles that make sense together (single-table and multi-table); training tables only.
BUNDLES = [
    ["countries"], ["populated_places"], ["airports"], ["lisbon_places"], ["lisbon_buildings"],
    ["lisbon_roads"], ["nyc_taxi_trips"],
    ["countries", "populated_places"], ["countries", "airports"], ["countries", "rivers"],
    ["countries", "lakes"], ["airports", "populated_places"], ["ports", "countries"],
    ["nyc_taxi_zones", "nyc_taxi_trips"], ["lisbon_places", "lisbon_buildings"],
    ["lisbon_places", "lisbon_roads"], ["lisbon_buildings", "lisbon_roads"],
    ["countries", "rivers", "lakes"],
    ["uk_naptan_stops"], ["uk_naptan_stops", "uk_local_authorities"], ["uk_local_authorities", "geonames_cities"],
    ["eu_nuts_regions"], ["eu_nuts_regions", "geonames_cities"], ["eu_nuts_regions", "airports"],
    ["paris_trees"], ["paris_trees", "paris_velib_stations"], ["paris_velib_stations", "eu_nuts_regions"],
    ["bike_share_stations"], ["bike_share_stations", "geonames_cities"],
    ["jp_prefectures"], ["jp_prefectures", "usgs_earthquakes"], ["jp_prefectures", "geonames_cities"],
    ["in_districts"], ["in_districts", "geonames_cities"], ["in_districts", "ourairports"],
    ["usgs_earthquakes"], ["usgs_earthquakes", "countries"], ["usgs_earthquakes", "populated_places"],
    ["ourairports", "ourairports_runways"], ["ourairports", "countries"], ["ourairports", "geonames_cities"],
    ["geonames_cities"], ["geonames_cities", "countries"],
]
assert not any(t in HELD_OUT for b in BUNDLES for t in b), "held-out table in a training bundle"

SKILLS = [
    "spatial predicates (ST_Intersects / ST_Within / ST_Contains / ST_Touches) for filtering",
    "spatial join with aggregation (e.g. count/sum points per polygon)",
    "distance and nearest-neighbour queries (ST_Distance_Sphere / ST_DWithin, ORDER BY distance LIMIT k, LATERAL)",
    "measurement in real units (area/length via ST_Area_Spheroid, ST_Length_Spheroid, or ST_Transform to a metric CRS)",
    "geometry construction and overlay (ST_Buffer, ST_Union_Agg, ST_Intersection, ST_Difference, ST_ConvexHull, ST_Centroid)",
    "nested Overture-style data: struct fields, lists, UNNEST, list functions, JSON-like attributes",
    "window functions and per-group ranking (ROW_NUMBER, RANK, running totals) combined with spatial attributes",
    "analytical SQL: GROUP BY / HAVING / CTEs / CASE / date-time functions / QUALIFY",
    "bounding-box and envelope filtering (ST_MakeEnvelope, bbox comparisons, ST_Extent)",
    "output shaping: ST_AsGeoJSON, ST_AsText, ST_X/ST_Y, ROUND, string formatting of results",
]
DIFFICULTIES = ["easy", "medium", "hard"]
STYLES = [
    "terse keyword-style requests a GIS analyst types into a search box",
    "full natural sentences from a non-technical user who does not know column names",
    "precise analyst questions that mention units and thresholds",
    "questions phrased as tasks ('give me', 'list', 'map', 'compare')",
]


def con():
    c = duckdb.connect()
    c.execute("LOAD spatial")
    c.execute("SET geometry_always_xy = true")  # DuckDB 2.0: lon/lat axis order for spheroid fns
    for p in sorted(LAKE.glob("*.parquet")):
        if p.stem in TABLE_DOCS:
            # Sources label lon/lat as EPSG:4326, OGC:CRS84 or EPSG:4979; DuckDB 2.0 refuses to join across labels.
            cols = {r[0] for r in c.execute(f"DESCRIBE SELECT * FROM '{p}'").fetchall()}
            star = "* REPLACE (geometry::GEOMETRY::GEOMETRY('EPSG:4326') AS geometry)" if "geometry" in cols else "*"
            c.execute(f"CREATE VIEW {p.stem} AS SELECT {star} FROM '{p}'")
    return c


def _short(v, n=60):
    s = str(v)
    return s if len(s) <= n else s[: n - 3] + "..."


def schema_card(c, table):
    """Compact, model-readable description of one table, derived from the data."""
    n = c.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    lines = [f"TABLE {table}  -- {TABLE_DOCS[table]} rows={n}"]
    for name, typ, *_ in c.execute(f"DESCRIBE {table}").fetchall():
        hint = ""
        if typ.startswith("GEOMETRY"):
            gt = c.execute(f"SELECT list(DISTINCT ST_GeometryType(geometry)::VARCHAR) FROM {table}").fetchone()[0]
            ext = c.execute(f"SELECT [round(ST_XMin(e),3), round(ST_YMin(e),3), round(ST_XMax(e),3), round(ST_YMax(e),3)] FROM (SELECT ST_Extent_Agg(geometry) e FROM {table})").fetchone()[0]
            hint = f"types={gt} extent={_short(ext, 90)}"
        elif typ.startswith(("STRUCT", "MAP")) or typ.endswith("[]"):
            ex = c.execute(f'SELECT "{name}" FROM {table} WHERE "{name}" IS NOT NULL LIMIT 1').fetchone()
            hint = f"example={_short(ex[0] if ex else None, 160)}"
            typ = _short(typ, 600)
        elif typ in ("VARCHAR",):
            top = c.execute(
                f'SELECT "{name}", count(*) k FROM {table} WHERE "{name}" IS NOT NULL '
                f"GROUP BY 1 ORDER BY k DESC LIMIT 6"
            ).fetchall()
            distinct = c.execute(f'SELECT approx_count_distinct("{name}") FROM {table}').fetchone()[0]
            if distinct > 0.5 * n:  # id-like / free text: a couple of examples is enough
                hint = f"distinct~{distinct} e.g. {[_short(t[0], 40) for t in top[:2]]}"
            else:
                hint = f"distinct~{distinct} top={[_short(t[0], 30) for t in top]}"
        elif typ in ("BIGINT", "INTEGER", "DOUBLE", "FLOAT", "DECIMAL", "TIMESTAMP", "TIMESTAMP_NS") or typ.startswith("DECIMAL"):
            lo, hi = c.execute(f'SELECT min("{name}"), max("{name}") FROM {table}').fetchone()
            hint = f"range=[{lo}, {hi}]"
        elif typ == "BOOLEAN":
            hint = "true/false"
        lines.append(f"  {name} {typ}  -- {hint}")
    return "\n".join(lines)


def grounding(c, table, seed, n_rows=5, max_values=40):
    """Real values for the generator only (not the student): random rows and full low-cardinality value lists."""
    cols, name_col, values = [], None, []
    for name, typ, *_ in c.execute(f"DESCRIBE {table}").fetchall():
        if typ.startswith("GEOMETRY"):
            cols.append("round(ST_X(ST_Centroid(geometry)), 4) AS lon, round(ST_Y(ST_Centroid(geometry)), 4) AS lat")
        elif typ.startswith('STRUCT("primary"'):
            cols.append(f'"{name}"."primary" AS "{name}.primary"')
            name_col = name_col or f'"{name}"."primary"'
        elif typ.startswith(("STRUCT", "MAP")) or typ.endswith("[]") or name == "id":
            continue
        else:
            cols.append(f'"{name}"')
            if name == "name":
                name_col = '"name"'
            if typ == "VARCHAR":
                distinct = c.execute(f'SELECT count(DISTINCT "{name}") FROM {table}').fetchone()[0]
                if distinct <= max_values:
                    vals = c.execute(f'SELECT list(DISTINCT "{name}" ORDER BY "{name}") FROM {table} WHERE "{name}" IS NOT NULL').fetchone()[0]
                    values.append(f"  {name}: {vals}")
    where = f"WHERE {name_col} IS NOT NULL" if name_col else ""
    rows = c.execute(
        f"SELECT {', '.join(cols)} FROM (SELECT * FROM {table} {where}) USING SAMPLE reservoir({n_rows} ROWS) REPEATABLE ({seed})"
    )
    header = [d[0] for d in rows.description]
    lines = [f"REAL ROWS from {table} (random sample; centroid lon/lat):"]
    lines += ["  " + ", ".join(f"{h}={_short(v, 40)}" for h, v in zip(header, r, strict=True) if v is not None) for r in rows.fetchall()]
    if values:
        lines.append(f"ALL VALUES in {table}:")
        lines += values
    return "\n".join(lines)


GEN_PROMPT = """You are generating training data for a small text-to-SQL model that writes DuckDB SQL (with the spatial extension loaded) for geospatial questions.

Below are the tables available. Each is a DuckDB view with exactly the columns shown.

{cards}

Real data to ground the questions in (use these names, categories and coordinates as literals):

{grounding}

Write {k} diverse (question, sql) pairs for these tables.
- Focus skill: {skill}
- Difficulty: {difficulty}
- Question style: {style}

Rules:
- SQL must be a single read-only DuckDB SELECT (CTEs allowed) that runs on exactly these tables and columns. Use only DuckDB / DuckDB-spatial functions.
- Geometries are EPSG:4326 lon/lat degrees. Use spheroid functions or ST_Transform(geom, 'EPSG:4326', '<metric EPSG>', always_xy := true) when the question asks for metres/km. Note ST_Distance on 4326 returns degrees.
- Only use literal values (names, categories, codes) that appear in the real rows, value lists or top values above. The query should return at least one row.
- Do not name places, neighbourhoods or landmarks that are not in the data. If a question needs an area or point that is not a row in the data, give its coordinates explicitly in the question.
- Never measure distances, buffers or areas in EPSG:3857 (Web Mercator) or EPSG:6933; their scale is wrong away from the equator. Use the spheroid functions or a local metric CRS (UTM zone or national grid).
- Results must be deterministic: no current_date, now() or random(); whenever you use LIMIT, ORDER BY columns that make the order unique.
- Watch join fan-out: do not SUM a per-country (or per-parent) value after joining to many child rows.
- Return geometries as ST_AsGeoJSON(...) only when the user would want the shape; otherwise return attributes.
- Every question must be answerable from the data alone and must not mention column or table names unless the style says so.
- Make the {k} pairs differ in intent, not just in literal values.

DuckDB spatial is NOT PostGIS. `SET geometry_always_xy = true` is active (lon/lat order). Differences that matter:
- No SRID arguments and no ST_SetSRID: ST_MakeEnvelope(xmin, ymin, xmax, ymax), ST_GeomFromText(wkt), ST_Point(lon, lat).
- ST_Transform takes CRS strings, not integers: ST_Transform(geom, 'EPSG:4326', 'EPSG:3763', always_xy := true).
- No GEOGRAPHY type and no 'spheroid' / use_spheroid argument on ST_Distance or ST_Area. For metres use ST_Distance_Sphere / ST_Distance_Spheroid / ST_DWithin_Spheroid (POINT inputs only), ST_Area_Spheroid, ST_Length_Spheroid, or ST_Transform to a metric CRS. For distances involving lines or polygons, transform to a metric CRS first.
- ST_Collect takes a list: ST_Collect(list(geom)). Aggregates are ST_Union_Agg(geom) and ST_Extent_Agg(geom).
- Table geometry columns carry a CRS type. ST_Distance_Spheroid and ST_DWithin_Spheroid reject it, so cast both arguments: ST_Distance_Spheroid(a.geometry::GEOMETRY, ST_Point(lon, lat)).
- A UTM zone is only valid for data inside that zone; for continental or global extents use the spheroid functions instead.
- List lambdas use the Python-style syntax: list_transform(l, lambda x: x + 1). The arrow form x -> x + 1 is rejected.

Return ONLY a JSON array, no markdown fences:
[{{"question": "...", "sql": "...", "skills": ["..."], "difficulty": "{difficulty}"}}]"""


def make_prompts(n, seed, out_dir="prompts", split="train"):
    rnd = random.Random(seed)
    c = con()
    cards = {t: schema_card(c, t) for t in TABLE_DOCS}
    out = HERE / out_dir
    out.mkdir(exist_ok=True)
    # Stratified: cycle skills/difficulties, walk a shuffled bundle list so coverage is even.
    bundles = (EVAL_BUNDLES if split == "eval" else BUNDLES)[:]
    rnd.shuffle(bundles)
    for i in range(n):
        bundle = bundles[i % len(bundles)]
        meta = dict(id=f"s{seed}_{i:03d}", tables=bundle, skill=SKILLS[i % len(SKILLS)],
                    difficulty=DIFFICULTIES[i % len(DIFFICULTIES)], style=rnd.choice(STYLES), k=6)
        prompt = GEN_PROMPT.format(
            cards="\n\n".join(cards[t] for t in bundle),
            grounding="\n\n".join(grounding(c, t, seed * 1000 + i) for t in bundle),
            **{k: meta[k] for k in ("skill", "difficulty", "style", "k")},
        )
        (out / f"{meta['id']}.txt").write_text(prompt)
        (out / f"{meta['id']}.json").write_text(json.dumps(meta))
    print(f"wrote {n} prompts to {out}")


QUERY_TIMEOUT_S = 30
FORBIDDEN = re.compile(r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|COPY|ATTACH|INSTALL|LOAD|PRAGMA|SET)\b", re.IGNORECASE)


def exec_one(sql):
    """Run one query on a fresh connection and summarise the result (used inside a worker process)."""
    c = con()
    c.execute("SET threads=2")
    c.execute("SET memory_limit='2GB'")
    t = time.time()
    try:
        cur = c.execute(sql)
        data = cur.fetchmany(1001)
    except Exception as e:
        return dict(error=f"{type(e).__name__}: {str(e).splitlines()[0][:200]}")
    all_null = bool(data) and all(v is None for row in data for v in row)
    return dict(
        n_rows=len(data), ms=round((time.time() - t) * 1000), cols=[d[0] for d in cur.description],
        preview=[[_short(v, 40) for v in row] for row in data[:8]], all_null=all_null,
        ok=len(data) > 0 and not all_null, error=None if data and not all_null else ("empty result" if not data else "all-null result"),
    )


def _validate_row(r):
    sql = r["sql"].strip().rstrip(";")
    res = dict(r, ok=False, error=None, n_rows=None, ms=None, preview=None)
    if FORBIDDEN.search(re.sub(r"'[^']*'", "''", sql)):
        res["error"] = "forbidden statement"
        return res
    # Each query runs in its own process, so a crash, memory blow-up or hang only fails that query.
    try:
        p = subprocess.run([sys.executable, "-I", __file__, "exec1"], input=sql, capture_output=True,
                           text=True, timeout=QUERY_TIMEOUT_S + 10)
    except subprocess.TimeoutExpired:
        res["error"] = f"timeout >{QUERY_TIMEOUT_S}s"
        return res
    if p.returncode != 0 or not p.stdout.strip():
        res["error"] = f"crash (exit {p.returncode}): {p.stderr.strip().splitlines()[-1][:150] if p.stderr.strip() else ''}"
        return res
    res.update(json.loads(p.stdout))
    return res


def validate(path):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    with ThreadPoolExecutor(4) as ex:
        results = list(ex.map(_validate_row, rows))
    outp = Path(path).with_suffix(".validated.jsonl")
    outp.write_text("\n".join(json.dumps(r, default=str) for r in results))
    ok = sum(r["ok"] for r in results)
    print(f"{ok}/{len(results)} executed with non-empty results -> {outp}")
    for r in results:
        if not r["ok"]:
            print(f"  FAIL [{r.get('scenario')}] {r['error']}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "cards":
        c = con()
        for t in TABLE_DOCS:
            print(schema_card(c, t), "\n")
    elif cmd == "prompts":
        make_prompts(int(sys.argv[2]), int(sys.argv[3]), *sys.argv[4:6])
    elif cmd == "validate":
        validate(sys.argv[2])
    elif cmd == "exec1":
        print(json.dumps(exec_one(sys.stdin.read()), default=str))
