"""Build a small multi-schema geo data lake (GeoParquet) for SQL validation."""
import sys
from pathlib import Path

import duckdb

S = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
RAW, LAKE = S / "raw", S / "lake"
LAKE.mkdir(exist_ok=True)
OV = "s3://overturemaps-us-west-2/release/2026-10-21.0"
# Lisbon metro bbox
XMIN, YMIN, XMAX, YMAX = -9.25, 38.68, -9.08, 38.80

con = duckdb.connect()
for ext in ("spatial", "httpfs"):
    con.execute(f"INSTALL {ext}; LOAD {ext};")
con.execute("SET s3_region='us-west-2'")


def write(name, sql):
    out = LAKE / f"{name}.parquet"
    con.execute(f"COPY ({sql}) TO '{out}' (FORMAT parquet)")
    n = con.execute(f"SELECT count(*) FROM '{out}'").fetchone()[0]
    print(f"{name:20s} {n:>8} rows")


ne = RAW / "ne"
write("countries", f"""
    SELECT ADM0_A3 AS iso_a3, ISO_A2 AS iso_a2, NAME AS name, NAME_LONG AS name_long,
           CONTINENT AS continent, REGION_UN AS region_un, SUBREGION AS subregion,
           POP_EST::BIGINT AS pop_est, GDP_MD::BIGINT AS gdp_md, INCOME_GRP AS income_grp,
           ECONOMY AS economy, geom AS geometry
    FROM ST_Read('{ne}/ne_50m_admin_0_countries/ne_50m_admin_0_countries.shp')""")
write("populated_places", f"""
    SELECT NAME AS name, ADM0NAME AS country, ISO_A2 AS iso_a2, ADM1NAME AS admin1,
           FEATURECLA AS feature_class, POP_MAX::BIGINT AS pop_max, POP_MIN::BIGINT AS pop_min,
           ADM0CAP::INT = 1 AS is_capital, WORLDCITY::INT = 1 AS is_world_city,
           geom AS geometry
    FROM ST_Read('{ne}/ne_50m_populated_places/ne_50m_populated_places.shp')""")
write("rivers", f"""
    SELECT name, featurecla AS feature_class, scalerank::INT AS scalerank, geom AS geometry
    FROM ST_Read('{ne}/ne_50m_rivers_lake_centerlines/ne_50m_rivers_lake_centerlines.shp')""")
write("lakes", f"""
    SELECT name, featurecla AS feature_class, scalerank::INT AS scalerank, geom AS geometry
    FROM ST_Read('{ne}/ne_50m_lakes/ne_50m_lakes.shp')""")
write("airports", f"""
    SELECT name, iata_code, gps_code AS icao_code, type, scalerank::INT AS scalerank,
           wikipedia, geom AS geometry
    FROM ST_Read('{ne}/ne_10m_airports/ne_10m_airports.shp')""")
write("ports", f"""
    SELECT name, website, scalerank::INT AS scalerank, geom AS geometry
    FROM ST_Read('{ne}/ne_10m_ports/ne_10m_ports.shp')""")

nyc = RAW / "nyc"
# Shapefile is EPSG:2263 (NY state plane, feet); store in EPSG:4326
write("nyc_taxi_zones", f"""
    SELECT LocationID::INT AS location_id, zone, borough,
           ST_Transform(geom, 'EPSG:2263', 'EPSG:4326', always_xy := true) AS geometry
    FROM ST_Read('{nyc}/taxi_zones/taxi_zones/taxi_zones.shp')""")
write("nyc_taxi_trips", f"""
    SELECT * EXCLUDE (store_and_fwd_flag) FROM '{nyc}/yellow_tripdata_2024-01.parquet'
    USING SAMPLE 200000 ROWS (reservoir, 42)""")

bbox = f"bbox.xmin > {XMIN} AND bbox.xmax < {XMAX} AND bbox.ymin > {YMIN} AND bbox.ymax < {YMAX}"
write("lisbon_places", f"""
    SELECT id, names, basic_category, taxonomy, confidence, operating_status, websites, socials, phones, addresses, brand,
           geometry
    FROM read_parquet('{OV}/theme=places/type=place/*', hive_partitioning=1) WHERE {bbox}""")
write("lisbon_buildings", f"""
    SELECT id, names, subtype, class, height, num_floors, roof_shape, geometry
    FROM read_parquet('{OV}/theme=buildings/type=building/*', hive_partitioning=1) WHERE {bbox}""")
write("lisbon_roads", f"""
    SELECT id, names, subtype, class, subclass, road_surface, speed_limits, geometry
    FROM read_parquet('{OV}/theme=transportation/type=segment/*', hive_partitioning=1)
    WHERE {bbox} AND subtype = 'road'""")

