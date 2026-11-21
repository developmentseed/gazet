"""Second lake expansion: tables that differ in schema style, not only in region (training set v2).

New styles: point data as plain lon/lat number columns (no geometry), geometry stored as WKT text, geometry in a
projected CRS under another column name, wide tables (30-40 columns, trimmed so a schema card stays under ~1,000
tokens), camelCase API column names, and plain data tables that join to other tables by country or airport code. Reuses the helpers of build_lake_world.py; downloads go to raw/world/.

ghcn_stations and fr_departements are held out (harness.HELD_OUT) for a second gold set.

Usage: python build_lake_more.py [DATASET_DIR] [builder ...]
"""
import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
import build_lake_world as w  # noqa: E402

LICENSES = {
    "states_provinces": "Public domain (Natural Earth)",
    "time_zones": "Public domain (Natural Earth)",
    "railroads": "Public domain (Natural Earth)",
    "marine_areas": "Public domain (Natural Earth)",
    "us_counties": "Public domain (US Census Bureau cartographic boundaries)",
    "world_bank_indicators": "CC BY 4.0 (World Bank World Development Indicators)",
    "openflights_routes": "ODbL (OpenFlights)",
    "openflights_airlines": "ODbL (OpenFlights)",
    "nairobi_places": "Overture Maps (CDLA Permissive 2.0 / ODbL by source)",
    "nairobi_roads": "Overture Maps (ODbL)",
    "cdmx_buildings": "Overture Maps (ODbL / CDLA Permissive 2.0 by source)",
    "tsunami_events": "Public domain (NOAA NCEI Global Historical Tsunami Database)",
    "volcano_eruptions": "Public domain (NOAA NCEI Significant Volcanic Eruptions Database)",
    "ghcn_stations": "Public domain (NOAA NCEI GHCN-Daily)",
    "fr_departements": "Etalab Open Licence 2.0 (IGN / INSEE via gregoiredavid/france-geojson)",
}
NE = "https://naciscdn.org/naturalearth/10m"


def _ne(theme, name):
    p = w.fetch(f"{NE}/{theme}/{name}.zip", f"{name}.zip")
    return f"/vsizip/{p}/{name}.shp"


def states_provinces():
    # A wide table: 35 of the 123 source columns (all name_xx translations and FCLASS_xx columns dropped).
    w.write("states_provinces", f"""
        SELECT ne_id, adm1_code, iso_3166_2, iso_a2, adm0_a3, admin, name, name_alt, name_local, name_en, name_es, name_fr,
               name_zh, type, type_en, code_hasc, postal, region, region_cod, region_sub, sub_code, provnum_ne, gadm_level,
               area_sqkm, fips, woe_name, gn_name, gn_id, wikidataid, wikipedia, latitude, longitude, scalerank,
               labelrank, featurecla, geom AS geometry
        FROM ST_Read('{_ne('cultural', 'ne_10m_admin_1_states_provinces')}')""")


def time_zones():
    # Geometry as WKT text in a column called wkt: queries must parse it with ST_GeomFromText.
    w.write("time_zones", f"""
        SELECT objectid, name, zone, utc_format, time_zone, tz_name1st, places, dst_places,
               ST_AsText(ST_SimplifyPreserveTopology(geom, 0.01)) AS wkt
        FROM ST_Read('{_ne('cultural', 'ne_10m_time_zones')}')""")


def railroads():
    w.write("railroads", f"""
        SELECT rwdb_rr_id, mult_track, electric, category, featurecla, scalerank, natlscale, continent, geom AS geometry
        FROM ST_Read('{_ne('cultural', 'ne_10m_railroads')}')""")


def marine_areas():
    w.write("marine_areas", f"""
        SELECT name, namealt, featurecla, label, wikidataid, min_label, geom AS geometry
        FROM ST_Read('{_ne('physical', 'ne_10m_geography_marine_polys')}')""")


def us_counties():
    # Projected CRS (CONUS Albers, metres) and a geometry column called geom.
    p = w.fetch("https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_20m.zip", "cb_2023_us_county_20m.zip")
    w.write("us_counties", f"""
        SELECT STATEFP AS statefp, COUNTYFP AS countyfp, GEOID AS geoid, NAME AS name, NAMELSAD AS namelsad,
               STUSPS AS stusps, STATE_NAME AS state_name, ALAND AS aland, AWATER AS awater,
               ST_Transform(geom, 'EPSG:4269', 'EPSG:5070', always_xy := true)::GEOMETRY('EPSG:5070') AS geom
        FROM ST_Read('/vsizip/{p}/cb_2023_us_county_20m.shp')""")


def world_bank_indicators():
    indicators = ["NY.GDP.PCAP.CD", "SP.POP.TOTL", "SP.DYN.LE00.IN", "SP.URB.TOTL.IN.ZS", "AG.LND.FRST.ZS", "EG.ELC.ACCS.ZS"]
    countries = json.loads(w.fetch("https://api.worldbank.org/v2/country?format=json&per_page=400", "wb_countries.json").read_text())[1]
    real = {c["id"]: c for c in countries if c["region"]["value"] != "Aggregates"}
    rows = []
    for ind in indicators:
        data = json.loads(w.fetch(f"https://api.worldbank.org/v2/country/all/indicator/{ind}?format=json&date=2010:2023&per_page=20000",
                                  f"wb_{ind}.json").read_text())[1]
        for r in data:
            iso3 = r["countryiso3code"]
            if iso3 in real and r["value"] is not None:
                rows.append(dict(country_iso3=iso3, country_name=r["country"]["value"], region=real[iso3]["region"]["value"],
                                 income_level=real[iso3]["incomeLevel"]["value"], indicator_code=ind,
                                 indicator_name=r["indicator"]["value"], year=int(r["date"]), value=float(r["value"])))
    p = w.RAW / "wb_indicators.json"
    p.write_text(json.dumps(rows))
    w.write("world_bank_indicators", f"SELECT * FROM read_json('{p}') ORDER BY country_iso3, indicator_code, year")


def openflights():
    r = w.fetch("https://raw.githubusercontent.com/jpatokal/openflights/master/data/routes.dat", "openflights_routes.dat")
    a = w.fetch("https://raw.githubusercontent.com/jpatokal/openflights/master/data/airlines.dat", "openflights_airlines.dat")
    w.write("openflights_routes", f"""
        SELECT * FROM read_csv('{r}', header=false, nullstr='\\N', columns={{'airline': 'VARCHAR', 'airline_id': 'BIGINT',
            'source_airport': 'VARCHAR', 'source_airport_id': 'BIGINT', 'destination_airport': 'VARCHAR',
            'destination_airport_id': 'BIGINT', 'codeshare': 'VARCHAR', 'stops': 'INTEGER', 'equipment': 'VARCHAR'}})""")
    w.write("openflights_airlines", f"""
        SELECT * FROM read_csv('{a}', header=false, nullstr='\\N', columns={{'airline_id': 'BIGINT', 'name': 'VARCHAR',
            'alias': 'VARCHAR', 'iata': 'VARCHAR', 'icao': 'VARCHAR', 'callsign': 'VARCHAR', 'country': 'VARCHAR',
            'active': 'VARCHAR'}}) WHERE airline_id > 0""")


NAIROBI = "bbox.xmin > 36.78 AND bbox.xmax < 36.86 AND bbox.ymin > -1.31 AND bbox.ymax < -1.25"


def nairobi_places():
    w.write("nairobi_places", f"""
        SELECT id, names, basic_category, taxonomy, confidence, operating_status, websites, phones, addresses, brand, geometry
        FROM read_parquet('{w.OV}/theme=places/type=place/*', hive_partitioning=1) WHERE {NAIROBI}""")


def nairobi_roads():
    w.write("nairobi_roads", f"""
        SELECT id, names, subtype, class, subclass, road_surface, speed_limits, geometry
        FROM read_parquet('{w.OV}/theme=transportation/type=segment/*', hive_partitioning=1)
        WHERE {NAIROBI} AND subtype = 'road'""")


def cdmx_buildings():
    bbox = "bbox.xmin > -99.17 AND bbox.xmax < -99.13 AND bbox.ymin > 19.415 AND bbox.ymax < 19.44"
    w.write("cdmx_buildings", f"""
        SELECT id, names, subtype, class, height, num_floors, roof_shape, geometry
        FROM read_parquet('{w.OV}/theme=buildings/type=building/*', hive_partitioning=1) WHERE {bbox}""")


def _hazel(kind, name):
    p = w.RAW / f"{name}.json"
    if not p.exists():
        items, page, pages = [], 1, 1
        while page <= pages:
            d = requests.get(f"https://www.ngdc.noaa.gov/hazel/hazard-service/api/v1/{kind}?page={page}", headers=w.UA, timeout=120).json()
            items += d["items"]
            pages, page = d["totalPages"], page + 1
        p.write_text(json.dumps(items))
    return p


def tsunami_events():
    # Point data as plain latitude/longitude columns; camelCase names as the API returns them.
    w.write("tsunami_events", f"""
        SELECT id, "year", "month", "day", "hour", "minute", eventValidity, causeCode, earthquakeEventId, volcanoEventId,
               country, locationName, regionCode, latitude, longitude, eqMagnitude, eqDepth, maxWaterHeight, numRunups,
               tsIntensity, oceanicTsunami, deathsTotal, deathsAmountOrderTotal, injuriesTotal, housesDestroyedTotal,
               damageMillionsDollarsTotal, damageAmountOrderTotal, warningStatusId
        FROM read_json('{_hazel('tsunamis/events', 'noaa_tsunamis')}') WHERE latitude IS NOT NULL AND longitude IS NOT NULL""")


def volcano_eruptions():
    w.write("volcano_eruptions", f"""
        SELECT id, "year", "month", "day", volcanoLocationId, volcanoLocationNum, name, location, country, latitude,
               longitude, elevation, morphology, agent, vei, status, timeErupt, significant, eruption, tsunamiEventId,
               earthquakeEventId, deathsTotal, deathsAmountOrderTotal, injuriesTotal, housesDestroyedTotal,
               damageMillionsDollarsTotal, damageAmountOrderTotal
        FROM read_json('{_hazel('volcanoes', 'noaa_volcanoes')}') WHERE latitude IS NOT NULL""")


def ghcn_stations():
    p = w.fetch("https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt", "ghcnd-stations.txt")
    w.write("ghcn_stations", f"""
        SELECT trim(line[1:11]) AS station_id, line[13:20]::DOUBLE AS latitude, line[22:30]::DOUBLE AS longitude,
               nullif(line[32:37]::DOUBLE, -999.9) AS elevation_m, nullif(trim(line[39:40]), '') AS state,
               trim(line[42:71]) AS name, nullif(trim(line[73:75]), '') AS gsn_flag,
               nullif(trim(line[77:79]), '') AS hcn_crn_flag, nullif(trim(line[81:85]), '') AS wmo_id,
               line[1:2] AS country_code
        FROM read_csv('{p}', header=false, delim='\\x1f', quote='', columns={{'line': 'VARCHAR'}})""")


def fr_departements():
    # Lambert-93 (EPSG:2154) under a column called geom.
    p = w.fetch("https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements.geojson", "fr_departements.geojson")
    w.write("fr_departements", f"""
        SELECT code, nom, ST_Transform(geom, 'EPSG:4326', 'EPSG:2154', always_xy := true)::GEOMETRY('EPSG:2154') AS geom
        FROM ST_Read('{p}')""")


if __name__ == "__main__":
    builders = [states_provinces, time_zones, railroads, marine_areas, us_counties, world_bank_indicators, openflights,
                nairobi_places, nairobi_roads, cdmx_buildings, tsunami_events, volcano_eruptions, ghcn_stations, fr_departements]
    only = set(sys.argv[2:])
    failed = []
    for b in builders:
        if only and b.__name__ not in only:
            continue
        try:
            b()
        except Exception as e:
            failed.append(b.__name__)
            print(f"{b.__name__:24s} FAILED: {type(e).__name__}: {str(e).splitlines()[0][:200]}")
    print("failed:", failed or "none")
