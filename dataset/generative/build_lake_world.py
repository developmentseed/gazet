"""Add regionally and institutionally diverse open datasets to the lake (UK, Europe, South America, Asia, Africa, global).

Each source is built independently, so one unreachable portal does not stop the rest. Downloads are cached in raw/world/.
Licences are listed in LICENSES; check them before publishing derived data.
"""
import datetime as dt
import json
import sys
import zipfile
from pathlib import Path

import duckdb
import requests

S = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
RAW, LAKE = S / "raw" / "world", S / "lake"
RAW.mkdir(parents=True, exist_ok=True)
LAKE.mkdir(exist_ok=True)
OV = "s3://overturemaps-us-west-2/release/2026-10-21.0"
UA = {"User-Agent": "Mozilla/5.0 (gazet dataset research)"}

LICENSES = {
    "uk_police_crimes": "Open Government Licence v3.0 (data.police.uk)",
    "uk_naptan_stops": "Open Government Licence v3.0 (DfT NaPTAN)",
    "uk_local_authorities": "Open Government Licence v3.0 (ONS Open Geography Portal)",
    "eu_nuts_regions": "Eurostat GISCO; (c) EuroGeographics for the administrative boundaries",
    "paris_trees": "ODbL (Ville de Paris)",
    "paris_velib_stations": "ODbL (Ville de Paris)",
    "nl_municipalities": "CC BY 4.0 (CBS / PDOK)",
    "br_fire_hotspots": "INPE Programa Queimadas open data (verify terms)",
    "br_sp_municipalities": "IBGE open data",
    "bike_share_stations": "GBFS feeds; licence per system (verify)",
    "jp_prefectures": "Derived from MLIT National Land Numerical Information (dataofjapan/land)",
    "in_districts": "DataMeet maps, Census 2011 districts (verify CC BY licence)",
    "sg_parks": "Singapore Open Data Licence (data.gov.sg)",
    "zm_health_facilities": "CC BY 4.0 (World Bank)",
    "usgs_earthquakes": "Public domain (USGS)",
    "power_plants": "CC BY 4.0 (WRI Global Power Plant Database)",
    "ourairports": "Public domain (OurAirports)",
    "ourairports_runways": "Public domain (OurAirports)",
    "geonames_cities": "CC BY 4.0 (GeoNames)",
    "tokyo_places": "Overture Maps (CDLA Permissive 2.0 / ODbL by source)",
}

con = duckdb.connect()
for ext in ("spatial", "httpfs"):
    con.execute(f"INSTALL {ext}; LOAD {ext};")
con.execute("SET s3_region='us-west-2'")


def fetch(url, name):
    path = RAW / name
    if not path.exists():
        r = requests.get(url, headers=UA, timeout=600)
        r.raise_for_status()
        path.write_bytes(r.content)
    return path


def write(name, sql):
    out = LAKE / f"{name}.parquet"
    con.execute(f"COPY ({sql}) TO '{out}' (FORMAT parquet)")
    n = con.execute(f"SELECT count(*) FROM '{out}'").fetchone()[0]
    print(f"{name:24s} {n:>8} rows")


def uk_police_crimes():
    # All street-level crime within one mile of central London, three months.
    rows = []
    for month in ("2024-04", "2024-05", "2024-06"):
        p = fetch(f"https://data.police.uk/api/crimes-street/all-crime?lat=51.5074&lng=-0.1278&date={month}", f"police_{month}.json")
        rows += json.loads(p.read_text())
    (RAW / "police_all.json").write_text(json.dumps(rows))
    write("uk_police_crimes", f"""
        SELECT id, persistent_id, month, category, location_type, location.street.name AS street_name,
               outcome_status.category AS outcome_category, outcome_status.date AS outcome_month,
               ST_Point(location.longitude::DOUBLE, location.latitude::DOUBLE) AS geometry
        FROM read_json_auto('{RAW / "police_all.json"}')""")


def uk_naptan_stops():
    p = fetch("https://naptan.api.dft.gov.uk/v1/access-nodes?dataFormat=csv&atcoAreaCodes=490", "naptan_london.csv")
    write("uk_naptan_stops", f"""
        SELECT ATCOCode AS atco_code, NaptanCode AS naptan_code, CommonName AS common_name, Street AS street,
               Indicator AS indicator, LocalityName AS locality_name, StopType AS stop_type, BusStopType AS bus_stop_type,
               Status AS status, Easting::INT AS easting, Northing::INT AS northing,
               ST_Point(Longitude::DOUBLE, Latitude::DOUBLE) AS geometry
        FROM read_csv_auto('{p}', all_varchar=true) WHERE Longitude IS NOT NULL""")


def uk_local_authorities():
    p = fetch("https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
              "Local_Authority_Districts_December_2023_Boundaries_UK_BGC/FeatureServer/0/query?where=1%3D1&outFields=*&f=geojson",
              "lad_2023.geojson")
    write("uk_local_authorities", f"SELECT LAD23CD AS lad23cd, LAD23NM AS lad23nm, LAD23NMW AS lad23nmw, geom AS geometry FROM ST_Read('{p}')")


def eu_nuts_regions():
    p = fetch("https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_20M_2021_4326.geojson", "nuts_2021.geojson")
    write("eu_nuts_regions", f"""
        SELECT NUTS_ID AS nuts_id, LEVL_CODE::INT AS levl_code, CNTR_CODE AS cntr_code, NAME_LATN AS name_latn,
               NUTS_NAME AS nuts_name, MOUNT_TYPE::INT AS mount_type, URBN_TYPE::INT AS urbn_type,
               COAST_TYPE::INT AS coast_type, geom AS geometry
        FROM ST_Read('{p}')""")


def paris_trees():
    p = fetch("https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/les-arbres/exports/parquet", "paris_arbres.parquet")
    write("paris_trees", f"""
        SELECT * EXCLUDE (geo_point_2d), geo_point_2d AS geometry FROM '{p}' WHERE geo_point_2d IS NOT NULL""")


def paris_velib_stations():
    p = fetch("https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/velib-emplacement-des-stations/exports/geojson", "velib.geojson")
    write("paris_velib_stations", f"SELECT * EXCLUDE (geom), geom AS geometry FROM ST_Read('{p}')")


def nl_municipalities():
    p = RAW / "nl_gemeenten.geojson"
    if not p.exists():
        # The OGC API returns 502 for large pages; follow the "next" links in pages of 100.
        url, features = "https://api.pdok.nl/cbs/wijken-en-buurten-2023/ogc/v1/collections/gemeenten/items?f=json&limit=100", []
        while url:
            page = requests.get(url, headers=UA, timeout=120).json()
            features += page["features"]
            url = next((link["href"] for link in page.get("links", []) if link.get("rel") == "next"), None)
        p.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    # The source has 236 CBS statistics columns (a ~6k-token schema card); keep a representative subset.
    write("nl_municipalities", f"""
        SELECT gemeentecode, gemeentenaam, water, aantal_inwoners, aantal_huishoudens, bevolkingsdichtheid_inwoners_per_km2,
               oppervlakte_totaal_in_ha, oppervlakte_land_in_ha, oppervlakte_water_in_ha, gemiddelde_woningwaarde,
               gemiddeld_inkomen_per_inwoner, percentage_personen_65_jaar_en_ouder, percentage_eenpersoonshuishoudens,
               geboortes_per_1000_inwoners, personenautos_per_huishouden, stedelijkheid_adressen_per_km2, geom AS geometry
        FROM ST_Read('{p}')""")


def br_fire_hotspots():
    files = []
    for d in range(1, 8):
        day = (dt.date(2026, 10, 27) - dt.timedelta(days=d)).strftime("%Y%m%d")
        try:
            files.append(str(fetch(f"https://dataserver-coids.inpe.br/queimadas/queimadas/focos/csv/diario/Brasil/focos_diario_br_{day}.csv",
                                   f"focos_{day}.csv")))
        except requests.HTTPError:
            continue
    write("br_fire_hotspots", f"""
        SELECT * EXCLUDE (lat, lon), ST_Point(lon::DOUBLE, lat::DOUBLE) AS geometry
        FROM read_csv_auto({files}, union_by_name=true)""")


def br_sp_municipalities():
    p = fetch("https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_municipais/municipio_2022/UFs/SP/SP_Municipios_2022.zip",
              "SP_Municipios_2022.zip")
    zipfile.ZipFile(p).extractall(RAW / "sp_mun")
    shp = next((RAW / "sp_mun").rglob("*.shp"))
    write("br_sp_municipalities", f"SELECT * EXCLUDE (geom), ST_Transform(geom, 'EPSG:4674', 'EPSG:4326', always_xy := true) AS geometry FROM ST_Read('{shp}')")


def bike_share_stations():
    systems = {"ecobici_buenos_aires": "https://buenosaires.publicbikesystem.net/customer/gbfs/v2/en/station_information",
               "ecobici_mexico_city": "https://gbfs.mex.lyftbikes.com/gbfs/en/station_information.json"}
    rows = []
    for system, url in systems.items():
        for s in json.loads(fetch(url, f"gbfs_{system}.json").read_text())["data"]["stations"]:
            rows.append(dict(system_id=system, station_id=str(s["station_id"]), name=s.get("name"), address=s.get("address"),
                             capacity=s.get("capacity"), lon=s["lon"], lat=s["lat"]))
    (RAW / "gbfs_all.json").write_text(json.dumps(rows))
    write("bike_share_stations", f"""
        SELECT system_id, station_id, name, address, capacity::INT AS capacity, ST_Point(lon, lat) AS geometry
        FROM read_json_auto('{RAW / "gbfs_all.json"}')""")


def jp_prefectures():
    p = fetch("https://raw.githubusercontent.com/dataofjapan/land/master/japan.geojson", "japan.geojson")
    write("jp_prefectures", f"SELECT * EXCLUDE (geom), geom AS geometry FROM ST_Read('{p}')")


def in_districts():
    base = "https://raw.githubusercontent.com/datameet/maps/master/Districts/Census_2011/2011_Dist"
    for ext in ("shp", "shx", "dbf", "prj"):
        fetch(f"{base}.{ext}", f"2011_Dist.{ext}")
    write("in_districts", f"SELECT * EXCLUDE (geom), geom AS geometry FROM ST_Read('{RAW / '2011_Dist.shp'}')")


def sg_parks():
    meta = requests.get("https://api-open.data.gov.sg/v1/public/api/datasets/d_0542d48f0991541706b58059381a6eca/poll-download",
                        headers=UA, timeout=60).json()
    p = fetch(meta["data"]["url"], "sg_parks.geojson")
    write("sg_parks", f"SELECT * EXCLUDE (geom), geom AS geometry FROM ST_Read('{p}')")


def _upto_last_brace(text):
    # The published file has a stray "System.IO.MemoryStream" line after the closing brace.
    return text[: text.rfind("}") + 1]


def zm_health_facilities():
    p = fetch("https://datacatalogfiles.worldbank.org/ddh-published/0042276/1/DR0052880/zamhealthfacilities.geojson", "zm_health.geojson")
    # DuckDB's GeoJSON reader rejects this file; flatten it to point rows in Python instead.
    rows = [dict(f["properties"], lon=f["geometry"]["coordinates"][0], lat=f["geometry"]["coordinates"][1])
            for f in json.loads(_upto_last_brace(p.read_text(encoding="utf-8-sig")))["features"] if f.get("geometry")]
    (RAW / "zm_health_rows.json").write_text(json.dumps(rows))
    write("zm_health_facilities", f"""
        SELECT * EXCLUDE (lon, lat), ST_Point(lon, lat) AS geometry FROM read_json_auto('{RAW / "zm_health_rows.json"}')""")


def usgs_earthquakes():
    p = fetch("https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.geojson", "usgs_all_month.geojson")
    write("usgs_earthquakes", f"""
        SELECT ids, mag, magType AS mag_type, place, epoch_ms("time") AS event_time, felt, tsunami, sig,
               net, "type" AS event_type, status, ST_Z(geom) AS depth_km, ST_Force2D(geom) AS geometry
        FROM ST_Read('{p}')""")


def power_plants():
    p = fetch("https://wri-dataportal-prod.s3.amazonaws.com/manual/global_power_plant_database_v_1_3.zip", "gppd.zip")
    csv = RAW / "gppd" / "global_power_plant_database.csv"
    if not csv.exists():
        zipfile.ZipFile(p).extractall(RAW / "gppd")
    write("power_plants", f"""
        SELECT * EXCLUDE (latitude, longitude), ST_Point(longitude, latitude) AS geometry
        FROM read_csv_auto('{csv}')""")


def ourairports():
    a = fetch("https://davidmegginson.github.io/ourairports-data/airports.csv", "ourairports.csv")
    r = fetch("https://davidmegginson.github.io/ourairports-data/runways.csv", "ourairports_runways.csv")
    write("ourairports", f"""
        SELECT * EXCLUDE (latitude_deg, longitude_deg), ST_Point(longitude_deg, latitude_deg) AS geometry
        FROM read_csv_auto('{a}')""")
    write("ourairports_runways", f"SELECT * FROM read_csv_auto('{r}')")


def geonames_cities():
    p = fetch("https://download.geonames.org/export/dump/cities15000.zip", "cities15000.zip")
    zipfile.ZipFile(p).extractall(RAW)
    cols = ["geonameid", "name", "asciiname", "alternatenames", "latitude", "longitude", "feature_class", "feature_code",
            "country_code", "cc2", "admin1_code", "admin2_code", "admin3_code", "admin4_code", "population", "elevation",
            "dem", "timezone", "modification_date"]
    col_spec = ", ".join(f"'{c}': '{'DOUBLE' if c in ('latitude', 'longitude') else 'BIGINT' if c in ('geonameid', 'population') else 'VARCHAR'}'" for c in cols)
    write("geonames_cities", f"""
        SELECT * EXCLUDE (latitude, longitude, cc2, admin3_code, admin4_code, dem), ST_Point(longitude, latitude) AS geometry
        FROM read_csv('{RAW / "cities15000.txt"}', delim='\t', header=false, quote='', columns={{{col_spec}}})""")


def tokyo_places():
    bbox = "bbox.xmin > 139.74 AND bbox.xmax < 139.78 AND bbox.ymin > 35.67 AND bbox.ymax < 35.70"
    write("tokyo_places", f"""
        SELECT id, names, basic_category, taxonomy, confidence, operating_status, websites, phones, addresses, brand, geometry
        FROM read_parquet('{OV}/theme=places/type=place/*', hive_partitioning=1) WHERE {bbox}""")


if __name__ == "__main__":
    builders = [uk_police_crimes, uk_naptan_stops, uk_local_authorities, eu_nuts_regions, paris_trees, paris_velib_stations,
                nl_municipalities, br_fire_hotspots, br_sp_municipalities, bike_share_stations, jp_prefectures, in_districts,
                sg_parks, zm_health_facilities, usgs_earthquakes, power_plants, ourairports, geonames_cities, tokyo_places]
    only = set(sys.argv[2:])
    failed = []
    for b in builders:
        if only and b.__name__ not in only:
            continue
        try:
            b()
        except Exception as e:
            failed.append(b.__name__)
            print(f"{b.__name__:24s} FAILED: {type(e).__name__}: {str(e).splitlines()[0][:150]}")
    print("failed:", failed or "none")
