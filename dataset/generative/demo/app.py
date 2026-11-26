"""Gazet SQL demo: question + tables -> DuckDB spatial SQL from the fine-tuned student, run and mapped.

The model runs in llama-server on CPU. The prompt is built by the same code as the training data
(student.py, hints.py, harness.py are copied in by deploy.sh), so what you see is what was evaluated.

Environment:
    MODEL_REPO / MODEL_FILE   GGUF to serve (default: developmentseed/gazet-models, rl2-q3-06b-pub-lr1e5/gguf/...q8_0.gguf)
    DATA_REPO                 dataset with lake/*.parquet (default: developmentseed/gazet-sql-data)
    HF_TOKEN                  read access to both (Space secret)
    LLAMA_SERVER              llama-server binary (default /opt/llama/llama-server, as installed by the Dockerfile)
"""
import html
import ipaddress
import json
import os
import re
import socket
import subprocess
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import gradio as gr
import pandas as pd
import requests
import sqlglot
from huggingface_hub import hf_hub_download, snapshot_download

HERE = Path(__file__).parent
MODEL_REPO = os.environ.get("MODEL_REPO", "developmentseed/gazet-models")
MODEL_FILE = os.environ.get("MODEL_FILE", "rl2-q3-06b-pub-lr1e5/gguf/gazet-rl2-q3-06b-pub-lr1e5-q8_0.gguf")
DATA_REPO = os.environ.get("DATA_REPO", "developmentseed/gazet-sql-data")
PORT = 8081
MAX_ROWS, MAX_FEATURES, QUERY_SECONDS = 500, 2000, 30

snapshot_download(DATA_REPO, repo_type="dataset", allow_patterns=["lake/*.parquet"], local_dir=HERE)
gguf = hf_hub_download(MODEL_REPO, MODEL_FILE)
# Fewer threads than cores: on shared VMs all cores collapse decode speed about tenfold (FINDINGS.md).
THREADS = (os.cpu_count() or 2) if (os.cpu_count() or 2) <= 4 else (os.cpu_count() - 2)
server = subprocess.Popen([os.environ.get("LLAMA_SERVER", "/opt/llama/llama-server"), "-m", gguf, "-c", "4096", "--port", str(PORT),
                           "-t", str(THREADS), "--jinja", "--log-disable"])

import harness  # noqa: E402  (needs lake/ in place)
import hints  # noqa: E402
import student  # noqa: E402

CON = harness.con()
for stmt in ("INSTALL httpfs", "LOAD httpfs", "SET enable_http_metadata_cache = true", "SET parquet_metadata_cache = true"):
    try:  # httpfs gives HTTP range requests: a remote Parquet file is read in pieces, never downloaded whole
        CON.execute(stmt)
    except Exception as e:
        print(f"warning: {stmt}: {e}")
# The public lake leaves out some tables (eu_nuts_regions: non-commercial licence), so list only those present.
TABLES = sorted(t for t in harness.TABLE_DOCS if (harness.LAKE / f"{t}.parquet").exists())
CARDS = {t: harness.schema_card(CON, t) for t in TABLES}
BLOCKED = re.compile(r"\b(COPY|ATTACH|DETACH|INSTALL|LOAD|PRAGMA|SET|RESET|CREATE|DROP|ALTER|INSERT|UPDATE|DELETE|EXPORT|IMPORT|"
                     r"CALL|read_\w+|glob|getenv|ST_Read)\b", re.IGNORECASE)

EXAMPLES = [
    # tables seen in training
    [["power_plants", "countries"], "Solar power plants over 100 MW in Africa, with their locations"],
    [["jp_prefectures"], "The shape of Fukushima minus Yamagata, as GeoJSON, with its area in km2"],
    [["lisbon_places"], "Cafes within 300 m of longitude -9.1393, latitude 38.7139, nearest first, with lon and lat"],
    [["usgs_earthquakes"], "Earthquakes of magnitude 5 or more, with lon and lat"],
    # tables never seen in training
    [["uk_police_crimes"], "Bicycle thefts, with lon and lat"],
    [["nl_municipalities"], "The 5 largest municipalities by land area, with their shapes as GeoJSON"],
    [["sg_parks"], "Parks within 1 km of Marina Bay Sands (longitude 103.8607, latitude 1.2834), with lon and lat"],
    [["br_fire_hotspots", "br_sp_municipalities"], "The 20 fire hotspots with the highest frp, the municipality each falls in, and lon and lat"],
]


UPLOADED = set()  # tables the user dropped in; shared by everyone using this process, so run it locally for private data
LOAD_WHOLE_ROWS, SAMPLE_ROWS = 200_000, 50_000  # smaller remote files are loaded whole; bigger ones are linked and sampled
UPLOAD_TYPES = [".parquet", ".geoparquet", ".csv", ".tsv", ".geojson", ".json", ".gpkg", ".shp", ".fgb", ".kml", ".zip"]


def table_label(t):
    return f"{t} (new)" if t in harness.HELD_OUT else f"{t} (uploaded)" if t in UPLOADED else t


def table_name(label):
    return label.split(" (")[0]


def safe_name(stem):
    name = re.sub(r"[^a-z0-9_]+", "_", stem.lower()).strip("_") or "upload"
    if name[0].isdigit():
        name = "t_" + name
    if name in TABLES and name not in UPLOADED:  # never replace a built-in table
        name += "_upload"
    return name


def normalise_crs(name, kind):
    """Make a new table easy for the model: snake_case column names (`Country (Admin 0)` -> country_admin_0), and geometry
    relabelled as lon/lat (EPSG:4326) like the built-in tables so joins across tables work. `kind` is TABLE or VIEW."""
    cols = CON.execute(f'DESCRIBE "{name}"').fetchall()
    seen, items, geo_items = set(), [], []
    for n, t, *_ in cols:
        new = re.sub(r"[^a-z0-9_]+", "_", n.lower()).strip("_") or "col"
        while new in seen:
            new += "_"
        seen.add(new)
        if t.startswith("GEOMETRY"):
            geo_items.append(f'"{n}"::GEOMETRY::GEOMETRY(\'EPSG:4326\') AS "{new}"')
            items.append(f'"{n}" AS "{new}"')  # fallback if the cast is refused (projected or unlabelled)
        else:
            geo_items.append(f'"{n}" AS "{new}"')
            items.append(f'"{n}" AS "{new}"')
    src = f'"{name}"' if kind == "TABLE" else f'"{name}_raw"'
    head = f'CREATE OR REPLACE {kind} "{name}" AS SELECT '
    try:
        CON.execute(head + ", ".join(geo_items) + f" FROM {src}")
    except Exception:
        CON.execute(head + ", ".join(items) + f" FROM {src}")


def register(name, doc, card_from=None, rows=None):
    """Add the table or view `name` to the demo. `card_from` is a local sample table to describe instead of the whole thing."""
    harness.TABLE_DOCS[name] = doc
    src = card_from or name
    harness.TABLE_DOCS[src] = doc
    card = harness.schema_card(CON, src)
    if card_from:  # name and row count of the real table, and say that the values come from a sample
        n = CON.execute(f'SELECT count(*) FROM "{src}"').fetchone()[0]
        card = card.replace(f"TABLE {src}  --", f"TABLE {name}  --", 1).replace(f"rows={n}", f"rows={rows} (value lists from the first {n})", 1)
        hints._INDEX[name] = hints.TableIndex(CON, src)
    else:
        hints._INDEX.pop(name, None)
    CARDS[name] = card
    if name not in TABLES:
        TABLES.append(name)
    UPLOADED.add(name)


def load_upload(path):
    """Register one dropped file as a DuckDB table and a schema card; returns the table name."""
    path = Path(path)
    ext = path.suffix.lower()
    name = safe_name(path.stem)
    src = str(path).replace("'", "''")
    if ext in (".parquet", ".geoparquet"):
        read = f"read_parquet('{src}')"
    elif ext in (".csv", ".tsv"):
        read = f"read_csv('{src}', auto_detect=true)"
    elif ext == ".zip":  # zipped shapefile
        read = f"ST_Read('/vsizip/{src}')"
    else:
        read = f"ST_Read('{src}')"
    CON.execute(f'CREATE OR REPLACE TABLE "{name}" AS SELECT * FROM {read}')
    normalise_crs(name, "TABLE")
    register(name, f"Uploaded file {path.name}.")
    return name


def check_public_url(url):
    """Refuse links to this machine or its network: the server fetches the URL, so this app must not be a way into them."""
    u = urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise ValueError("only http(s) links work")
    for info in socket.getaddrinfo(u.hostname, u.port or (443 if u.scheme == "https" else 80)):
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise ValueError("that address is not public")
    return u


def load_url(url):
    """Register a remote file. Parquet stays remote (a view; DuckDB fetches only the columns and row groups a query needs);
    small files are loaded whole, other formats are read through GDAL."""
    url = url.strip()
    u = check_public_url(url)
    ext = Path(u.path).suffix.lower()
    name = safe_name(Path(u.path).stem)
    q = url.replace("'", "''")
    if ext not in (".parquet", ".geoparquet"):
        CON.execute(f'CREATE OR REPLACE TABLE "{name}" AS SELECT * FROM ST_Read(\'{q}\')')
        normalise_crs(name, "TABLE")
        register(name, f"Remote file {Path(u.path).name}.")
        return name, "downloaded whole (not Parquet, so it cannot be read in pieces)"
    rows = CON.execute(f"SELECT num_rows FROM parquet_file_metadata('{q}')").fetchone()[0]  # footer only
    if rows <= LOAD_WHOLE_ROWS:
        CON.execute(f'CREATE OR REPLACE TABLE "{name}" AS SELECT * FROM read_parquet(\'{q}\')')
        normalise_crs(name, "TABLE")
        register(name, f"Remote file {Path(u.path).name}.")
        return name, f"{rows:,} rows, small, so downloaded whole"
    CON.execute(f'CREATE OR REPLACE VIEW "{name}_raw" AS SELECT * FROM read_parquet(\'{q}\')')
    CON.execute(f'CREATE OR REPLACE VIEW "{name}" AS SELECT * FROM "{name}_raw"')
    normalise_crs(name, "VIEW")
    # LIMIT stops after the first row groups, so describing the file costs a small read, not the whole file.
    CON.execute(f'CREATE OR REPLACE TABLE "{name}__sample" AS SELECT * FROM "{name}" LIMIT {SAMPLE_ROWS}')
    register(name, f"Remote file {Path(u.path).name}.", card_from=f"{name}__sample", rows=rows)
    return name, f"{rows:,} rows, linked, not downloaded: each query reads only the columns and row groups it needs"


def refresh(names, notes, selected):
    chosen = [table_label(table_name(t)) for t in selected or []] + [table_label(n) for n in names]
    return gr.update(choices=[table_label(t) for t in sorted(TABLES)], value=list(dict.fromkeys(chosen))), " ".join(notes)


def on_add(msg, selected):
    """One box for dropped files and pasted links (a message is {"text": ..., "files": [...]})."""
    msg = msg or {}
    names, notes = [], []
    for f in msg.get("files") or []:
        path = f if isinstance(f, str) else f.get("path") or getattr(f, "name", "")
        try:
            names.append(load_upload(path))
            notes.append(f"**{names[-1]}**: uploaded.")
        except Exception as e:
            notes.append(f"Could not load {Path(path).name}: {str(e).splitlines()[0][:200]}")
    for url in re.findall(r"https?://\S+", msg.get("text") or ""):
        try:
            name, how = load_url(url)
            names.append(name)
            notes.append(f"**{name}**: {how}.")
        except Exception as e:
            notes.append(f"Could not load {url[:80]}: {str(e).splitlines()[0][:250]}")
    if not names and not notes:
        notes.append("Drop a file, click to upload, or paste a link to a Parquet, GeoParquet, GeoJSON or GeoPackage file.")
    dropdown, note = refresh(names, notes, selected)
    return dropdown, "  \n".join(notes), gr.update(value=None)


def ask_model(system, user):
    t0 = time.time()
    r = requests.post(f"http://127.0.0.1:{PORT}/v1/chat/completions", timeout=300, json=dict(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0, max_tokens=512, chat_template_kwargs={"enable_thinking": False}))  # trained without thinking
    r.raise_for_status()
    j = r.json()
    text = re.sub(r"<think>.*?</think>", "", j["choices"][0]["message"]["content"] or "", flags=re.DOTALL)
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    sql = (m.group(1) if m else text).strip().rstrip(";").strip()
    t = j.get("timings") or {}
    return sql, time.time() - t0, t


def pretty(sql):
    """Indented SQL for display. Only the display changes: the query that runs is the model's text."""
    try:
        return sqlglot.transpile(sql, read="duckdb", write="duckdb", pretty=True, normalize_functions=False)[0]
    except Exception:
        return sql


def run_sql(sql):
    """Run a SELECT with a time limit; geometry columns come back as GeoJSON strings."""
    if not re.match(r"^\s*(WITH|SELECT|FROM)\b", sql, re.IGNORECASE) or ";" in sql or BLOCKED.search(sql):
        raise ValueError("only a single read-only SELECT is run in this demo")
    cur = CON.cursor()
    cur.execute("SET geometry_always_xy = true")  # per connection; the spheroid functions need it (see harness.con)
    cols = cur.execute(f"DESCRIBE SELECT * FROM ({sql}) q").fetchall()
    sel = ", ".join(f'ST_AsGeoJSON("{n}") AS "{n}"' if t.startswith("GEOMETRY") else f'"{n}"' for n, t, *_ in cols)
    timer = threading.Timer(QUERY_SECONDS, cur.interrupt)
    timer.start()
    try:
        df = cur.execute(f"SELECT {sel} FROM ({sql}) q LIMIT {MAX_FEATURES}").df()
    finally:
        timer.cancel()
    return df


def features(df):
    """GeoJSON features from GeoJSON-string columns, or from lon/lat columns."""
    geo_cols = [c for c in df.columns if df[c].dtype == object and df[c].dropna().astype(str).str.startswith('{"type"').any()]
    lon = next((c for c in df.columns if c.lower() in ("lon", "lng", "longitude", "x", "centre_lon", "center_lon")), None)
    lat = next((c for c in df.columns if c.lower() in ("lat", "latitude", "y", "centre_lat", "center_lat")), None)
    out = []
    for _, row in df.iterrows():
        props = {c: (None if pd.isna(v) else str(v)[:80]) for c, v in row.items() if c not in geo_cols}
        if geo_cols:
            for c in geo_cols:
                if isinstance(row[c], str) and row[c].startswith("{"):
                    out.append({"type": "Feature", "geometry": json.loads(row[c]), "properties": props})
        elif lon and lat and pd.notna(row[lon]) and pd.notna(row[lat]):
            try:
                out.append({"type": "Feature", "properties": props,
                            "geometry": {"type": "Point", "coordinates": [float(row[lon]), float(row[lat])]}})
            except (TypeError, ValueError):
                pass
    return out


def map_html(feats):
    if not feats:
        return "<p style='color:gray'>No geometry or lon/lat columns in the result, so nothing to map.</p>"
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>html,body,#m{{height:100%;margin:0}}</style></head><body><div id="m"></div><script>
const m = L.map('m');
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{maxZoom: 19, attribution: '&copy; OpenStreetMap'}}).addTo(m);
const layer = L.geoJSON({json.dumps({"type": "FeatureCollection", "features": feats})}, {{
  pointToLayer: (f, ll) => L.circleMarker(ll, {{radius: 5, weight: 1, fillOpacity: 0.7}}),
  onEachFeature: (f, l) => l.bindPopup(Object.entries(f.properties).map(([k, v]) => `<b>${{k}}</b>: ${{v}}`).join('<br>'))
}}).addTo(m);
m.fitBounds(layer.getBounds(), {{maxZoom: 14, padding: [20, 20]}});
</script></body></html>"""
    return f'<iframe style="width:100%;height:480px;border:0" srcdoc="{html.escape(page)}"></iframe>'


def map_update(feats):
    """Show the map only when the result has something to map; the table is always shown."""
    return gr.update(value=map_html(feats), visible=bool(feats))


def answer(tables, question, use_hints):
    tables = [table_name(t) for t in (tables or [])]
    if not tables or not question.strip():
        return "", "Pick at least one table and ask a question.", None, map_update([]), ""
    h = hints.value_hints(CON, tables, question) if use_hints else None
    prompt = student.user_prompt(CARDS, tables, question, h)
    try:
        sql, secs, timings = ask_model(student.SYSTEM_MINIMAL, prompt)
    except Exception as e:
        return "", f"Model not ready or failed: {e}", None, map_update([]), prompt
    speed = (f"prompt {timings.get('prompt_n', '?')} tokens at {timings.get('prompt_per_second', 0):.0f} tok/s, "
             f"answer {timings.get('predicted_n', '?')} tokens at {timings.get('predicted_per_second', 0):.0f} tok/s") if timings else ""
    status = f"Model: {secs:.1f} s ({speed})."
    try:
        t0 = time.time()
        df = run_sql(sql)
        status += f" Query: {len(df)} rows in {time.time() - t0:.2f} s" + (f" (first {MAX_FEATURES})" if len(df) == MAX_FEATURES else "") + "."
        return pretty(sql), status, df.head(MAX_ROWS), map_update(features(df)), prompt
    except Exception as e:
        return pretty(sql), status + f" Query failed: {str(e).splitlines()[0][:300]}", None, map_update([]), prompt


with gr.Blocks(title="Gazet SQL demo") as demo:
    gr.Markdown("## Gazet SQL demo\nA 0.8B model writes DuckDB spatial SQL for your question, on CPU. "
                "Tables marked *(new)* were never seen in training.")
    with gr.Row():
        with gr.Column(scale=2):
            tables = gr.Dropdown([table_label(t) for t in TABLES], value=[table_label(t) for t in ("power_plants", "countries")], multiselect=True, label="Tables")
            add_box = gr.MultimodalTextbox(
                label="Add your own data", file_count="multiple", file_types=UPLOAD_TYPES, submit_btn=True, lines=1,
                placeholder="Drop files here, click to upload, or paste a link to a Parquet / GeoParquet / GeoJSON file, then press Enter")
            upload_status = gr.Markdown()
            question = gr.Textbox(label="Question", lines=2, placeholder="e.g. Total solar capacity in MW per country in Africa")
            use_hints = gr.Checkbox(value=True, label="Value hints (look up names from the question in the data)")
            go = gr.Button("Write SQL and run it", variant="primary")
            gr.Examples([[[table_label(t) for t in ts], q] for ts, q in EXAMPLES], [tables, question])
        with gr.Column(scale=3):
            sql = gr.Code(label="SQL", language="sql")
            status = gr.Markdown()
            mapview = gr.HTML(visible=False)  # shown only when the result has geometry or lon/lat
            table = gr.Dataframe(wrap=True)
            with gr.Accordion("Prompt sent to the model", open=False):
                prompt = gr.Textbox(lines=20, show_label=False)
    add_box.submit(on_add, [add_box, tables], [tables, upload_status, add_box], show_progress="minimal")
    go.click(answer, [tables, question, use_hints], [sql, status, table, mapview, prompt], show_progress="minimal")
    question.submit(answer, [tables, question, use_hints], [sql, status, table, mapview, prompt], show_progress="minimal")

if __name__ == "__main__":
    for _ in range(120):  # wait for llama-server to load the model
        try:
            if requests.get(f"http://127.0.0.1:{PORT}/health", timeout=2).ok:
                break
        except requests.RequestException:
            pass
        time.sleep(1)
    demo.queue(default_concurrency_limit=1).launch(server_name="0.0.0.0", server_port=7860)
