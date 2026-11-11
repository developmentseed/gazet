"""Gazet SQL demo: question + tables -> DuckDB spatial SQL from the fine-tuned student, run and mapped.

The model runs in llama-server on CPU. The prompt is built by the same code as the training data
(student.py, hints.py, harness.py are copied in by deploy.sh), so what you see is what was evaluated.

Environment:
    MODEL_REPO / MODEL_FILE   GGUF to serve (default: developmentseed/gazet-sql-qwen35, sql-v1-r16/gguf/...q8_0.gguf)
    DATA_REPO                 dataset with lake/*.parquet (default: developmentseed/gazet-generative-sql)
    HF_TOKEN                  read access to both (Space secret)
    LLAMA_SERVER              llama-server binary (default /opt/llama/llama-server, as installed by the Dockerfile)
"""
import html
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path

import gradio as gr
import pandas as pd
import requests
from huggingface_hub import hf_hub_download, snapshot_download

HERE = Path(__file__).parent
MODEL_REPO = os.environ.get("MODEL_REPO", "developmentseed/gazet-sql-qwen35")
MODEL_FILE = os.environ.get("MODEL_FILE", "sql-v1-r16/gguf/gazet-sql-v1-r16-q8_0.gguf")
DATA_REPO = os.environ.get("DATA_REPO", "developmentseed/gazet-generative-sql")
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
TABLES = sorted(harness.TABLE_DOCS)
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


def table_label(t):
    return f"{t} (new)" if t in harness.HELD_OUT else t


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


def answer(tables, question, use_hints):
    tables = [t.replace(" (new)", "") for t in (tables or [])]
    if not tables or not question.strip():
        return "", "Pick at least one table and ask a question.", None, map_html([]), ""
    h = hints.value_hints(CON, tables, question) if use_hints else None
    prompt = student.user_prompt(CARDS, tables, question, h)
    try:
        sql, secs, timings = ask_model(student.SYSTEM_MINIMAL, prompt)
    except Exception as e:
        return "", f"Model not ready or failed: {e}", None, map_html([]), prompt
    speed = (f"prompt {timings.get('prompt_n', '?')} tokens at {timings.get('prompt_per_second', 0):.0f} tok/s, "
             f"answer {timings.get('predicted_n', '?')} tokens at {timings.get('predicted_per_second', 0):.0f} tok/s") if timings else ""
    status = f"Model: {secs:.1f} s ({speed})."
    try:
        t0 = time.time()
        df = run_sql(sql)
        status += f" Query: {len(df)} rows in {time.time() - t0:.2f} s" + (f" (first {MAX_FEATURES})" if len(df) == MAX_FEATURES else "") + "."
        return sql, status, df.head(MAX_ROWS), map_html(features(df)), prompt
    except Exception as e:
        return sql, status + f" Query failed: {str(e).splitlines()[0][:300]}", None, map_html([]), prompt


with gr.Blocks(title="Gazet SQL demo") as demo:
    gr.Markdown("## Gazet SQL demo\nA 0.8B model writes DuckDB spatial SQL for your question, on CPU. "
                "Tables marked *(new)* were never seen in training.")
    with gr.Row():
        with gr.Column(scale=2):
            tables = gr.Dropdown([table_label(t) for t in TABLES], value=["power_plants", "countries"], multiselect=True, label="Tables")
            question = gr.Textbox(label="Question", lines=2, placeholder="e.g. Total solar capacity in MW per country in Africa")
            use_hints = gr.Checkbox(value=True, label="Value hints (look up names from the question in the data)")
            go = gr.Button("Write SQL and run it", variant="primary")
            gr.Examples([[[table_label(t) for t in ts], q] for ts, q in EXAMPLES], [tables, question])
        with gr.Column(scale=3):
            sql = gr.Code(label="SQL", language="sql")
            status = gr.Markdown()
            with gr.Tab("Map"):
                mapview = gr.HTML(map_html([]))
            with gr.Tab("Table"):
                table = gr.Dataframe(wrap=True)
            with gr.Tab("Prompt sent to the model"):
                prompt = gr.Textbox(lines=20, show_label=False)
    go.click(answer, [tables, question, use_hints], [sql, status, table, mapview, prompt])
    question.submit(answer, [tables, question, use_hints], [sql, status, table, mapview, prompt])

if __name__ == "__main__":
    for _ in range(120):  # wait for llama-server to load the model
        try:
            if requests.get(f"http://127.0.0.1:{PORT}/health", timeout=2).ok:
                break
        except requests.RequestException:
            pass
        time.sleep(1)
    demo.queue(default_concurrency_limit=1).launch(server_name="0.0.0.0", server_port=7860)
