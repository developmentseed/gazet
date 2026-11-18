import * as maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import './style.css';
import { Db } from './db.ts';
import type { Candidate } from './format.ts';
import { Llm } from './llm.ts';
import { type Event, search } from './pipeline.ts';

const MODEL_URL = import.meta.env.VITE_MODEL_URL ?? './models/gazet-plan-v1-smol135-q8_0.gguf';
const DATA_URL = import.meta.env.VITE_DATA_URL ?? './data/';
const EXAMPLES = [
  'counties that border Loja',
  'northern half of Ecuador',
  '10 km buffer around Lake Titicaca',
  'regions of Peru that touch the Pacific Ocean',
  'the three largest counties in Albania',
  'border between Loja and Piura',
];

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const form = $<HTMLFormElement>('form');
const input = $<HTMLInputElement>('query');
const button = $<HTMLButtonElement>('go');
const steps = $('steps');
const status = $('status');

const map = new maplibregl.Map({
  container: 'map',
  style: 'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json',
  center: [0, 20],
  zoom: 1.5,
});
const mapReady = new Promise<void>((resolve) => map.on('load', () => resolve()));
mapReady.then(() => {
  map.addSource('result', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
  map.addLayer({ id: 'result-fill', type: 'fill', source: 'result', paint: { 'fill-color': '#2f7d6d', 'fill-opacity': 0.35 } });
  map.addLayer({ id: 'result-line', type: 'line', source: 'result', paint: { 'line-color': '#1d5247', 'line-width': 1.5 } });
  map.on('click', 'result-fill', (e: maplibregl.MapLayerMouseEvent) => {
    const p = e.features?.[0]?.properties;
    if (p) new maplibregl.Popup().setLngLat(e.lngLat).setText(`${p.name ?? 'result'} ${p.subtype ? `(${p.subtype})` : ''}`).addTo(map);
  });
});

function showResult(fc: GeoJSON.FeatureCollection) {
  (map.getSource('result') as maplibregl.GeoJSONSource).setData(fc);
  const b = new maplibregl.LngLatBounds();
  const visit = (c: any): void => (typeof c[0] === 'number' ? void b.extend(c as [number, number]) : c.forEach(visit));
  for (const f of fc.features) {
    const g = f.geometry as any;
    (g.type === 'GeometryCollection' ? g.geometries : [g]).forEach((x: any) => visit(x.coordinates));
  }
  if (!b.isEmpty()) map.fitBounds(b, { padding: 40, maxZoom: 10, duration: 800 });
}

function step(title: string, body: HTMLElement | string, meta = ''): void {
  const el = document.createElement('details');
  el.open = true;
  el.innerHTML = `<summary><span>${title}</span><span class="meta">${meta}</span></summary>`;
  if (typeof body === 'string') {
    const pre = document.createElement('pre');
    pre.textContent = body;
    el.append(pre);
  } else el.append(body);
  steps.append(el);
}

function candidatesTable(rows: Candidate[]): HTMLElement {
  const t = document.createElement('table');
  t.innerHTML = '<thead><tr><th>name</th><th>subtype</th><th>country</th><th>region</th><th>score</th></tr></thead>';
  const tb = document.createElement('tbody');
  for (const r of rows) {
    const tr = document.createElement('tr');
    tr.title = `${r.source} ${r.id}`;
    for (const v of [r.name, r.subtype, r.country, r.region, r.similarity?.toFixed(2)]) {
      const td = document.createElement('td');
      td.textContent = v ?? '';
      tr.append(td);
    }
    tb.append(tr);
  }
  t.append(tb);
  return t;
}

const secs = (s: number) => `${s.toFixed(1)} s`;

function render(ev: Event): void {
  switch (ev.type) {
    case 'places':
      step('Places', ev.places.join(' · '), `${secs(ev.answer.seconds)} · ${ev.answer.promptTokens} → ${ev.answer.outputTokens} tok`);
      break;
    case 'candidates':
      step('Candidates', candidatesTable(ev.candidates), secs(ev.seconds));
      break;
    case 'plan':
      step('Plan', JSON.stringify(ev.plan, null, 2), `${secs(ev.answer.seconds)} · ${ev.answer.promptTokens} → ${ev.answer.outputTokens} tok`);
      break;
    case 'sql': {
      step('SQL', ev.sql);
      (steps.lastElementChild as HTMLDetailsElement).open = false;
      break;
    }
    case 'status':
      status.textContent = ev.message;
      break;
    case 'result': {
      const n = ev.geojson.features.length;
      const a = document.createElement('a');
      a.textContent = 'Download GeoJSON';
      a.href = URL.createObjectURL(new Blob([JSON.stringify(ev.geojson)], { type: 'application/geo+json' }));
      a.download = 'gazet.geojson';
      step('Result', a, `${n} feature${n === 1 ? '' : 's'} · ${secs(ev.seconds)}`);
      showResult(ev.geojson);
      break;
    }
    case 'error':
      step('Error', ev.message);
      steps.lastElementChild?.classList.add('error');
      break;
  }
}

let llm: Llm;
let db: Db;
let busy = false;

async function run(query: string) {
  if (busy || !query.trim()) return;
  busy = true;
  button.disabled = true;
  steps.replaceChildren();
  status.textContent = 'Running…';
  const t = performance.now();
  try {
    for await (const ev of search(query, llm, db)) render(ev);
    status.textContent = `Done in ${secs((performance.now() - t) / 1000)}`;
  } catch (e) {
    render({ type: 'error', message: (e as Error).message });
    status.textContent = '';
    console.error(e);
  } finally {
    busy = false;
    button.disabled = false;
  }
}

form.addEventListener('submit', (e) => {
  e.preventDefault();
  run(input.value);
});
for (const q of EXAMPLES) {
  const b = document.createElement('button');
  b.type = 'button';
  b.textContent = q;
  b.addEventListener('click', () => {
    input.value = q;
    run(q);
  });
  $('examples').append(b);
}

async function init() {
  const progress = $<HTMLProgressElement>('model-progress');
  const dataStatus = $('data-status');
  try {
    [llm, db] = await Promise.all([
      Llm.load(MODEL_URL, (loaded, total) => {
        progress.value = total ? loaded / total : 0;
      }),
      Db.open(DATA_URL, (m) => {
        dataStatus.textContent = m;
      }),
      mapReady,
    ]).then(([l, d]) => [l, d] as const);
  } catch (e) {
    status.textContent = `Could not start: ${(e as Error).message}`;
    console.error(e);
    return;
  }
  $('loading').hidden = true;
  $('release').textContent = `Overture ${db.manifest.release}`;
  input.disabled = false;
  button.disabled = false;
  document.body.dataset.ready = 'true';
  input.focus();
  const q = new URLSearchParams(location.search).get('q');
  if (q) {
    input.value = q;
    run(q);
  }
}

init();

// For scripted runs (tests/e2e.mjs).
Object.assign(window, {
  gazet: {
    run: async (query: string) => {
      const events: Event[] = [];
      for await (const ev of search(query, llm, db)) events.push(ev);
      return events;
    },
    plan: (query: string, candidates: Candidate[]) => llm.plan(query, candidates),
  },
});
