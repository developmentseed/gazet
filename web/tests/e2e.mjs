// Browser checks in headless Chromium against the Vite dev server.
//
//   node tests/e2e.mjs model          replay gazet_compat prompts through the browser model (tests/model-check.html)
//   node tests/e2e.mjs search [q...]  run full searches on the demo page (model, index, Overture S3)
//   PREVIEW=1 node tests/e2e.mjs ...  the same against the production build in dist/
//
// Needs public/models/ and public/data/ (see README.md).
import { writeFileSync } from 'node:fs';
import { chromium } from 'playwright';
import { createServer, preview } from 'vite';

const [mode = 'model', ...queries] = process.argv.slice(2);
// PREVIEW=1 tests the production build (npm run build first) instead of the dev server; BASE_URL=<url> tests a page
// served elsewhere (e.g. dist/ from a plain static server, as on GitHub Pages, or the deployed site).
const server = process.env.BASE_URL ? null : process.env.PREVIEW
  ? await preview({ preview: { port: 5199 }, logLevel: 'warn' })
  : await createServer({ server: { port: 5199 }, logLevel: 'warn' });
if (server && !process.env.PREVIEW) await server.listen();
const base = process.env.BASE_URL ?? 'http://localhost:5199';
const browser = await chromium.launch();
const page = await browser.newPage();
page.on('console', (m) => m.type() === 'error' && console.error('[page]', m.text()));
page.on('pageerror', (e) => console.error('[page error]', e.message));

try {
  if (mode === 'model') {
    await page.goto(`${base}/tests/model-check.html${process.env.CHECK_QUERY ?? ''}`);
    const r = await page.waitForFunction(() => window.result, null, { timeout: 30 * 60_000, polling: 1000 }).then((h) => h.jsonValue());
    writeFileSync('test-results-model.json', JSON.stringify(r, null, 1));
    const pct = (xs, f) => `${xs.filter(f).length}/${xs.length}`;
    const med = (xs) => xs.map((x) => x).sort((a, b) => a - b)[Math.floor(xs.length / 2)].toFixed(2);
    console.log(`isolated=${r.isolated} threads=${r.threads} model load ${r.loadSeconds.toFixed(1)} s`);
    console.log(`plans: same SQL as GPU ${pct(r.plans, (p) => p.same)}, invalid ${pct(r.plans, (p) => p.error)}, ` +
      `median ${med(r.plans.map((p) => p.seconds))} s, prompt ~${med(r.plans.map((p) => p.promptTokens))} tok`);
    console.log(`places: same as GPU ${pct(r.places, (p) => p.same)}, correct ${pct(r.places, (p) => p.correct)}, ` +
      `median ${med(r.places.map((p) => p.seconds))} s`);
    for (const p of r.plans.filter((p) => !p.same)) console.log('  plan differs', p.idx, p.family, p.text, p.error ?? '');
  } else {
    await page.goto(base);
    // coi-serviceworker reloads the page once to become cross-origin isolated; wait for that before using it.
    await page.waitForFunction(() => window.crossOriginIsolated, null, { timeout: 60_000 });
    console.log(`crossOriginIsolated=${await page.evaluate(() => window.crossOriginIsolated)}`);
    await page.waitForFunction(() => document.body.dataset.ready === 'true', null, { timeout: 10 * 60_000 });
    const all = [];
    for (const q of queries.length ? queries : ['counties that border Loja']) {
      const t = Date.now();
      const events = await page.evaluate((q) => window.gazet.run(q), q);
      const get = (type) => events.find((e) => e.type === type);
      console.log(`\n${q}  (${((Date.now() - t) / 1000).toFixed(1)} s)`);
      console.log('  places', JSON.stringify(get('places')?.places), get('places')?.answer.seconds.toFixed(2), 's');
      console.log('  candidates', get('candidates')?.candidates.slice(0, 4).map((c) => `${c.name}/${c.subtype}/${c.country}`).join(', '));
      console.log('  plan', JSON.stringify(get('plan')?.plan), get('plan')?.answer.seconds.toFixed(2), 's');
      for (const e of events.filter((e) => e.type === 'status')) console.log('  ', e.message);
      const res = get('result');
      console.log(res ? `  result: ${res.geojson.features.length} features, ${res.seconds.toFixed(1)} s data` : `  ERROR ${get('error')?.message}`);
      all.push({ query: q, plan: get('plan')?.plan, sql: get('sql')?.sql,
                 ids: res?.geojson.features.map((f) => f.properties.id) ?? [],
                 areas: res?.geojson.features.map((f) => f.geometry) ? undefined : undefined });
    }
    writeFileSync('test-results-search.json', JSON.stringify(all, null, 1));
    await page.screenshot({ path: 'test-results-search.png' });
  }
} finally {
  await browser.close();
  if (server) await (server.close?.() ?? server.httpServer.close());
}
