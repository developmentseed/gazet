/** gazet's natural-language search, end to end in the browser (mirrors gazet.api._run_stream in plan mode). */

import type { Db } from './db.ts';
import { type Candidate, mergeCandidates, perSourceLimit } from './format.ts';
import type { Answer, Llm } from './llm.ts';
import { compilePlan, type Expr, parsePlan } from './plan.ts';

export type Event =
  | { type: 'places'; places: string[]; answer: Answer }
  | { type: 'candidates'; candidates: Candidate[]; seconds: number }
  | { type: 'plan'; plan: Expr; answer: Answer }
  | { type: 'sql'; sql: string }
  | { type: 'status'; message: string }
  | { type: 'result'; geojson: GeoJSON.FeatureCollection; seconds: number }
  | { type: 'error'; message: string };

export async function* search(query: string, llm: Llm, db: Db): AsyncGenerator<Event> {
  const { places, answer: placesAnswer } = await llm.places(query);
  yield { type: 'places', places, answer: placesAnswer };

  let t = performance.now();
  const limit = perSourceLimit(places.length);
  const lists: Candidate[][] = [];
  for (const place of places) {
    for (const source of ['divisions_area', 'natural_earth']) lists.push(await db.search(place, source, limit));
  }
  const candidates = mergeCandidates(lists);
  yield { type: 'candidates', candidates, seconds: (performance.now() - t) / 1000 };
  if (!candidates.length) {
    yield { type: 'error', message: 'No candidates found' };
    return;
  }

  const planAnswer = await llm.plan(query, candidates);
  let plan: Expr;
  try {
    plan = parsePlan(planAnswer.text);
  } catch (e) {
    yield { type: 'error', message: `The model did not write a valid plan: ${(e as Error).message}` };
    return;
  }
  yield { type: 'plan', plan, answer: planAnswer };

  const sources = Object.fromEntries(candidates.map((c) => [c.id, c.source]));
  const sql = compilePlan(plan, sources);
  yield { type: 'sql', sql };

  t = performance.now();
  const status: string[] = [];
  const need = await db.resolve(plan, sources);
  await db.fetchGeometry(need, (m) => status.push(m));
  for (const message of status) yield { type: 'status', message };
  let rows: Record<string, any>[];
  try {
    rows = await db.run(sql);
  } catch (e) {
    const message = (e as Error).message;
    if (/out of memory|allocation failure/i.test(message)) {
      await db.restart();
      yield {
        type: 'error',
        message:
          'The geometry operation ran out of browser memory. Overture geometry is full resolution, and some ' +
          'operations on large areas (such as wide buffers around long borders) are too heavy for a browser tab.',
      };
      return;
    }
    throw e;
  } finally {
    await db.clear().catch(() => {});
  }
  if (!rows.length) {
    yield { type: 'error', message: 'The plan returned no geometry' };
    return;
  }
  const geojson: GeoJSON.FeatureCollection = {
    type: 'FeatureCollection',
    features: rows.map((r) => ({
      type: 'Feature',
      geometry: JSON.parse(r.geometry),
      properties: { id: r.id, name: r.name, subtype: r.subtype, country: r.country },
    })),
  };
  yield { type: 'result', geojson, seconds: (performance.now() - t) / 1000 };
}
