/**
 * DuckDB-WASM side of the demo: the place index, fuzzy search, and running compiled plans.
 *
 * Geometry is never loaded whole. Before a plan runs, `resolve` walks it against the in-memory index (names,
 * attributes, bbox, area; no geometry) to find every feature the plan could touch; `fetchGeometry` reads only those,
 * divisions from Overture's S3 bucket and Natural Earth from our file, with bbox filters that let DuckDB skip the
 * row groups elsewhere. The compiled SQL then runs unchanged against the two small tables.
 */

import * as duckdb from '@duckdb/duckdb-wasm';
import * as arrow from 'apache-arrow';
import ehWorker from '@duckdb/duckdb-wasm/dist/duckdb-browser-eh.worker.js?url';
import mvpWorker from '@duckdb/duckdb-wasm/dist/duckdb-browser-mvp.worker.js?url';
import ehWasm from '@duckdb/duckdb-wasm/dist/duckdb-eh.wasm?url';
import mvpWasm from '@duckdb/duckdb-wasm/dist/duckdb-mvp.wasm?url';
import type { Candidate } from './format.ts';
import { GeoReader } from './georeader.ts';
import { DEG_PER_M, type Expr, opOf, PREDICATES, SOURCES, sourceOfId } from './plan.ts';

export interface Manifest {
  release: string;
  division_area: string[];
  natural_earth: string;
  index: string;
}

export interface Box {
  xmin: number;
  ymin: number;
  xmax: number;
  ymax: number;
}

/** Most features a plan may fetch geometry for; beyond this the browser would download too much. */
export const MAX_FEATURES = 3000;
/** Most compressed geometry bytes a plan may fetch. */
export const MAX_DOWNLOAD = 1.2e9;

export class Db {
  private constructor(
    private db: duckdb.AsyncDuckDB,
    private conn: duckdb.AsyncDuckDBConnection,
    readonly manifest: Manifest,
    readonly dataUrl: string,
    private readonly index: Uint8Array,
    readonly reader = new GeoReader(),
  ) {}

  /** A DuckDB instance with spatial loaded and the place index in table `idx`. */
  private static async start(index: Uint8Array, log: (msg: string) => void) {
    const bundle = await duckdb.selectBundle({
      mvp: { mainModule: mvpWasm, mainWorker: mvpWorker },
      eh: { mainModule: ehWasm, mainWorker: ehWorker },
    });
    const db = new duckdb.AsyncDuckDB(new duckdb.VoidLogger(), new Worker(bundle.mainWorker!));
    await db.instantiate(bundle.mainModule, bundle.pthreadWorker);
    const conn = await db.connect();
    log(`DuckDB ${await db.getVersion()}: loading spatial`);
    await conn.query('INSTALL spatial; LOAD spatial;');
    await db.registerFileBuffer('index.parquet', index.slice());
    await conn.query(`CREATE TABLE idx AS SELECT * FROM read_parquet('index.parquet')`);
    return { db, conn };
  }

  static async open(dataUrl: string, log: (msg: string) => void): Promise<Db> {
    const base = new URL(dataUrl, location.href).href;
    const manifest: Manifest = await (await fetch(new URL('manifest.json', base))).json();
    log(`loading place index (Overture ${manifest.release})`);
    // Read whole (10 MB): one request, instead of DuckDB's many range requests. Kept for restarts.
    const index = new Uint8Array(await (await fetch(new URL(manifest.index, base))).arrayBuffer());
    const { db, conn } = await Db.start(index, log);
    return new Db(db, conn, manifest, base, index);
  }

  /** Replace the DuckDB instance: after running out of memory, WebAssembly memory cannot be recovered. */
  async restart(): Promise<void> {
    await this.db.terminate().catch(() => {});
    ({ db: this.db, conn: this.conn } = await Db.start(this.index, () => {}));
  }

  async rows<T = Record<string, any>>(sql: string): Promise<T[]> {
    const table = await this.conn.query(sql);
    return table.toArray().map((r) => normalize(r.toJSON())) as T[];
  }

  /** Fuzzy search on one source, as gazet.search.simple_fuzzy_search does on the parquet files. */
  async search(place: string, source: string, limit: number): Promise<Candidate[]> {
    const query = `lower(strip_accents(${lit(place)}))`;
    return this.rows<Candidate>(`
      WITH scored AS (
        SELECT *, list_max(list_transform(search_names, (s, i) -> {
            'substring': contains(s.folded, ${query}),
            'similarity': jaro_winkler_similarity(s.folded, ${query}),
            'rank': -i,
            'name': s.name
          })) AS best
        FROM idx WHERE source = ${lit(source)} AND len(search_names) > 0
      )
      SELECT source, id, name, subtype, country, region, admin_level, population, bbox,
             best.name AS matched_name, best.similarity AS similarity, best.substring AS is_substring_match
      FROM scored
      ORDER BY best.substring DESC, best.similarity DESC, population DESC NULLS LAST, admin_level ASC, id
      LIMIT ${Math.trunc(limit)}`);
  }

  /** Every feature a plan could touch, by source, found on the index alone. */
  async resolve(plan: Expr, sources: Record<string, string>): Promise<Map<string, Set<string>>> {
    const need = new Map<string, Set<string>>([
      ['divisions_area', new Set()],
      ['natural_earth', new Set()],
    ]);
    const add = (src: string, id: string) => need.get(src)!.add(id);

    const boxOf = async (ids: string[]): Promise<Box | null> => {
      if (!ids.length) return null;
      const [r] = await this.rows<Box>(
        `SELECT min(bbox.xmin) AS xmin, min(bbox.ymin) AS ymin, max(bbox.xmax) AS xmax, max(bbox.ymax) AS ymax
         FROM idx WHERE id IN (${ids.map(lit).join(', ')})`,
      );
      return r && r.xmin !== null ? r : null;
    };

    // Returns the bbox the expression's features can lie in (null: no features).
    const walk = async (e: Expr): Promise<Box | null> => {
      const kind = opOf(e);
      if (kind === 'ids') {
        const ids: string[] = Array.isArray(e.ids) ? e.ids : [e.ids];
        for (const id of ids) add(sourceOfId(id, sources), id);
        return boxOf(ids);
      }
      if (kind === 'find') {
        const source = SOURCES[e.find];
        const conds = [`source = ${lit(source)}`];
        if (e.subtype?.length) conds.push(`subtype IN (${[e.subtype].flat().map(lit).join(', ')})`);
        if (e.country?.length) conds.push(`country IN (${[e.country].flat().map(lit).join(', ')})`);
        for (const k of ['is_land', 'is_territorial']) if (k in e) conds.push(`${k} = ${e[k] ? 'TRUE' : 'FALSE'}`);
        let spatial = false;
        for (const p of PREDICATES) {
          for (const sub of [e[p] ?? []].flat()) {
            const box = await walk(sub);
            if (p === 'not_intersects') continue;
            spatial = true;
            if (!box) return null; // relates to nothing: no feature can match
            conds.push(overlaps(box));
          }
        }
        const top = 'largest' in e ? 'largest' : 'smallest' in e ? 'smallest' : null;
        const where = conds.join(' AND ');
        let sql = `SELECT id, bbox FROM idx WHERE ${where}`;
        if (top && !spatial) {
          // Ranking needs only the area, which the index stores (ST_Area, as the SQL ranks): fetch just the winners.
          const order = top === 'largest' ? 'DESC' : 'ASC';
          const part = e.per ? `PARTITION BY ${e.per} ` : '';
          sql = `SELECT id, bbox FROM (SELECT id, bbox, ROW_NUMBER() OVER (${part}ORDER BY area ${order}) AS rn
                 FROM idx WHERE ${where}) WHERE rn <= ${Math.max(1, Math.trunc(e[top]))}`;
        }
        const found = await this.rows<{ id: string; bbox: Box }>(sql);
        if (found.length + need.get(source)!.size > MAX_FEATURES) {
          throw new Error(
            `This query needs geometry for ${found.length.toLocaleString()} features; ` +
              `the browser demo stops at ${MAX_FEATURES.toLocaleString()}. Try a smaller area.`,
          );
        }
        for (const f of found) add(source, f.id);
        return union(found.map((f) => f.bbox));
      }
      if (kind === 'buffer') {
        const box = await walk(e.buffer);
        const d = Math.abs(e.m) / Number(DEG_PER_M);
        return box && { xmin: box.xmin - d, ymin: box.ymin - d, xmax: box.xmax + d, ymax: box.ymax + d };
      }
      if (kind === 'union') return walk(e.union);
      if (kind === 'half') return walk(e.half);
      const [a, b] = [await walk(e[kind][0]), await walk(e[kind][1])];
      if (kind === 'difference') return a;
      return a && b ? intersect(a, b) : null;
    };

    await walk(plan);
    return need;
  }

  /**
   * Load the geometry of the given features into tables `divisions_area` and `natural_earth`.
   *
   * DuckDB-WASM falls back to downloading whole files from S3 (hundreds of MB), so geometry is read with hyparquet
   * instead: the index knows each feature's file and row, and only the `id` and `geometry` column chunks of the row
   * groups holding them are fetched, with HTTP range requests. Attributes come from the index.
   */
  async fetchGeometry(need: Map<string, Set<string>>, log: (msg: string) => void): Promise<void> {
    const ids = (src: string) => [...need.get(src)!].map(lit).join(', ') || 'NULL';
    const located = await this.rows<{ source: string; id: string; file: number | null; row: number }>(`
      SELECT source, id, file, "row" FROM idx
      WHERE (source = 'divisions_area' AND id IN (${ids('divisions_area')}))
         OR (source = 'natural_earth' AND id IN (${ids('natural_earth')}))`);

    const byFile = new Map<string, typeof located>();
    for (const r of located) {
      const url =
        r.source === 'divisions_area'
          ? this.manifest.division_area[r.file!]
          : new URL(this.manifest.natural_earth, this.dataUrl).href;
      if (!byFile.has(url)) byFile.set(url, []);
      byFile.get(url)!.push(r);
    }
    const files = [...byFile];
    let estimate = 0;
    for (const [url, rows] of files) estimate += await this.reader.estimate(url, rows.map((r) => r.row));
    if (estimate > MAX_DOWNLOAD) {
      throw new Error(
        `This query needs ${(estimate / 1e6).toFixed(0)} MB of geometry from Overture; the browser demo stops at ` +
          `${(MAX_DOWNLOAD / 1e6).toFixed(0)} MB (country-scale geometry is large). Try a smaller area.`,
      );
    }
    log(`reading ${located.length} geometries (about ${(estimate / 1e6).toFixed(0)} MB)`);
    const t = performance.now();
    const before = this.reader.bytes;
    const source: string[] = [];
    const fileCol: (number | null)[] = [];
    const rowCol: number[] = [];
    const wkb: Uint8Array[] = [];
    await Promise.all(
      files.map(async ([url, rows]) => {
        const feats = await this.reader.read(url, rows.map((r) => r.row));
        for (const r of rows) {
          const f = feats.get(r.row);
          if (f?.id !== r.id) throw new Error(`row ${r.row} of ${url} is ${f?.id}, expected ${r.id}`);
          source.push(r.source);
          fileCol.push(r.file);
          rowCol.push(r.row);
          wkb.push(f.wkb);
        }
      }),
    );
    const mb = (this.reader.bytes - before) / 1e6;
    log(`read ${located.length} geometries (${mb.toFixed(1)} MB) in ${((performance.now() - t) / 1000).toFixed(1)} s`);

    await this.conn.query('DROP TABLE IF EXISTS fetched');
    await this.conn.insertArrowTable(
      new arrow.Table({
        source: arrow.vectorFromArray(source, new arrow.Utf8()),
        file: arrow.vectorFromArray(fileCol, new arrow.Int32()),
        row: arrow.vectorFromArray(rowCol, new arrow.Float64()),
        wkb: arrow.vectorFromArray(wkb, new arrow.Binary()),
      }),
      { name: 'fetched', create: true },
    );
    for (const src of ['divisions_area', 'natural_earth']) {
      await this.conn.query(`CREATE OR REPLACE TEMP TABLE ${src} AS
        SELECT i.id, {'primary': i.primary_name} AS names, i.subtype, i.country, i.region, i.is_land, i.is_territorial,
               ST_GeomFromWKB(f.wkb) AS geometry
        FROM fetched AS f JOIN idx AS i
          ON i.source = f.source AND i."row" = CAST(f."row" AS BIGINT) AND i.file IS NOT DISTINCT FROM f.file
        WHERE f.source = '${src}'`);
    }
    await this.conn.query('DROP TABLE fetched');
  }

  /** Free the geometry of the last query. */
  async clear(): Promise<void> {
    await this.conn.query('DROP TABLE IF EXISTS divisions_area; DROP TABLE IF EXISTS natural_earth;');
  }

  /** Run compiled plan SQL against the loaded tables. */
  async run(sql: string): Promise<Record<string, any>[]> {
    const local = sql
      .replaceAll("read_parquet('divisions_area')", 'divisions_area')
      .replaceAll("read_parquet('natural_earth')", 'natural_earth');
    return this.rows(local);
  }
}

const lit = (s: string) => `'${String(s).replaceAll("'", "''")}'`;

/** bbox-overlap condition on the index's (or Overture's) bbox column. */
function overlaps(b: Box): string {
  return `bbox.xmin <= ${b.xmax} AND bbox.xmax >= ${b.xmin} AND bbox.ymin <= ${b.ymax} AND bbox.ymax >= ${b.ymin}`;
}

function union(boxes: Box[]): Box | null {
  if (!boxes.length) return null;
  return {
    xmin: Math.min(...boxes.map((b) => b.xmin)),
    ymin: Math.min(...boxes.map((b) => b.ymin)),
    xmax: Math.max(...boxes.map((b) => b.xmax)),
    ymax: Math.max(...boxes.map((b) => b.ymax)),
  };
}

function intersect(a: Box, b: Box): Box | null {
  const r = {
    xmin: Math.max(a.xmin, b.xmin),
    ymin: Math.max(a.ymin, b.ymin),
    xmax: Math.min(a.xmax, b.xmax),
    ymax: Math.min(a.ymax, b.ymax),
  };
  return r.xmin <= r.xmax && r.ymin <= r.ymax ? r : null;
}

/** Arrow values to plain JS: BigInt to number, nested vectors and structs to arrays and objects. */
function normalize(v: any): any {
  if (typeof v === 'bigint') return Number(v);
  if (v === null || typeof v !== 'object') return v;
  if (typeof v.toArray === 'function' && !Array.isArray(v)) return [...v.toArray()].map(normalize);
  if (Array.isArray(v)) return v.map(normalize);
  if (typeof v.toJSON === 'function') v = v.toJSON();
  return Object.fromEntries(Object.entries(v).map(([k, x]) => [k, normalize(x)]));
}
