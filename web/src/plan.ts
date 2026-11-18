/**
 * Structured query plans: a TypeScript port of src/gazet/plan.py.
 *
 * The model writes a JSON plan; this module validates it and compiles it to the same DuckDB SQL as the Python
 * compiler (checked by tests/plan.test.mjs against SQL that Python wrote). Table references stay symbolic
 * (read_parquet('divisions_area')); the caller rewrites them to the tables it loaded.
 */

export type Expr = Record<string, any>;

export const SOURCES: Record<string, string> = { divisions: 'divisions_area', natural: 'natural_earth' };
const PREDICATE_TEMPLATES: Record<string, string> = {
  touches: 'ST_Touches({f}, {o})',
  intersects: 'ST_Intersects({f}, {o})',
  within: 'ST_Within({f}, {o})',
  contains: 'ST_Contains({f}, {o})',
  not_intersects: 'ST_Intersects({f}, {o})', // negated below
};
export const PREDICATES = Object.keys(PREDICATE_TEMPLATES);
const SIDES: Record<string, string> = {
  north: 'xmin, (ymin + ymax) / 2.0, xmax, ymax',
  south: 'xmin, ymin, xmax, (ymin + ymax) / 2.0',
  east: '(xmin + xmax) / 2.0, ymin, xmax, ymax',
  west: 'xmin, ymin, (xmin + xmax) / 2.0, ymax',
};
const GROUPS = ['region', 'country'];
const EXCLUDING = ['touches', 'intersects'];
const FIND_KEYS = new Set([
  'find', 'subtype', 'country', 'is_land', 'is_territorial', 'largest', 'smallest', 'per', ...PREDICATES,
]);
const COLS = 'id, name, subtype, country, region, geometry';
export const DEG_PER_M = '111320.0';
const OPS = ['ids', 'find', 'buffer', 'union', 'intersection', 'difference', 'half'];
const GEOMETRY_OPS = ['buffer', 'union', 'intersection', 'difference', 'half'];

export class PlanError extends Error {}

// ── JSON schema (for constrained decoding) ───────────────────────────────────

const exprRef = () => ({ $ref: '#/$defs/expr' });
const list = (item: object, minItems = 1) => ({ type: 'array', items: item, minItems });

/** JSON schema of a plan. With candidate IDs, "ids" may only name those IDs. Same as plan_schema() in Python. */
export function planSchema(candidateIds?: string[]): object {
  const ids = [...new Set(candidateIds ?? [])].sort();
  const idItem = ids.length ? { enum: ids } : { type: 'string', minLength: 1 };
  const idList = ids.length ? { ...list(idItem), maxItems: ids.length } : list(idItem);
  const str = { type: 'string' };
  const spatial = { anyOf: [exprRef(), list(exprRef(), 2)] };
  const pair = { ...list(exprRef(), 2), maxItems: 2 };
  const obj = (properties: object, required: string[]) => ({
    type: 'object', properties, required, additionalProperties: false,
  });
  return {
    $defs: {
      expr: {
        anyOf: [
          obj({ ids: idList }, ['ids']),
          obj(
            {
              find: { enum: Object.keys(SOURCES) },
              subtype: list(str),
              country: list(str),
              is_land: { type: 'boolean' },
              is_territorial: { type: 'boolean' },
              ...Object.fromEntries(PREDICATES.map((p) => [p, spatial])),
              largest: { type: 'integer', minimum: 1 },
              smallest: { type: 'integer', minimum: 1 },
              per: { enum: GROUPS },
            },
            ['find'],
          ),
          obj({ buffer: exprRef(), m: { type: 'number' } }, ['buffer', 'm']),
          obj({ union: exprRef() }, ['union']),
          obj({ intersection: pair }, ['intersection']),
          obj({ difference: pair }, ['difference']),
          obj({ half: exprRef(), side: { enum: Object.keys(SIDES) } }, ['half', 'side']),
        ],
      },
    },
    $ref: '#/$defs/expr',
  };
}

// ── parsing and validation ───────────────────────────────────────────────────

/** Read a plan from model output (tolerates code fences and text around the JSON object). */
export function parsePlan(text: string): Expr {
  let s = text.trim();
  if (s.includes('```')) {
    const inner = s.startsWith('```') ? s.split('```')[1] ?? '' : s;
    s = inner.replace(/^```\w*\s*|\s*```$/g, '');
  }
  const start = s.indexOf('{');
  const end = s.lastIndexOf('}');
  if (start < 0 || end < start) throw new PlanError(`no JSON object in output: ${JSON.stringify(text.slice(0, 200))}`);
  let plan: unknown;
  try {
    plan = JSON.parse(s.slice(start, end + 1));
  } catch (e) {
    throw new PlanError(`invalid JSON: ${(e as Error).message}`);
  }
  validate(plan);
  return plan as Expr;
}

const asList = (v: any): any[] => (Array.isArray(v) ? v : [v]);
const isObj = (e: any): e is Expr => typeof e === 'object' && e !== null && !Array.isArray(e);
const isInt = (v: any) => typeof v === 'number' && Number.isInteger(v);
const keysEqual = (e: Expr, keys: string[]) =>
  Object.keys(e).length === keys.length && keys.every((k) => k in e);

export function opOf(e: Expr): string {
  for (const k of OPS) if (k in e) return k;
  throw new PlanError(`unknown expression with keys ${JSON.stringify(Object.keys(e).sort())}`);
}

export function validate(e: any, path = 'plan'): void {
  if (!isObj(e) || Object.keys(e).length === 0) throw new PlanError(`${path}: expected an object`);
  const kind = opOf(e);
  if (kind === 'ids') {
    const ids = asList(e.ids);
    if (!keysEqual(e, ['ids']) || !ids.length || !ids.every((i) => typeof i === 'string' && i)) {
      throw new PlanError(`${path}: 'ids' needs a list of ID strings`);
    }
  } else if (kind === 'find') {
    if (!(e.find in SOURCES)) throw new PlanError(`${path}: 'find' must be one of ${JSON.stringify(Object.keys(SOURCES))}`);
    const extra = Object.keys(e).filter((k) => !FIND_KEYS.has(k));
    if (extra.length) throw new PlanError(`${path}: unknown keys ${JSON.stringify(extra.sort())}`);
    for (const k of ['subtype', 'country']) {
      if (k in e && !asList(e[k]).every((v) => typeof v === 'string')) throw new PlanError(`${path}.${k}: expected strings`);
    }
    for (const k of ['is_land', 'is_territorial']) {
      if (k in e && typeof e[k] !== 'boolean') throw new PlanError(`${path}.${k}: expected true or false`);
    }
    for (const k of ['largest', 'smallest']) {
      if (k in e && !isInt(e[k])) throw new PlanError(`${path}.${k}: expected an integer`);
    }
    if ('per' in e && !GROUPS.includes(e.per)) throw new PlanError(`${path}.per: expected one of ${JSON.stringify(GROUPS)}`);
    for (const p of PREDICATES) {
      asList(e[p] ?? []).forEach((sub, i) => validate(sub, `${path}.${p}[${i}]`));
    }
  } else if (kind === 'buffer') {
    if (!keysEqual(e, ['buffer', 'm']) || typeof e.m !== 'number') {
      throw new PlanError(`${path}: 'buffer' needs an expression and 'm' (metres)`);
    }
    validate(e.buffer, `${path}.buffer`);
  } else if (kind === 'union') {
    if (!keysEqual(e, ['union'])) throw new PlanError(`${path}: 'union' takes one expression`);
    validate(e.union, `${path}.union`);
  } else if (kind === 'intersection' || kind === 'difference') {
    if (!keysEqual(e, [kind]) || !Array.isArray(e[kind]) || e[kind].length !== 2) {
      throw new PlanError(`${path}: '${kind}' takes a list of two expressions`);
    }
    e[kind].forEach((sub: any, i: number) => validate(sub, `${path}.${kind}[${i}]`));
  } else if (kind === 'half') {
    if (!keysEqual(e, ['half', 'side']) || !(e.side in SIDES)) {
      throw new PlanError(`${path}: 'half' needs an expression and 'side' in ${JSON.stringify(Object.keys(SIDES))}`);
    }
    validate(e.half, `${path}.half`);
  }
}

// ── compiler ─────────────────────────────────────────────────────────────────

const q = (s: any) => `'${String(s).replaceAll("'", "''")}'`;

function inList(col: string, values: any): string {
  const vs = asList(values);
  if (vs.length === 1) return `${col} = ${q(vs[0])}`;
  return `${col} IN (${vs.map(q).join(', ')})`;
}

export function sourceOfId(i: string, sources: Record<string, string>): string {
  if (i in sources) return sources[i];
  return i.startsWith('ne_') ? 'natural_earth' : 'divisions_area';
}

const table = (source: string, where: string) =>
  `SELECT id, names."primary" AS name, subtype, country, region, geometry FROM read_parquet('${source}') WHERE ${where}`;

/** IDs a spatial filter refers to directly (possibly grown by a buffer): excluded from the filter's result. */
function anchorIds(e: Expr): string[] {
  if ('ids' in e) return asList(e.ids);
  if ('buffer' in e) return anchorIds(e.buffer);
  return [];
}

/** Python's float() repr for the buffer distance: 250 -> "250.0", 1e-05 stays exponent-free enough for DuckDB. */
function pyFloat(v: number): string {
  if (Number.isInteger(v) && Math.abs(v) < 1e16) return `${v}.0`;
  return String(v);
}

function compile(e: Expr, sources: Record<string, string>): string {
  let sql = compileOp(e, sources);
  if (GEOMETRY_OPS.includes(opOf(e))) {
    sql = `SELECT * FROM (${sql}) WHERE geometry IS NOT NULL AND NOT ST_IsEmpty(geometry)`;
  }
  return sql;
}

/** Python's sorted() on strings: by code point, as JS's default sort on UTF-16 is for BMP text. */
const pySorted = (xs: string[]) => [...xs].sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));

function compileOp(e: Expr, sources: Record<string, string>): string {
  const kind = opOf(e);
  if (kind === 'ids') {
    const bySrc = new Map<string, string[]>();
    for (const i of asList(e.ids)) {
      const src = sourceOfId(i, sources);
      if (!bySrc.has(src)) bySrc.set(src, []);
      bySrc.get(src)!.push(i);
    }
    return [...bySrc].map(([src, ids]) => table(src, inList('id', ids))).join(' UNION ALL ');
  }

  if (kind === 'find') {
    const source = SOURCES[e.find];
    const conds: string[] = e.subtype && asList(e.subtype).length ? [inList('subtype', e.subtype)] : [];
    if (e.country && asList(e.country).length) conds.push(inList('country', e.country));
    for (const k of ['is_land', 'is_territorial']) {
      if (k in e) conds.push(`${k} = ${e[k] ? 'TRUE' : 'FALSE'}`);
    }
    const exclude = pySorted([
      ...new Set(
        EXCLUDING.flatMap((p) => asList(e[p] ?? []).flatMap((sub) => anchorIds(sub)))
          .filter((i) => sourceOfId(i, sources) === source),
      ),
    ]);
    if (exclude.length) conds.unshift(`NOT (${inList('id', exclude)})`);
    let body = table(source, conds.length ? conds.join(' AND ') : 'TRUE');
    for (const p of PREDICATES) {
      for (const sub of asList(e[p] ?? [])) {
        const test = PREDICATE_TEMPLATES[p].replace('{f}', 'f.geometry').replace('{o}', 'o.geometry');
        const join = p === 'not_intersects' ? 'ANTI JOIN' : 'SEMI JOIN';
        body = `SELECT f.* FROM (${body}) AS f ${join} (${compile(sub, sources)}) AS o ON ${test}`;
      }
    }
    const top = 'largest' in e ? 'largest' : 'smallest' in e ? 'smallest' : null;
    if (top === null) return body;
    const n = Math.max(1, e[top]);
    const order = top === 'largest' ? 'DESC' : 'ASC';
    if (e.per) {
      return (
        `SELECT ${COLS} FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY ${e.per} ` +
        `ORDER BY ST_Area(geometry) ${order}) AS rn FROM (${body})) WHERE rn <= ${Math.trunc(n)}`
      );
    }
    return `SELECT ${COLS} FROM (${body}) ORDER BY ST_Area(geometry) ${order} LIMIT ${Math.trunc(n)}`;
  }

  if (kind === 'buffer') {
    return (
      `SELECT id, name, subtype, country, region, ST_Buffer(geometry, ${pyFloat(e.m)} / ${DEG_PER_M}) AS geometry ` +
      `FROM (${compile(e.buffer, sources)})`
    );
  }

  if (kind === 'union') {
    return (
      'SELECT NULL AS id, NULL AS name, NULL AS subtype, NULL AS country, NULL AS region, ' +
      `ST_Union_Agg(geometry) AS geometry FROM (${compile(e.union, sources)})`
    );
  }

  if (kind === 'intersection' || kind === 'difference') {
    const [a, b] = e[kind].map((x: Expr) => compile(x, sources));
    const fn = kind === 'intersection' ? 'ST_Intersection' : 'ST_Difference';
    return (
      `SELECT a.id, a.name, a.subtype, a.country, a.region, ${fn}(a.geometry, b.geometry) AS geometry ` +
      `FROM (${a}) AS a, (${b}) AS b WHERE ST_Intersects(a.geometry, b.geometry)`
    );
  }

  if (kind === 'half') {
    const inner = compile(e.half, sources);
    return (
      `SELECT id, name, subtype, country, region, ST_Intersection(geometry, ST_MakeEnvelope(${SIDES[e.side]})) ` +
      `AS geometry FROM (SELECT *, ST_XMin(geometry) AS xmin, ST_XMax(geometry) AS xmax, ` +
      `ST_YMin(geometry) AS ymin, ST_YMax(geometry) AS ymax FROM (${inner}))`
    );
  }
  throw new PlanError(`unknown expression ${kind}`);
}

/** Plan -> one DuckDB SELECT returning id, name, subtype, country and the geometry as GeoJSON. */
export function compilePlan(plan: Expr, sources: Record<string, string> = {}): string {
  validate(plan);
  const body = compile(plan, sources);
  return (
    `SELECT id, name, subtype, country, ST_AsGeoJSON(geometry) AS geometry FROM (${body}) ` +
    'WHERE geometry IS NOT NULL AND NOT ST_IsEmpty(geometry)'
  );
}
