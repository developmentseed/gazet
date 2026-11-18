// The TypeScript plan compiler must give the same SQL, schema and errors as Python's gazet.plan.
// Fixtures: scripts/make_plan_fixtures.py.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { compilePlan, parsePlan, planSchema } from '../src/plan.ts';

const fx = JSON.parse(readFileSync(new URL('./fixtures/plans.json', import.meta.url), 'utf8'));

test('compiled SQL equals Python for every fixture plan', () => {
  for (const c of fx.cases) {
    assert.equal(compilePlan(c.plan), c.sql, JSON.stringify(c.plan));
    assert.equal(compilePlan(c.plan, fx.sources), c.sql_sources, JSON.stringify(c.plan));
  }
});

test('parsing accepts and rejects the same texts', () => {
  for (const p of fx.parse) {
    if (p.error) assert.throws(() => parsePlan(p.text), p.text);
    else assert.deepEqual(parsePlan(p.text), p.plan);
  }
});

test('schema equals Python', () => {
  assert.deepEqual(planSchema(['b', 'a', 'ne_1', 'a']), fx.schema);
  assert.deepEqual(planSchema(), fx.schema_open);
});
