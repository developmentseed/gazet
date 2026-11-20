// The browser prompts must equal the ones in src/gazet/lm.py that the model was trained on.
// Fixture: written from gazet.lm (see web/README.md).
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { PLACES_SYSTEM_PROMPT, PLAN_SYSTEM_PROMPT, USER_PROMPT_TEMPLATE } from '../src/prompts.ts';
import { candidatesCsv, planUserPrompt, parsePlaces } from '../src/format.ts';

const fx = JSON.parse(readFileSync(new URL('./fixtures/prompts.json', import.meta.url), 'utf8'));

test('prompts equal gazet.lm', () => {
  assert.equal(PLACES_SYSTEM_PROMPT, fx.places_system);
  assert.equal(PLAN_SYSTEM_PROMPT, fx.plan_system);
  assert.equal(USER_PROMPT_TEMPLATE, fx.user_template);
});

test('plan user prompt matches an evaluation prompt', () => {
  for (const p of fx.examples) {
    assert.equal(planUserPrompt(p.question, p.candidates), p.user);
  }
});

test('candidates CSV quotes like pandas', () => {
  const csv = candidatesCsv([
    { source: 'divisions_area', id: 'a', name: 'Bonaire, Sint Eustatius and Saba', subtype: 'country', country: 'BQ', region: null, admin_level: 0 },
    { source: 'natural_earth', id: 'ne_1', name: 'Say "hi"', subtype: 'river', country: null, region: null, admin_level: null },
  ]);
  assert.equal(csv, 'source,id,name,subtype,country,region,admin_level\ndivisions_area,a,"Bonaire, Sint Eustatius and Saba",country,BQ,,0\nnatural_earth,ne_1,"Say ""hi""",river,,,');
});

test('places parsing falls back to the whole query', () => {
  assert.deepEqual(parsePlaces('```json\n{"places": [{"place": "Loja"}]}\n```', 'q'), ['Loja']);
  assert.deepEqual(parsePlaces('not json', 'north half of Loja'), ['north half of Loja']);
});
