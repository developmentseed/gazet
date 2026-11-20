/** Text the model reads and writes, formatted exactly as gazet's Python pipeline does (src/gazet/lm.py). */

import { USER_PROMPT_TEMPLATE } from './prompts.ts';

export interface Candidate {
  source: string;
  id: string;
  name: string | null;
  subtype: string | null;
  country: string | null;
  region: string | null;
  admin_level: number | null;
  similarity?: number;
  is_substring_match?: boolean;
  matched_name?: string | null;
  population?: number | null;
  bbox?: { xmin: number; ymin: number; xmax: number; ymax: number };
}

/** The columns the model was trained on, in order. */
export const CANDIDATE_COLUMNS = ['source', 'id', 'name', 'subtype', 'country', 'region', 'admin_level'] as const;

/** One CSV field as pandas' to_csv writes it (QUOTE_MINIMAL; missing values empty). */
function csvField(v: unknown): string {
  if (v === null || v === undefined || (typeof v === 'number' && Number.isNaN(v))) return '';
  const s = String(v);
  return /[",\n\r]/.test(s) ? `"${s.replaceAll('"', '""')}"` : s;
}

/** The candidates table as the model saw it in training: header plus one line per candidate, no trailing newline. */
export function candidatesCsv(rows: Candidate[]): string {
  const lines = [CANDIDATE_COLUMNS.join(',')];
  for (const r of rows) lines.push(CANDIDATE_COLUMNS.map((c) => csvField(r[c])).join(','));
  return lines.join('\n');
}

export function planUserPrompt(question: string, candidates: Candidate[]): string {
  return USER_PROMPT_TEMPLATE.replace('{candidates_csv}', candidatesCsv(candidates)).replace(
    '{question}',
    question.trim(),
  );
}

/** Place names from the model's answer; the whole query as one place when the answer is not valid JSON. */
export function parsePlaces(raw: string, query: string): string[] {
  let s = raw.trim();
  if (s.startsWith('```')) s = (s.split('```')[1] ?? '').replace(/^json/, '').trim();
  try {
    const data = JSON.parse(s);
    const places = (data?.places ?? []).map((p: { place?: unknown }) => p?.place);
    if (Array.isArray(data?.places) && places.every((p: unknown) => typeof p === 'string')) return places;
  } catch {
    // fall through
  }
  return [query];
}

/** Candidates per source per place, as gazet.api._per_source_limit. */
export function perSourceLimit(numPlaces: number): number {
  if (numPlaces <= 1) return 5;
  if (numPlaces <= 2) return 4;
  return 3;
}

/** Merge per-place candidate lists as gazet.api does: drop repeats, then substring hits, similarity, admin level. */
export function mergeCandidates(lists: Candidate[][]): Candidate[] {
  const seen = new Set<string>();
  const out: Candidate[] = [];
  for (const r of lists.flat()) {
    const key = `${r.source}\u0000${r.id}`;
    if (!seen.has(key)) {
      seen.add(key);
      out.push(r);
    }
  }
  const lvl = (r: Candidate) => (r.admin_level ?? Number.POSITIVE_INFINITY);
  return out.sort(
    (a, b) =>
      Number(b.is_substring_match ?? false) - Number(a.is_substring_match ?? false) ||
      (b.similarity ?? 0) - (a.similarity ?? 0) ||
      lvl(a) - lvl(b),
  );
}
