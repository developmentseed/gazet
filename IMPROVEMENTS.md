# Gazet Improvement Notes

Issues identified during testing. Each item is a candidate for the next training/template pass.

---

## 1. Missing "buffer-only" template

**Query**: "10 km buffer around Odisha"
**Expected**: Return the buffered geometry polygon itself.
**Actual**: Model picks `buffer_01`, which finds all features intersecting the buffer (200 rows).

**Root cause**: All buffer templates (`buffer_01` through `buffer_05`) perform an intersection join to find neighboring features. No template simply returns `ST_AsGeoJSON(ST_Buffer(...))`.

**Fix**: Add a `buffer_06` template that returns the buffer polygon directly:

```sql
SELECT ST_AsGeoJSON(ST_Buffer(geometry, {buffer_km} * 1000.0 / 111320.0)) AS geometry
FROM read_parquet('divisions_area')
WHERE id = '{anchor_id}'
```

With hints like "10 km buffer around {anchor_name}", "draw a {buffer_km} km buffer around {anchor_name}". Consider a NE variant too.

---

## 2. Place extractor misses NE physical features (mixed-source queries)

**Query**: "The part of Ecuador that is in the Amazon Basin"
**Expected**: Place extractor returns both "Ecuador" and "Amazon Basin"; candidate search finds correct IDs for both.
**Actual**: Only "Ecuador" extracted. SQL model uses memorized wrong NE ID (`ne_1159120655` = Cuando River) instead of the correct one (`ne_1159104325` = AMAZON BASIN).

**Root cause**: The GGUF place extraction model was not trained to extract physical features. The runtime prompt (`_PLACES_SYSTEM_PROMPT`) has been updated but the finetuned model may ignore prompt changes. A re-finetune with NE feature examples is the definitive fix.

**Affected templates**: `partial_05`, `diff_02` (mixed-source), and all NE-anchored templates (`intersect_03`, `contain_03/04`, `buffer_03/04/05`, `lookup_02`).

---

## 3. Missing NE-anchor to county intersection template

**Query**: "Indravati River flows through which districts"
**Expected**: `ST_Intersects` with `target_subtype='county'`
**Actual**: Model sometimes uses `ST_Within` (wrong predicate) because `intersect_03` only targets `region`, not `county`.

**Fix**: Add `intersect_05` (NE anchor -> county, `ST_Intersects`) with district-oriented question hints.

---

## 4. Model hallucinates NE subtype values

**Query**: "which mountain ranges cross Odisha"
**Expected**: `n.subtype IN ('range/mtn', 'peninsula', 'depression')` (from `adj_05`)
**Actual**: Model generates `'Terrain area'` which does not exist in the data.

**Fix**: More training examples for `adj_05`. Consider adding common hallucinated values to `_NE_SUBTYPE_FIXES` in `sql.py` as a runtime safety net.

---

## 5. NE subtype casing inconsistency between model output and data

**Example**: Model generates `'River'`, `'Basin'`, `'Ocean'` but data has `'river'`, `'basin'`, `'ocean'`.

**Current workaround**: `_normalize_ne_subtypes()` in `sql.py` does string replacement of known title-cased literals at query time (`_NE_SUBTYPE_FIXES` dict). This is brittle and only covers a hardcoded list.

**Root cause**: The original Natural Earth data had title-cased `featurecla` values (e.g. `River`, `Basin`, `Ocean`). Training data was generated before the lowercase fix to `convert_natural_earth.py`, so the model learned to emit title-cased subtypes. The data is now lowercased but the model still outputs the old casing.

**Fix**: Regenerate training data with the lowercased NE parquet so all subtype literals in SQL examples are lowercase. After re-finetune, the model will natively emit lowercase subtypes and the `_normalize_ne_subtypes` hack can be removed.

---

## 6. "Largest/smallest" queries always return at least 3 results

**Query**: "the largest region in India", "smallest county in France"
**Expected**: Return 1 result (the single largest/smallest).
**Actual**: Model generates `LIMIT 3` by default, returning top 3 instead of 1.

**Root cause**: The aggregation templates (`agg_01`, `agg_02`) use `LIMIT 3` as the default. The model learns this as a fixed pattern and applies it even when the query clearly asks for a single result ("the largest", "the smallest").

**Fix**: During data generation, vary the LIMIT value based on the question hint phrasing. Use `LIMIT 1` for singular hints ("the largest X", "the smallest X") and `LIMIT 3` or `LIMIT 5` for plural hints ("the 3 largest", "top 5 smallest"). This teaches the model to infer the correct LIMIT from the query.
