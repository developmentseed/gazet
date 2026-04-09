"""
SQL template definitions for synthetic data generation.

Geometry output convention
--------------------------
Every final SELECT wraps geometry with ST_AsGeoJSON():
    ST_AsGeoJSON(geometry) AS geometry
This returns a GeoJSON string instead of raw WKB bytes, which is directly
JSON-serialisable and matches what the serving stack expects.

CTEs that compute intermediate geometries (used only for spatial predicates
or ST_Area) keep the column as raw GEOMETRY so DuckDB spatial functions work.

Buffer distance convention
--------------------------
All buffer templates use {buffer_km} or {buffer_m} (never degrees).
SQL converts to degrees: metres / 111_320.

Mixed-source candidates
-----------------------
generate_samples.py pads every candidate list with 50 % cross-source
distractors so the model always sees both source values and learns the
correct parquet path from the candidates table.

Template families
-----------------
direct_lookup      Simple single-feature fetch by ID.
adjacency          ST_Touches — features sharing a border.
multi_adjacency    Features that simultaneously touch TWO anchors.
containment        ST_Within / ST_Contains — hierarchical nesting.
intersection       ST_Intersects — overlapping or crossing features.
buffer             ST_Buffer — proximity zones in km or metres.
chained            Containment + EXISTS/NOT EXISTS sea predicate.
difference         ST_Difference — geometry subtraction.
border_corridor    Buffered ST_Intersection of a shared border.
set_operations     ST_Union_Agg — merging multiple geometries.
partial_selection  Bbox clipping — directional halves or feature clips.
aggregation        TOP-N by area with ORDER BY.
window_function    ROW_NUMBER() OVER (PARTITION BY) — per-group ranking.
attribute_filter   Pure attribute predicates: is_land, country, etc.
"""

from dataclasses import dataclass
from typing import List, Literal


@dataclass
class SQLTemplate:
    """SQL template for synthetic data generation."""

    template_id: str
    family: str
    sql_difficulty: Literal["easy", "medium", "medium-hard", "hard"]
    anchor_source: Literal["divisions_area", "natural_earth", "mixed"]
    num_anchors: int
    sql_template: str
    question_hints: List[str]
    target_subtype: str = None
    requires_buffer: bool = False
    requires_aggregation: bool = False


# ---------------------------------------------------------------------------
# Template catalog
# ---------------------------------------------------------------------------

TEMPLATES = [

    # ── DIRECT LOOKUP ────────────────────────────────────────────────────────

    SQLTemplate(
        template_id="lookup_01",
        family="direct_lookup",
        sql_difficulty="easy",
        anchor_source="divisions_area",
        num_anchors=1,
        sql_template=(
            "SELECT ST_AsGeoJSON(geometry) AS geometry,"
            " names.\"primary\" AS name, id, subtype, country"
            " FROM read_parquet('divisions_area')"
            " WHERE id = '{anchor_id}'"
        ),
        question_hints=[
            "Show me {anchor_name}",
            "Get the boundary of {anchor_name}",
            "Find {anchor_name}",
            "Show the geometry of {anchor_name}",
            "Where is {anchor_name}?",
            "Give me the outline of {anchor_name}",
            "Fetch {anchor_name}",
            "Display {anchor_name} on a map",
            "What does {anchor_name} look like?",
        ],
    ),

    SQLTemplate(
        template_id="lookup_02",
        family="direct_lookup",
        sql_difficulty="easy",
        anchor_source="natural_earth",
        num_anchors=1,
        sql_template=(
            "SELECT ST_AsGeoJSON(geometry) AS geometry,"
            " names.\"primary\" AS name, id, subtype"
            " FROM read_parquet('natural_earth')"
            " WHERE id = '{anchor_id}'"
        ),
        question_hints=[
            "Show me the {anchor_name}",
            "Get {anchor_name}",
            "Find the {anchor_name}",
            "Where is the {anchor_name}?",
            "Show the extent of the {anchor_name}",
            "Give me the geometry of the {anchor_name}",
            "Fetch the {anchor_name}",
            "Display the {anchor_name}",
        ],
    ),

    # ── ADJACENCY ────────────────────────────────────────────────────────────

    SQLTemplate(
        template_id="adj_01",
        family="adjacency",
        sql_difficulty="medium",
        anchor_source="divisions_area",
        num_anchors=1,
        sql_template=(
            "WITH a AS ("
            "  SELECT geometry FROM read_parquet('divisions_area') WHERE id = '{anchor_id}'"
            ")"
            " SELECT b.id, b.names.\"primary\" AS name, b.subtype, b.country,"
            "        ST_AsGeoJSON(b.geometry) AS geometry"
            " FROM read_parquet('divisions_area') AS b, a"
            " WHERE b.id != '{anchor_id}'"
            "   AND ST_Touches(a.geometry, b.geometry)"
        ),
        question_hints=[
            "Which regions border {anchor_name}?",
            "What administrative units touch {anchor_name}?",
            "List all places adjacent to {anchor_name}",
            "What shares a border with {anchor_name}?",
            "Neighbours of {anchor_name}",
            "What is adjacent to {anchor_name}?",
            "All places that share a boundary with {anchor_name}",
        ],
    ),

    SQLTemplate(
        template_id="adj_02",
        family="adjacency",
        sql_difficulty="medium",
        anchor_source="divisions_area",
        num_anchors=1,
        target_subtype="region",
        sql_template=(
            "WITH a AS ("
            "  SELECT geometry FROM read_parquet('divisions_area') WHERE id = '{anchor_id}'"
            ")"
            " SELECT b.id, b.names.\"primary\" AS name, b.subtype, b.country,"
            "        ST_AsGeoJSON(b.geometry) AS geometry"
            " FROM read_parquet('divisions_area') AS b, a"
            " WHERE b.id != '{anchor_id}'"
            "   AND b.subtype = '{target_subtype}'"
            "   AND ST_Touches(a.geometry, b.geometry)"
        ),
        question_hints=[
            "Which {target_subtype}s border {anchor_name}?",
            "What {target_subtype}s share a border with {anchor_name}?",
            "{target_subtype}s that touch {anchor_name}",
            "Neighbouring {target_subtype}s of {anchor_name}",
            "Which {target_subtype}s are adjacent to {anchor_name}?",
            "States bordering {anchor_name}",
            "Countries that share a boundary with {anchor_name}",
        ],
    ),

    SQLTemplate(
        template_id="adj_03",
        family="adjacency",
        sql_difficulty="medium",
        anchor_source="divisions_area",
        num_anchors=1,
        target_subtype="sea",
        sql_template=(
            "WITH a AS ("
            "  SELECT geometry FROM read_parquet('divisions_area') WHERE id = '{anchor_id}'"
            ")"
            " SELECT n.id, n.names.\"primary\" AS name, n.subtype,"
            "        ST_AsGeoJSON(n.geometry) AS geometry"
            " FROM read_parquet('natural_earth') AS n, a"
            " WHERE n.subtype IN ('ocean', 'sea')"
            "   AND ST_Touches(a.geometry, n.geometry)"
        ),
        question_hints=[
            "Which seas touch {anchor_name}?",
            "What seas border {anchor_name}?",
            "Which bodies of water is {anchor_name} adjacent to?",
            "What ocean or sea borders {anchor_name}?",
            "Which oceans touch {anchor_name}?",
            "What coastline does {anchor_name} have?",
            "Which water bodies does {anchor_name} border?",
        ],
    ),

    # ── MULTI-ADJACENCY ──────────────────────────────────────────────────────

    SQLTemplate(
        template_id="multi_adj_01",
        family="multi_adjacency",
        sql_difficulty="hard",
        anchor_source="divisions_area",
        num_anchors=2,
        sql_template=(
            "WITH a AS ("
            "  SELECT geometry FROM read_parquet('divisions_area') WHERE id = '{anchor_id_1}'"
            "),"
            " b AS ("
            "  SELECT geometry FROM read_parquet('divisions_area') WHERE id = '{anchor_id_2}'"
            ")"
            " SELECT c.id, c.names.\"primary\" AS name, c.subtype, c.country,"
            "        ST_AsGeoJSON(c.geometry) AS geometry"
            " FROM read_parquet('divisions_area') AS c, a, b"
            " WHERE c.id NOT IN ('{anchor_id_1}', '{anchor_id_2}')"
            "   AND ST_Touches(c.geometry, a.geometry)"
            "   AND ST_Touches(c.geometry, b.geometry)"
        ),
        question_hints=[
            "Which regions border both {anchor_1_name} and {anchor_2_name}?",
            "What places touch both {anchor_1_name} and {anchor_2_name}?",
            "Regions adjacent to both {anchor_1_name} and {anchor_2_name}",
            "Which states share a border with both {anchor_1_name} and {anchor_2_name}?",
            "Countries that are neighbours of both {anchor_1_name} and {anchor_2_name}",
        ],
    ),

    # ── CONTAINMENT ──────────────────────────────────────────────────────────

    SQLTemplate(
        template_id="contain_01",
        family="containment",
        sql_difficulty="medium",
        anchor_source="divisions_area",
        num_anchors=1,
        target_subtype="locality",
        sql_template=(
            "WITH a AS ("
            "  SELECT geometry FROM read_parquet('divisions_area') WHERE id = '{anchor_id}'"
            ")"
            " SELECT b.id, b.names.\"primary\" AS name, b.subtype, b.country,"
            "        ST_AsGeoJSON(b.geometry) AS geometry"
            " FROM read_parquet('divisions_area') AS b, a"
            " WHERE b.id != '{anchor_id}'"
            "   AND b.subtype = '{target_subtype}'"
            "   AND ST_Within(b.geometry, a.geometry)"
        ),
        question_hints=[
            "What {target_subtype}s are in {anchor_name}?",
            "Which {target_subtype}s fall within {anchor_name}?",
            "List all {target_subtype}s inside {anchor_name}",
            "{target_subtype}s contained by {anchor_name}",
            "All {target_subtype}s within the boundaries of {anchor_name}",
            "Cities in {anchor_name}",
            "Towns inside {anchor_name}",
        ],
    ),

    SQLTemplate(
        template_id="contain_02",
        family="containment",
        sql_difficulty="medium",
        anchor_source="divisions_area",
        num_anchors=1,
        target_subtype="country",
        sql_template=(
            "WITH a AS ("
            "  SELECT geometry FROM read_parquet('divisions_area') WHERE id = '{anchor_id}'"
            ")"
            " SELECT b.id, b.names.\"primary\" AS name, b.subtype,"
            "        ST_AsGeoJSON(b.geometry) AS geometry"
            " FROM read_parquet('divisions_area') AS b, a"
            " WHERE b.id != '{anchor_id}'"
            "   AND b.subtype = '{target_subtype}'"
            "   AND ST_Contains(b.geometry, a.geometry)"
        ),
        question_hints=[
            "What country contains {anchor_name}?",
            "Which country is {anchor_name} in?",
            "What country does {anchor_name} belong to?",
            "Which nation contains {anchor_name}?",
        ],
    ),

    SQLTemplate(
        template_id="contain_03",
        family="containment",
        sql_difficulty="medium",
        anchor_source="natural_earth",
        num_anchors=1,
        target_subtype="region",
        sql_template=(
            "WITH a AS ("
            "  SELECT geometry FROM read_parquet('natural_earth') WHERE id = '{anchor_id}'"
            ")"
            " SELECT b.id, b.names.\"primary\" AS name, b.subtype, b.country,"
            "        ST_AsGeoJSON(b.geometry) AS geometry"
            " FROM read_parquet('divisions_area') AS b, a"
            " WHERE b.subtype = '{target_subtype}'"
            "   AND ST_Within(b.geometry, a.geometry)"
        ),
        question_hints=[
            "Which {target_subtype}s are in the {anchor_name}?",
            "What {target_subtype}s fall within the {anchor_name}?",
            "{target_subtype}s inside the {anchor_name}",
            "Administrative {target_subtype}s within the {anchor_name}",
            "All regions contained by the {anchor_name}",
        ],
    ),

    # ── INTERSECTION ─────────────────────────────────────────────────────────

    SQLTemplate(
        template_id="intersect_01",
        family="intersection",
        sql_difficulty="medium-hard",
        anchor_source="divisions_area",
        num_anchors=1,
        target_subtype="region",
        sql_template=(
            "WITH a AS ("
            "  SELECT geometry FROM read_parquet('divisions_area') WHERE id = '{anchor_id}'"
            ")"
            " SELECT b.id, b.names.\"primary\" AS name, b.subtype, b.country,"
            "        ST_AsGeoJSON(b.geometry) AS geometry"
            " FROM read_parquet('divisions_area') AS b, a"
            " WHERE b.id != '{anchor_id}'"
            "   AND b.subtype = '{target_subtype}'"
            "   AND ST_Intersects(b.geometry, a.geometry)"
        ),
        question_hints=[
            "Which {target_subtype}s intersect {anchor_name}?",
            "What {target_subtype}s overlap with {anchor_name}?",
            "{target_subtype}s that cross into {anchor_name}",
            "Which {target_subtype}s overlap {anchor_name}?",
        ],
    ),

    SQLTemplate(
        template_id="intersect_02",
        family="intersection",
        sql_difficulty="medium-hard",
        anchor_source="natural_earth",
        num_anchors=1,
        target_subtype="country",
        sql_template=(
            "WITH a AS ("
            "  SELECT geometry FROM read_parquet('natural_earth') WHERE id = '{anchor_id}'"
            ")"
            " SELECT b.id, b.names.\"primary\" AS name, b.subtype,"
            "        ST_AsGeoJSON(b.geometry) AS geometry"
            " FROM read_parquet('divisions_area') AS b, a"
            " WHERE b.subtype = '{target_subtype}'"
            "   AND ST_Intersects(b.geometry, a.geometry)"
        ),
        question_hints=[
            "Which countries intersect the {anchor_name}?",
            "What countries does the {anchor_name} pass through?",
            "Countries that overlap with the {anchor_name}",
            "Which countries touch the {anchor_name}?",
            "Nations intersected by the {anchor_name}",
            "Countries the {anchor_name} flows through",
            "Which nations does the {anchor_name} cross?",
            "Countries along the {anchor_name}",
            "States the {anchor_name} runs through",
        ],
    ),

    # ── BUFFER ───────────────────────────────────────────────────────────────
    # CTE computes the buffered geometry (raw) for the spatial join.
    # Final SELECT wraps the result features with ST_AsGeoJSON.

    SQLTemplate(
        template_id="buffer_01",
        family="buffer",
        sql_difficulty="hard",
        anchor_source="divisions_area",
        num_anchors=1,
        requires_buffer=True,
        sql_template=(
            "WITH a AS ("
            "  SELECT ST_Buffer(geometry, {buffer_km} * 1000.0 / 111320.0) AS geom"
            "  FROM read_parquet('divisions_area')"
            "  WHERE id = '{anchor_id}'"
            ")"
            " SELECT b.id, b.names.\"primary\" AS name, b.subtype, b.country,"
            "        ST_AsGeoJSON(b.geometry) AS geometry"
            " FROM read_parquet('divisions_area') AS b, a"
            " WHERE b.id != '{anchor_id}'"
            "   AND ST_Intersects(b.geometry, a.geom)"
        ),
        question_hints=[
            "What is within {buffer_km} km of {anchor_name}?",
            "Administrative units within {buffer_km} km of {anchor_name}",
            "Features within a {buffer_km} km radius of {anchor_name}",
            "Places within {buffer_km} kilometers of {anchor_name}",
            "{buffer_km} km buffer around {anchor_name}",
            "What falls within {buffer_km} km of {anchor_name}?",
            "Everything within {buffer_km} km of {anchor_name}",
        ],
    ),

    SQLTemplate(
        template_id="buffer_02",
        family="buffer",
        sql_difficulty="hard",
        anchor_source="divisions_area",
        num_anchors=1,
        requires_buffer=True,
        sql_template=(
            "WITH a AS ("
            "  SELECT ST_Buffer(geometry, {buffer_m} / 111320.0) AS geom"
            "  FROM read_parquet('divisions_area')"
            "  WHERE id = '{anchor_id}'"
            ")"
            " SELECT b.id, b.names.\"primary\" AS name, b.subtype, b.country,"
            "        ST_AsGeoJSON(b.geometry) AS geometry"
            " FROM read_parquet('divisions_area') AS b, a"
            " WHERE b.id != '{anchor_id}'"
            "   AND ST_Intersects(b.geometry, a.geom)"
        ),
        question_hints=[
            "What is within {buffer_m} meters of {anchor_name}?",
            "Features within {buffer_m} m of {anchor_name}",
            "Places within {buffer_m} metres of {anchor_name}",
            "{buffer_m} meter buffer around {anchor_name}",
            "What falls within {buffer_m} m of {anchor_name}?",
            "Administrative units within {buffer_m} metres of {anchor_name}",
        ],
    ),

    SQLTemplate(
        template_id="buffer_03",
        family="buffer",
        sql_difficulty="hard",
        anchor_source="natural_earth",
        num_anchors=1,
        requires_buffer=True,
        sql_template=(
            "WITH a AS ("
            "  SELECT ST_Buffer(geometry, {buffer_km} * 1000.0 / 111320.0) AS geom"
            "  FROM read_parquet('natural_earth')"
            "  WHERE id = '{anchor_id}'"
            ")"
            " SELECT b.id, b.names.\"primary\" AS name, b.subtype, b.country,"
            "        ST_AsGeoJSON(b.geometry) AS geometry"
            " FROM read_parquet('divisions_area') AS b, a"
            " WHERE ST_Intersects(b.geometry, a.geom)"
        ),
        question_hints=[
            "What administrative units are within {buffer_km} km of the {anchor_name}?",
            "Countries within {buffer_km} km of the {anchor_name}",
            "Regions within {buffer_km} km of the {anchor_name}",
            "What falls within {buffer_km} km of the {anchor_name}?",
            "Administrative divisions within a {buffer_km} km radius of the {anchor_name}",
            "Places within {buffer_km} kilometers of the {anchor_name}",
        ],
    ),

    SQLTemplate(
        template_id="buffer_04",
        family="buffer",
        sql_difficulty="hard",
        anchor_source="natural_earth",
        num_anchors=1,
        requires_buffer=True,
        sql_template=(
            "WITH a AS ("
            "  SELECT ST_Buffer(geometry, {buffer_m} / 111320.0) AS geom"
            "  FROM read_parquet('natural_earth')"
            "  WHERE id = '{anchor_id}'"
            ")"
            " SELECT b.id, b.names.\"primary\" AS name, b.subtype, b.country,"
            "        ST_AsGeoJSON(b.geometry) AS geometry"
            " FROM read_parquet('divisions_area') AS b, a"
            " WHERE ST_Intersects(b.geometry, a.geom)"
        ),
        question_hints=[
            "What is within {buffer_m} meters of the {anchor_name}?",
            "Administrative units within {buffer_m} m of the {anchor_name}",
            "Places within {buffer_m} metres of the {anchor_name}",
            "{buffer_m} meter buffer around the {anchor_name}",
        ],
    ),

    # ── CHAINED ──────────────────────────────────────────────────────────────
    # Containment + EXISTS/NOT EXISTS ocean/sea.
    # CTE holds raw geometry for ST_Within; final SELECT wraps with ST_AsGeoJSON.

    SQLTemplate(
        template_id="chained_01",
        family="chained",
        sql_difficulty="hard",
        anchor_source="divisions_area",
        num_anchors=1,
        target_subtype="locality",
        sql_template=(
            "WITH region AS ("
            "  SELECT geometry FROM read_parquet('divisions_area') WHERE id = '{anchor_id}'"
            ")"
            " SELECT b.id, b.names.\"primary\" AS name, b.subtype, b.country,"
            "        ST_AsGeoJSON(b.geometry) AS geometry"
            " FROM read_parquet('divisions_area') AS b, region"
            " WHERE b.subtype = '{target_subtype}'"
            "   AND ST_Within(b.geometry, region.geometry)"
            "   AND EXISTS ("
            "     SELECT 1 FROM read_parquet('natural_earth') AS n"
            "     WHERE n.subtype IN ('ocean', 'sea')"
            "       AND ST_Intersects(b.geometry, n.geometry)"
            "   )"
        ),
        question_hints=[
            "Coastal {target_subtype}s of {anchor_name}",
            "{target_subtype}s in {anchor_name} with sea access",
            "Which {target_subtype}s in {anchor_name} are on the coast?",
            "Seaside {target_subtype}s within {anchor_name}",
            "Coastal towns of {anchor_name}",
            "Which towns in {anchor_name} touch the ocean?",
            "{target_subtype}s in {anchor_name} bordering the sea",
        ],
    ),

    SQLTemplate(
        template_id="chained_02",
        family="chained",
        sql_difficulty="hard",
        anchor_source="divisions_area",
        num_anchors=1,
        target_subtype="country",
        sql_template=(
            "WITH region AS ("
            "  SELECT geometry FROM read_parquet('divisions_area') WHERE id = '{anchor_id}'"
            ")"
            " SELECT b.id, b.names.\"primary\" AS name, b.subtype, b.country,"
            "        ST_AsGeoJSON(b.geometry) AS geometry"
            " FROM read_parquet('divisions_area') AS b, region"
            " WHERE b.subtype = '{target_subtype}'"
            "   AND ST_Intersects(b.geometry, region.geometry)"
            "   AND NOT EXISTS ("
            "     SELECT 1 FROM read_parquet('natural_earth') AS n"
            "     WHERE n.subtype IN ('ocean', 'sea')"
            "       AND ST_Intersects(b.geometry, n.geometry)"
            "   )"
        ),
        question_hints=[
            "Landlocked {target_subtype}s in {anchor_name}",
            "Which {target_subtype}s in {anchor_name} have no sea access?",
            "{target_subtype}s in {anchor_name} that are landlocked",
            "Countries in {anchor_name} with no coastline",
            "Which countries near {anchor_name} are landlocked?",
            "Landlocked nations in the region of {anchor_name}",
        ],
    ),

    SQLTemplate(
        template_id="chained_03",
        family="chained",
        sql_difficulty="hard",
        anchor_source="divisions_area",
        num_anchors=1,
        target_subtype="locality",
        sql_template=(
            "WITH region AS ("
            "  SELECT geometry FROM read_parquet('divisions_area') WHERE id = '{anchor_id}'"
            ")"
            " SELECT b.id, b.names.\"primary\" AS name, b.subtype, b.country,"
            "        ST_AsGeoJSON(b.geometry) AS geometry"
            " FROM read_parquet('divisions_area') AS b, region"
            " WHERE b.subtype = '{target_subtype}'"
            "   AND ST_Within(b.geometry, region.geometry)"
            "   AND EXISTS ("
            "     SELECT 1 FROM read_parquet('natural_earth') AS n"
            "     WHERE n.subtype IN ('Terrain area', 'Island group', 'Peninsula')"
            "       AND ST_Intersects(b.geometry, n.geometry)"
            "   )"
        ),
        question_hints=[
            "{target_subtype}s in {anchor_name} in a terrain or island area",
            "Hill or mountain {target_subtype}s within {anchor_name}",
            "{target_subtype}s of {anchor_name} on terrain features",
            "Island or highland {target_subtype}s of {anchor_name}",
        ],
    ),

    # ── DIFFERENCE ───────────────────────────────────────────────────────────
    # CTEs hold raw geometry; ST_Difference result wrapped with ST_AsGeoJSON.

    SQLTemplate(
        template_id="diff_01",
        family="difference",
        sql_difficulty="hard",
        anchor_source="divisions_area",
        num_anchors=2,
        sql_template=(
            "WITH a AS ("
            "  SELECT geometry FROM read_parquet('divisions_area') WHERE id = '{anchor_id_1}'"
            "),"
            " b AS ("
            "  SELECT geometry FROM read_parquet('divisions_area') WHERE id = '{anchor_id_2}'"
            ")"
            " SELECT ST_AsGeoJSON(ST_Difference(a.geometry, b.geometry)) AS geometry"
            " FROM a, b"
            " WHERE ST_Intersects(a.geometry, b.geometry)"
        ),
        question_hints=[
            "{anchor_1_name} excluding {anchor_2_name}",
            "{anchor_1_name} minus {anchor_2_name}",
            "The part of {anchor_1_name} that is not in {anchor_2_name}",
            "{anchor_1_name} without the {anchor_2_name} area",
            "Remove {anchor_2_name} from {anchor_1_name}",
            "{anchor_1_name} with {anchor_2_name} cut out",
        ],
    ),

    SQLTemplate(
        template_id="diff_02",
        family="difference",
        sql_difficulty="hard",
        anchor_source="mixed",
        num_anchors=2,
        sql_template=(
            "WITH a AS ("
            "  SELECT geometry FROM read_parquet('divisions_area') WHERE id = '{anchor_id}'"
            "),"
            " b AS ("
            "  SELECT geometry FROM read_parquet('natural_earth') WHERE id = '{clip_feature_id}'"
            ")"
            " SELECT ST_AsGeoJSON(ST_Difference(a.geometry, b.geometry)) AS geometry"
            " FROM a, b"
            " WHERE ST_Intersects(a.geometry, b.geometry)"
        ),
        question_hints=[
            "The part of {anchor_name} outside the {clip_feature_name}",
            "{anchor_name} excluding the {clip_feature_name}",
            "{anchor_name} minus the {clip_feature_name}",
            "The land area of {anchor_name} not covered by the {clip_feature_name}",
        ],
    ),

    # ── BORDER CORRIDOR ──────────────────────────────────────────────────────
    # Intermediate intersection kept raw; final buffer wrapped with ST_AsGeoJSON.

    SQLTemplate(
        template_id="corridor_01",
        family="border_corridor",
        sql_difficulty="hard",
        anchor_source="divisions_area",
        num_anchors=2,
        requires_buffer=True,
        sql_template=(
            "WITH a AS ("
            "  SELECT geometry FROM read_parquet('divisions_area') WHERE id = '{anchor_id_1}'"
            "),"
            " b AS ("
            "  SELECT geometry FROM read_parquet('divisions_area') WHERE id = '{anchor_id_2}'"
            "),"
            " border AS ("
            "  SELECT ST_Intersection(a.geometry, b.geometry) AS line"
            "  FROM a, b"
            "  WHERE ST_Intersects(a.geometry, b.geometry)"
            ")"
            " SELECT ST_AsGeoJSON(ST_Buffer(border.line, {buffer_km} * 1000.0 / 111320.0)) AS geometry"
            " FROM border"
            " WHERE border.line IS NOT NULL"
        ),
        question_hints=[
            "{buffer_km} km zone along the border between {anchor_1_name} and {anchor_2_name}",
            "The {buffer_km} km border corridor between {anchor_1_name} and {anchor_2_name}",
            "Area within {buffer_km} km of the {anchor_1_name}-{anchor_2_name} border",
            "The region straddling the border of {anchor_1_name} and {anchor_2_name} within {buffer_km} km",
            "{buffer_km} km on either side of the {anchor_1_name} and {anchor_2_name} border",
        ],
    ),

    # ── SET OPERATIONS ───────────────────────────────────────────────────────
    # union_01 / union_02: 2-anchor and filtered-containment unions.
    # union_03: 3-anchor union — trains the model on IN-clause with 3 IDs.
    # contain_multi: subtype within multiple countries via country IN clause.

    SQLTemplate(
        template_id="union_01",
        family="set_operations",
        sql_difficulty="medium-hard",
        anchor_source="divisions_area",
        num_anchors=2,
        sql_template=(
            "SELECT ST_AsGeoJSON(ST_Union_Agg(geometry)) AS geometry,"
            " array_agg(names.\"primary\") AS names"
            " FROM read_parquet('divisions_area')"
            " WHERE id IN ('{anchor_id_1}', '{anchor_id_2}')"
        ),
        question_hints=[
            "The combined area of {anchor_1_name} and {anchor_2_name}",
            "Union of {anchor_1_name} and {anchor_2_name}",
            "Merge {anchor_1_name} and {anchor_2_name}",
            "{anchor_1_name} and {anchor_2_name} together",
            "Combined geometry of {anchor_1_name} and {anchor_2_name}",
        ],
    ),

    SQLTemplate(
        template_id="union_03",
        family="set_operations",
        sql_difficulty="medium-hard",
        anchor_source="divisions_area",
        num_anchors=3,
        sql_template=(
            "SELECT ST_AsGeoJSON(ST_Union_Agg(geometry)) AS geometry,"
            " array_agg(names.\"primary\") AS names"
            " FROM read_parquet('divisions_area')"
            " WHERE id IN ('{anchor_id_1}', '{anchor_id_2}', '{anchor_id_3}')"
        ),
        question_hints=[
            "Show me {anchor_1_name}, {anchor_2_name} and {anchor_3_name}",
            "The combined area of {anchor_1_name}, {anchor_2_name} and {anchor_3_name}",
            "Union of {anchor_1_name}, {anchor_2_name} and {anchor_3_name}",
            "Merge {anchor_1_name}, {anchor_2_name} and {anchor_3_name}",
            "{anchor_1_name}, {anchor_2_name} and {anchor_3_name} together",
            "Display {anchor_1_name}, {anchor_2_name} and {anchor_3_name}",
        ],
    ),

    SQLTemplate(
        template_id="contain_multi_01",
        family="set_operations",
        sql_difficulty="medium-hard",
        anchor_source="divisions_area",
        num_anchors=2,
        target_subtype="region",
        sql_template=(
            "SELECT id, names.\"primary\" AS name, subtype, country,"
            "       ST_AsGeoJSON(geometry) AS geometry"
            " FROM read_parquet('divisions_area')"
            " WHERE country IN ('{country_1}', '{country_2}')"
            "   AND subtype = '{target_subtype}'"
        ),
        question_hints=[
            "{target_subtype}s of {anchor_1_name} and {anchor_2_name}",
            "All {target_subtype}s in {anchor_1_name} and {anchor_2_name}",
            "Show {target_subtype}s across {anchor_1_name} and {anchor_2_name}",
            "{target_subtype}s belonging to {anchor_1_name} and {anchor_2_name}",
            "List {target_subtype}s in both {anchor_1_name} and {anchor_2_name}",
        ],
    ),

    SQLTemplate(
        template_id="contain_multi_02",
        family="set_operations",
        sql_difficulty="medium-hard",
        anchor_source="divisions_area",
        num_anchors=3,
        target_subtype="region",
        sql_template=(
            "SELECT id, names.\"primary\" AS name, subtype, country,"
            "       ST_AsGeoJSON(geometry) AS geometry"
            " FROM read_parquet('divisions_area')"
            " WHERE country IN ('{country_1}', '{country_2}', '{country_3}')"
            "   AND subtype = '{target_subtype}'"
        ),
        question_hints=[
            "{target_subtype}s of {anchor_1_name}, {anchor_2_name} and {anchor_3_name}",
            "All {target_subtype}s in {anchor_1_name}, {anchor_2_name} and {anchor_3_name}",
            "Show {target_subtype}s across {anchor_1_name}, {anchor_2_name} and {anchor_3_name}",
            "List {target_subtype}s in {anchor_1_name}, {anchor_2_name} and {anchor_3_name}",
        ],
    ),

    SQLTemplate(
        template_id="union_02",
        family="set_operations",
        sql_difficulty="hard",
        anchor_source="divisions_area",
        num_anchors=1,
        target_subtype="locality",
        requires_aggregation=True,
        sql_template=(
            "WITH a AS ("
            "  SELECT geometry FROM read_parquet('divisions_area') WHERE id = '{anchor_id}'"
            ")"
            " SELECT ST_AsGeoJSON(ST_Union_Agg(b.geometry)) AS geometry"
            " FROM read_parquet('divisions_area') AS b, a"
            " WHERE b.subtype = '{target_subtype}'"
            "   AND ST_Within(b.geometry, a.geometry)"
        ),
        question_hints=[
            "Merge all {target_subtype}s in {anchor_name} into one geometry",
            "Combined geometry of all {target_subtype}s in {anchor_name}",
            "Union of all {target_subtype}s within {anchor_name}",
            "All {target_subtype}s of {anchor_name} merged together",
            "The overall extent of {target_subtype}s in {anchor_name}",
        ],
    ),

    # ── PARTIAL SELECTION ────────────────────────────────────────────────────
    # Bbox clip CTEs use raw geometry; ST_Intersection result wrapped.

    SQLTemplate(
        template_id="partial_01",
        family="partial_selection",
        sql_difficulty="hard",
        anchor_source="divisions_area",
        num_anchors=1,
        sql_template=(
            "WITH a AS ("
            "  SELECT geometry FROM read_parquet('divisions_area') WHERE id = '{anchor_id}'"
            "),"
            " bbox AS ("
            "  SELECT ST_XMin(geometry) AS xmin, ST_XMax(geometry) AS xmax,"
            "         ST_YMin(geometry) AS ymin, ST_YMax(geometry) AS ymax FROM a"
            "),"
            " clip AS ("
            "  SELECT ST_MakeEnvelope(xmin, (ymin + ymax) / 2.0, xmax, ymax) AS half_geom FROM bbox"
            ")"
            " SELECT ST_AsGeoJSON(ST_Intersection(a.geometry, clip.half_geom)) AS geometry"
            " FROM a, clip"
        ),
        question_hints=[
            "The northern half of {anchor_name}",
            "Northern part of {anchor_name}",
            "The top half of {anchor_name}",
            "Northern portion of {anchor_name}",
        ],
    ),

    SQLTemplate(
        template_id="partial_02",
        family="partial_selection",
        sql_difficulty="hard",
        anchor_source="divisions_area",
        num_anchors=1,
        sql_template=(
            "WITH a AS ("
            "  SELECT geometry FROM read_parquet('divisions_area') WHERE id = '{anchor_id}'"
            "),"
            " bbox AS ("
            "  SELECT ST_XMin(geometry) AS xmin, ST_XMax(geometry) AS xmax,"
            "         ST_YMin(geometry) AS ymin, ST_YMax(geometry) AS ymax FROM a"
            "),"
            " clip AS ("
            "  SELECT ST_MakeEnvelope(xmin, ymin, xmax, (ymin + ymax) / 2.0) AS half_geom FROM bbox"
            ")"
            " SELECT ST_AsGeoJSON(ST_Intersection(a.geometry, clip.half_geom)) AS geometry"
            " FROM a, clip"
        ),
        question_hints=[
            "The southern half of {anchor_name}",
            "Southern part of {anchor_name}",
            "The bottom half of {anchor_name}",
            "Southern portion of {anchor_name}",
        ],
    ),

    SQLTemplate(
        template_id="partial_03",
        family="partial_selection",
        sql_difficulty="hard",
        anchor_source="divisions_area",
        num_anchors=1,
        sql_template=(
            "WITH a AS ("
            "  SELECT geometry FROM read_parquet('divisions_area') WHERE id = '{anchor_id}'"
            "),"
            " bbox AS ("
            "  SELECT ST_XMin(geometry) AS xmin, ST_XMax(geometry) AS xmax,"
            "         ST_YMin(geometry) AS ymin, ST_YMax(geometry) AS ymax FROM a"
            "),"
            " clip AS ("
            "  SELECT ST_MakeEnvelope((xmin + xmax) / 2.0, ymin, xmax, ymax) AS half_geom FROM bbox"
            ")"
            " SELECT ST_AsGeoJSON(ST_Intersection(a.geometry, clip.half_geom)) AS geometry"
            " FROM a, clip"
        ),
        question_hints=[
            "The eastern half of {anchor_name}",
            "Eastern part of {anchor_name}",
            "The right half of {anchor_name}",
            "Eastern portion of {anchor_name}",
        ],
    ),

    SQLTemplate(
        template_id="partial_04",
        family="partial_selection",
        sql_difficulty="hard",
        anchor_source="divisions_area",
        num_anchors=1,
        sql_template=(
            "WITH a AS ("
            "  SELECT geometry FROM read_parquet('divisions_area') WHERE id = '{anchor_id}'"
            "),"
            " bbox AS ("
            "  SELECT ST_XMin(geometry) AS xmin, ST_XMax(geometry) AS xmax,"
            "         ST_YMin(geometry) AS ymin, ST_YMax(geometry) AS ymax FROM a"
            "),"
            " clip AS ("
            "  SELECT ST_MakeEnvelope(xmin, ymin, (xmin + xmax) / 2.0, ymax) AS half_geom FROM bbox"
            ")"
            " SELECT ST_AsGeoJSON(ST_Intersection(a.geometry, clip.half_geom)) AS geometry"
            " FROM a, clip"
        ),
        question_hints=[
            "The western half of {anchor_name}",
            "Western part of {anchor_name}",
            "The left half of {anchor_name}",
            "Western portion of {anchor_name}",
        ],
    ),

    SQLTemplate(
        template_id="partial_05",
        family="partial_selection",
        sql_difficulty="hard",
        anchor_source="mixed",
        num_anchors=2,
        sql_template=(
            "WITH a AS ("
            "  SELECT geometry AS g1 FROM read_parquet('divisions_area') WHERE id = '{anchor_id}'"
            "),"
            " b AS ("
            "  SELECT geometry AS g2 FROM read_parquet('natural_earth') WHERE id = '{clip_feature_id}'"
            ")"
            " SELECT ST_AsGeoJSON(ST_Intersection(a.g1, b.g2)) AS geometry"
            " FROM a, b"
            " WHERE ST_Intersects(a.g1, b.g2)"
        ),
        question_hints=[
            "The part of {anchor_name} that overlaps the {clip_feature_name}",
            "{anchor_name} within the {clip_feature_name}",
            "The portion of {anchor_name} inside the {clip_feature_name}",
            "Clip {anchor_name} to the {clip_feature_name}",
        ],
    ),

    # ── AGGREGATION ──────────────────────────────────────────────────────────
    # ST_Area uses raw geometry in the ORDER BY; final SELECT wraps output.

    SQLTemplate(
        template_id="agg_01",
        family="aggregation",
        sql_difficulty="hard",
        anchor_source="divisions_area",
        num_anchors=1,
        target_subtype=None,  # filled at generation time: locality or region
        requires_aggregation=True,
        sql_template=(
            "WITH a AS ("
            "  SELECT geometry FROM read_parquet('divisions_area') WHERE id = '{anchor_id}'"
            ")"
            " SELECT b.id, b.names.\"primary\" AS name,"
            "        ST_AsGeoJSON(b.geometry) AS geometry,"
            "        ST_Area(b.geometry) AS area"
            " FROM read_parquet('divisions_area') AS b, a"
            " WHERE ST_Within(b.geometry, a.geometry)"
            "   AND b.subtype = '{target_subtype}'"
            " ORDER BY area DESC"
            " LIMIT {top_n}"
        ),
        question_hints=[
            "Top {top_n} largest {target_subtype}s in {anchor_name}",
            "Biggest {top_n} {target_subtype}s in {anchor_name}",
            "{top_n} largest {target_subtype}s inside {anchor_name}",
            "The {top_n} biggest {target_subtype}s within {anchor_name}",
            "Largest {target_subtype} in {anchor_name}",
            "Which {target_subtype} in {anchor_name} has the most area?",
        ],
    ),

    SQLTemplate(
        template_id="agg_02",
        family="aggregation",
        sql_difficulty="hard",
        anchor_source="divisions_area",
        num_anchors=1,
        target_subtype=None,  # filled at generation time: locality or region
        requires_aggregation=True,
        sql_template=(
            "WITH a AS ("
            "  SELECT geometry FROM read_parquet('divisions_area') WHERE id = '{anchor_id}'"
            ")"
            " SELECT b.id, b.names.\"primary\" AS name,"
            "        ST_AsGeoJSON(b.geometry) AS geometry,"
            "        ST_Area(b.geometry) AS area"
            " FROM read_parquet('divisions_area') AS b, a"
            " WHERE ST_Within(b.geometry, a.geometry)"
            "   AND b.subtype = '{target_subtype}'"
            " ORDER BY area ASC"
            " LIMIT {top_n}"
        ),
        question_hints=[
            "Top {top_n} smallest {target_subtype}s in {anchor_name}",
            "Smallest {top_n} {target_subtype}s in {anchor_name}",
            "{top_n} smallest {target_subtype}s inside {anchor_name}",
            "The {top_n} tiniest {target_subtype}s within {anchor_name}",
            "Smallest {target_subtype} in {anchor_name}",
            "Which {target_subtype} in {anchor_name} has the least area?",
        ],
    ),

    SQLTemplate(
        template_id="agg_03",
        family="aggregation",
        sql_difficulty="hard",
        anchor_source="divisions_area",
        num_anchors=1,
        target_subtype=None,  # filled at generation time: locality or region
        requires_aggregation=True,
        sql_template=(
            "SELECT id, names.\"primary\" AS name,"
            "       ST_AsGeoJSON(geometry) AS geometry,"
            "       ST_Area(geometry) AS area"
            " FROM read_parquet('divisions_area')"
            " WHERE country = '{country}'"
            "   AND subtype = '{target_subtype}'"
            " ORDER BY area DESC"
            " LIMIT {top_n}"
        ),
        question_hints=[
            "Top {top_n} largest {target_subtype}s in {anchor_name}",
            "{top_n} biggest {target_subtype}s in {anchor_name}",
            "Largest {top_n} {target_subtype}s in {anchor_name}",
            "The {top_n} largest {target_subtype}s in {anchor_name}",
            "Biggest {target_subtype} in {anchor_name}",
            "Which {target_subtype} in {anchor_name} is the largest?",
        ],
    ),

    SQLTemplate(
        template_id="agg_04",
        family="aggregation",
        sql_difficulty="hard",
        anchor_source="divisions_area",
        num_anchors=1,
        target_subtype=None,  # filled at generation time: locality or region
        requires_aggregation=True,
        sql_template=(
            "SELECT id, names.\"primary\" AS name,"
            "       ST_AsGeoJSON(geometry) AS geometry,"
            "       ST_Area(geometry) AS area"
            " FROM read_parquet('divisions_area')"
            " WHERE country = '{country}'"
            "   AND subtype = '{target_subtype}'"
            " ORDER BY area ASC"
            " LIMIT {top_n}"
        ),
        question_hints=[
            "Top {top_n} smallest {target_subtype}s in {anchor_name}",
            "{top_n} smallest {target_subtype}s in {anchor_name}",
            "Smallest {top_n} {target_subtype}s in {anchor_name}",
            "The {top_n} smallest {target_subtype}s in {anchor_name}",
            "Smallest {target_subtype} in {anchor_name}",
            "Which {target_subtype} in {anchor_name} is the smallest?",
        ],
    ),

    # ── WINDOW FUNCTION ──────────────────────────────────────────────────────
    # CTE keeps raw geometry for ST_Area; final SELECT wraps with ST_AsGeoJSON.

    SQLTemplate(
        template_id="window_01",
        family="window_function",
        sql_difficulty="hard",
        anchor_source="divisions_area",
        num_anchors=1,
        target_subtype="locality",
        requires_aggregation=True,
        sql_template=(
            "WITH ranked AS ("
            "  SELECT id, names.\"primary\" AS name, subtype, country, region, geometry,"
            "         ST_Area(geometry) AS area,"
            "         ROW_NUMBER() OVER (PARTITION BY region ORDER BY ST_Area(geometry) DESC) AS rn"
            "  FROM read_parquet('divisions_area')"
            "  WHERE country = '{country}'"
            "    AND subtype = '{target_subtype}'"
            ")"
            " SELECT id, name, subtype, country, region,"
            "        ST_AsGeoJSON(geometry) AS geometry, area"
            " FROM ranked"
            " WHERE rn = 1"
        ),
        question_hints=[
            "The largest {target_subtype} in each region of {anchor_name}",
            "Biggest {target_subtype} per region in {anchor_name}",
            "Largest {target_subtype} for every region of {anchor_name}",
            "The biggest {target_subtype} in each province of {anchor_name}",
        ],
    ),

    SQLTemplate(
        template_id="window_02",
        family="window_function",
        sql_difficulty="hard",
        anchor_source="divisions_area",
        num_anchors=1,
        target_subtype="locality",
        requires_aggregation=True,
        sql_template=(
            "WITH ranked AS ("
            "  SELECT id, names.\"primary\" AS name, subtype, country, region, geometry,"
            "         ST_Area(geometry) AS area,"
            "         ROW_NUMBER() OVER (PARTITION BY region ORDER BY ST_Area(geometry) ASC) AS rn"
            "  FROM read_parquet('divisions_area')"
            "  WHERE country = '{country}'"
            "    AND subtype = '{target_subtype}'"
            ")"
            " SELECT id, name, subtype, country, region,"
            "        ST_AsGeoJSON(geometry) AS geometry, area"
            " FROM ranked"
            " WHERE rn = 1"
        ),
        question_hints=[
            "The smallest {target_subtype} in each region of {anchor_name}",
            "Smallest {target_subtype} per region in {anchor_name}",
            "Tiniest {target_subtype} for every region of {anchor_name}",
            "The smallest {target_subtype} in each province of {anchor_name}",
        ],
    ),

    # ── ATTRIBUTE FILTER ─────────────────────────────────────────────────────
    # No spatial op — pure WHERE on is_land / is_territorial / country.

    SQLTemplate(
        template_id="attr_01",
        family="attribute_filter",
        sql_difficulty="medium",
        anchor_source="divisions_area",
        num_anchors=1,
        target_subtype="dependency",
        sql_template=(
            "SELECT id, names.\"primary\" AS name, subtype, country,"
            "       ST_AsGeoJSON(geometry) AS geometry"
            " FROM read_parquet('divisions_area')"
            " WHERE country = '{country}'"
            "   AND is_land = TRUE"
            "   AND subtype = '{target_subtype}'"
        ),
        question_hints=[
            "Island territories of {anchor_name}",
            "Overseas island {target_subtype}s belonging to {anchor_name}",
            "Which islands are part of {anchor_name}?",
            "Land territories of {anchor_name}",
            "Island possessions of {anchor_name}",
            "{anchor_name}'s island {target_subtype}s",
        ],
    ),

    SQLTemplate(
        template_id="attr_02",
        family="attribute_filter",
        sql_difficulty="medium",
        anchor_source="divisions_area",
        num_anchors=1,
        target_subtype="region",
        sql_template=(
            "SELECT id, names.\"primary\" AS name, subtype, country,"
            "       ST_AsGeoJSON(geometry) AS geometry"
            " FROM read_parquet('divisions_area')"
            " WHERE country = '{country}'"
            "   AND is_territorial = TRUE"
            "   AND subtype = '{target_subtype}'"
        ),
        question_hints=[
            "Territorial {target_subtype}s of {anchor_name}",
            "Official territorial divisions of {anchor_name}",
            "Recognised territorial {target_subtype}s belonging to {anchor_name}",
            "Which territorial regions does {anchor_name} have?",
        ],
    ),

    SQLTemplate(
        template_id="attr_03",
        family="attribute_filter",
        sql_difficulty="medium",
        anchor_source="divisions_area",
        num_anchors=1,
        target_subtype="locality",
        sql_template=(
            "SELECT id, names.\"primary\" AS name, subtype, country,"
            "       ST_AsGeoJSON(geometry) AS geometry"
            " FROM read_parquet('divisions_area')"
            " WHERE country = '{country}'"
            "   AND subtype = '{target_subtype}'"
            "   AND is_land = TRUE"
        ),
        question_hints=[
            "Land-based {target_subtype}s of {anchor_name}",
            "{target_subtype}s on the mainland of {anchor_name}",
            "All {target_subtype}s on land in {anchor_name}",
            "Non-island {target_subtype}s of {anchor_name}",
        ],
    ),

]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_templates_by_family(family: str) -> List[SQLTemplate]:
    """Return all templates for a specific task family."""
    return [t for t in TEMPLATES if t.family == family]


def get_template_by_id(template_id: str) -> SQLTemplate:
    """Return a template by its ID, raising ValueError if not found."""
    for t in TEMPLATES:
        if t.template_id == template_id:
            return t
    raise ValueError(f"Template '{template_id}' not found")


if __name__ == "__main__":
    families: dict = {}
    for t in TEMPLATES:
        families[t.family] = families.get(t.family, 0) + 1

    print("SQL Template Catalog")
    print("=" * 60)
    for family, count in sorted(families.items()):
        print(f"{family:20s}: {count:2d} templates")
    print(f"{'TOTAL':20s}: {len(TEMPLATES):2d} templates")

    # Verify every template's final SELECT wraps geometry with ST_AsGeoJSON
    print()
    print("Geometry output check (all should show ST_AsGeoJSON)")
    print("=" * 60)
    for t in TEMPLATES:
        has_geojson = "ST_AsGeoJSON" in t.sql_template
        status = "OK" if has_geojson else "MISSING"
        print(f"  {t.template_id:20s}: {status}")
