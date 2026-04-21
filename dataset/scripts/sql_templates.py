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
disambiguation     "Place, Container" queries like "Puri, Odisha" — lookup by
                   ID after resolving an ambiguous name via containing region
                   or country mentioned in the query.
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
            "show me {anchor_name}",
            "get the boundary of {anchor_name}",
            "find {anchor_name}",
            "where is {anchor_name}?",
            "outline of {anchor_name}",
            "map {anchor_name}",
            "what does {anchor_name} look like",
            "i need the shape of {anchor_name}",
            "pull up {anchor_name}",
            "can you show {anchor_name}",
            "map of {anchor_name}",
            "{anchor_name} boundary",
            "locate {anchor_name}",
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
            "show me the {anchor_name}",
            "get the {anchor_name}",
            "find the {anchor_name}",
            "where is the {anchor_name}?",
            "extent of the {anchor_name}",
            "geometry of the {anchor_name}",
            "display the {anchor_name}",
            "pull up the {anchor_name}",
            "i want to see the {anchor_name}",
            "map the {anchor_name}",
            "how big is the {anchor_name}?",
            "outline of the {anchor_name}",
        ],
    ),

    # ── DISAMBIGUATION ──────────────────────────────────────────────────────
    # "Puri, Odisha", "Lisbon, Portugal", "Goa, India" — a common real-world
    # query pattern where users give a place plus its containing region or
    # country to disambiguate same-name localities.
    # SQL is a plain lookup by id (disambiguation happens at candidate-pick
    # time). Candidates include same-name localities in other regions plus
    # the container, so the model must read the CSV to choose correctly.
    #
    # disambiguate_01: locality scoped by its region / county
    # disambiguate_02: locality scoped by its country
    # disambiguate_03: region / dependency scoped by its country

    SQLTemplate(
        template_id="disambiguate_01",
        family="disambiguation",
        sql_difficulty="easy",
        anchor_source="divisions_area",
        num_anchors=1,
        sql_template=(
            "SELECT ST_AsGeoJSON(geometry) AS geometry,"
            " names.\"primary\" AS name, id, subtype, country, region"
            " FROM read_parquet('divisions_area')"
            " WHERE id = '{anchor_id}'"
        ),
        question_hints=[
            "{anchor_name}, {container_name}",
            "{anchor_name} in {container_name}",
            "the {anchor_name} that's in {container_name}",
            "show me {anchor_name}, {container_name}",
            "where is {anchor_name}, {container_name}?",
            "map of {anchor_name} ({container_name})",
            "{anchor_name} ({container_name})",
            "{anchor_name} {container_name}",
            "pull up {anchor_name} in {container_name}",
            "find {anchor_name} in {container_name}",
        ],
    ),

    SQLTemplate(
        template_id="disambiguate_02",
        family="disambiguation",
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
            "{anchor_name}, {container_name}",
            "{anchor_name} in {container_name}",
            "{anchor_name}, {container_name}.",
            "show me {anchor_name}, {container_name}",
            "where is {anchor_name} in {container_name}?",
            "the {anchor_name} that's in {container_name}",
            "map of {anchor_name}, {container_name}",
            "pull up {anchor_name} ({container_name})",
            "find {anchor_name} in {container_name}",
            "{anchor_name} {container_name}",
        ],
    ),

    SQLTemplate(
        template_id="disambiguate_03",
        family="disambiguation",
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
            "{anchor_name}, {container_name}",
            "{anchor_name} state of {container_name}",
            "the {anchor_name} region in {container_name}",
            "show me {anchor_name}, {container_name}",
            "where is {anchor_name} in {container_name}?",
            "map of {anchor_name}, {container_name}",
            "{anchor_name} ({container_name})",
            "{anchor_name} province of {container_name}",
            "pull up {anchor_name} in {container_name}",
            "find {anchor_name} {container_name}",
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
            "which regions border {anchor_name}?",
            "what places touch {anchor_name}",
            "list everything adjacent to {anchor_name}",
            "what shares a border with {anchor_name}",
            "neighbours of {anchor_name}",
            "what's next to {anchor_name}",
            "what surrounds {anchor_name}?",
            "places next to {anchor_name}",
            "everything bordering {anchor_name}",
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
            "which {target_subtype}s border {anchor_name}?",
            "what {target_subtype}s share a border with {anchor_name}",
            "{target_subtype}s that touch {anchor_name}",
            "neighbouring {target_subtype}s of {anchor_name}",
            "which {target_subtype}s are adjacent to {anchor_name}?",
            "{target_subtype}s along the {anchor_name} border",
            "find {target_subtype}s next to {anchor_name}",
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
            "which seas touch {anchor_name}?",
            "what seas border {anchor_name}?",
            "which bodies of water is {anchor_name} next to?",
            "what ocean or sea borders {anchor_name}",
            "which oceans touch {anchor_name}?",
            "what coastline does {anchor_name} have?",
            "which water bodies does {anchor_name} border?",
            "does {anchor_name} have sea access?",
            "what ocean is {anchor_name} on?",
        ],
    ),

    SQLTemplate(
        template_id="adj_06",
        family="adjacency",
        sql_difficulty="medium",
        anchor_source="divisions_area",
        num_anchors=1,
        target_subtype="county",
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
            "neighbouring counties of {anchor_name}",
            "neighbouring districts of {anchor_name}",
            "which counties border {anchor_name}?",
            "which districts border {anchor_name}?",
            "counties adjacent to {anchor_name}",
            "districts next to {anchor_name}",
            "counties sharing a border with {anchor_name}",
            "what counties touch {anchor_name}?",
            "nearby counties of {anchor_name}",
            "counties along the {anchor_name} boundary",
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
            "which regions border both {anchor_1_name} and {anchor_2_name}?",
            "what places touch both {anchor_1_name} and {anchor_2_name}?",
            "regions adjacent to both {anchor_1_name} and {anchor_2_name}",
            "what lies between {anchor_1_name} and {anchor_2_name}?",
            "common neighbours of {anchor_1_name} and {anchor_2_name}",
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
            "what {target_subtype}s are in {anchor_name}?",
            "which {target_subtype}s fall within {anchor_name}?",
            "list all {target_subtype}s inside {anchor_name}",
            "{target_subtype}s contained by {anchor_name}",
            "all {target_subtype}s within {anchor_name}",
            "{target_subtype}s of {anchor_name}",
            "show every {target_subtype} in {anchor_name}",
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
            "what country contains {anchor_name}?",
            "which country is {anchor_name} in?",
            "what country does {anchor_name} belong to?",
            "which nation contains {anchor_name}?",
            "{anchor_name} is part of which country?",
            "where is {anchor_name}",
            "what country is {anchor_name} in",
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
            "which {target_subtype}s are in the {anchor_name}?",
            "what {target_subtype}s fall within the {anchor_name}?",
            "{target_subtype}s inside the {anchor_name}",
            "admin {target_subtype}s within the {anchor_name}",
            "all regions inside the {anchor_name}",
            "what {target_subtype}s does the {anchor_name} contain?",
            "{target_subtype}s covered by the {anchor_name}",
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
            "which {target_subtype}s intersect {anchor_name}?",
            "what {target_subtype}s overlap with {anchor_name}?",
            "{target_subtype}s that cross into {anchor_name}",
            "which {target_subtype}s overlap {anchor_name}?",
            "{target_subtype}s partially inside {anchor_name}",
            "what {target_subtype}s extend into {anchor_name}?",
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
            "which countries does the {anchor_name} pass through?",
            "what countries does the {anchor_name} cross?",
            "countries that overlap the {anchor_name}",
            "which countries touch the {anchor_name}?",
            "nations intersected by the {anchor_name}",
            "which nations does the {anchor_name} cross?",
            "countries along the {anchor_name}",
            "what countries does the {anchor_name} cover?",
            "countries the {anchor_name} spans across",
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
            "what's within {buffer_km} km of {anchor_name}?",
            "admin units within {buffer_km} km of {anchor_name}",
            "features within a {buffer_km} km radius of {anchor_name}",
            "places within {buffer_km} kilometers of {anchor_name}",
            "{buffer_km} km buffer around {anchor_name}",
            "what falls within {buffer_km} km of {anchor_name}?",
            "everything within {buffer_km} km of {anchor_name}",
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
            "what's within {buffer_m} meters of {anchor_name}?",
            "features within {buffer_m} m of {anchor_name}",
            "places within {buffer_m} metres of {anchor_name}",
            "{buffer_m} meter buffer around {anchor_name}",
            "what falls within {buffer_m} m of {anchor_name}?",
            "admin units within {buffer_m} metres of {anchor_name}",
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
            "what admin units are within {buffer_km} km of the {anchor_name}?",
            "countries within {buffer_km} km of the {anchor_name}",
            "regions within {buffer_km} km of the {anchor_name}",
            "what falls within {buffer_km} km of the {anchor_name}?",
            "admin divisions within a {buffer_km} km radius of the {anchor_name}",
            "places within {buffer_km} kilometers of the {anchor_name}",
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
            "what's within {buffer_m} meters of the {anchor_name}?",
            "admin units within {buffer_m} m of the {anchor_name}",
            "places within {buffer_m} metres of the {anchor_name}",
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
            "coastal {target_subtype}s of {anchor_name}",
            "{target_subtype}s in {anchor_name} with sea access",
            "which {target_subtype}s in {anchor_name} are on the coast?",
            "seaside {target_subtype}s within {anchor_name}",
            "{target_subtype}s in {anchor_name} bordering the sea",
            "oceanfront {target_subtype}s in {anchor_name}",
            "which {target_subtype}s in {anchor_name} have a coastline?",
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
            "landlocked {target_subtype}s in {anchor_name}",
            "which {target_subtype}s in {anchor_name} have no sea access?",
            "{target_subtype}s in {anchor_name} that are landlocked",
            "{target_subtype}s in {anchor_name} with no coastline",
            "which {target_subtype}s within {anchor_name} are landlocked?",
            "interior {target_subtype}s of {anchor_name} with no ocean border",
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
            "{target_subtype}s in {anchor_name} on a terrain feature or island",
            "{target_subtype}s of {anchor_name} on a peninsula or island group",
            "{target_subtype}s within {anchor_name} on notable landforms",
            "island and peninsula {target_subtype}s of {anchor_name}",
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
            "the part of {anchor_1_name} that is not in {anchor_2_name}",
            "{anchor_1_name} without the {anchor_2_name} area",
            "remove {anchor_2_name} from {anchor_1_name}",
            "{anchor_1_name} with {anchor_2_name} cut out",
            "subtract {anchor_2_name} from {anchor_1_name}",
            "what's left of {anchor_1_name} after removing {anchor_2_name}?",
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
            "the part of {anchor_name} outside the {clip_feature_name}",
            "{anchor_name} excluding the {clip_feature_name}",
            "{anchor_name} minus the {clip_feature_name}",
            "parts of {anchor_name} not covered by the {clip_feature_name}",
            "{anchor_name} with the {clip_feature_name} removed",
            "what's left of {anchor_name} after removing the {clip_feature_name}?",
            "show me {anchor_name} excluding the {clip_feature_name}",
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
            "the {buffer_km} km border corridor between {anchor_1_name} and {anchor_2_name}",
            "area within {buffer_km} km of the {anchor_1_name}-{anchor_2_name} border",
            "the region straddling the border of {anchor_1_name} and {anchor_2_name} within {buffer_km} km",
            "{buffer_km} km on either side of the {anchor_1_name} and {anchor_2_name} border",
            "buffer the {anchor_1_name}-{anchor_2_name} boundary by {buffer_km} km",
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
            "the combined area of {anchor_1_name} and {anchor_2_name}",
            "union of {anchor_1_name} and {anchor_2_name}",
            "merge {anchor_1_name} and {anchor_2_name}",
            "{anchor_1_name} and {anchor_2_name} together",
            "combined geometry of {anchor_1_name} and {anchor_2_name}",
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
            "show me {anchor_1_name}, {anchor_2_name} and {anchor_3_name}",
            "the combined area of {anchor_1_name}, {anchor_2_name} and {anchor_3_name}",
            "union of {anchor_1_name}, {anchor_2_name} and {anchor_3_name}",
            "merge {anchor_1_name}, {anchor_2_name} and {anchor_3_name}",
            "{anchor_1_name}, {anchor_2_name} and {anchor_3_name} together",
            "display {anchor_1_name}, {anchor_2_name} and {anchor_3_name}",
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
            "all {target_subtype}s in {anchor_1_name} and {anchor_2_name}",
            "show {target_subtype}s across {anchor_1_name} and {anchor_2_name}",
            "{target_subtype}s belonging to {anchor_1_name} and {anchor_2_name}",
            "list {target_subtype}s in both {anchor_1_name} and {anchor_2_name}",
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
            "all {target_subtype}s in {anchor_1_name}, {anchor_2_name} and {anchor_3_name}",
            "show {target_subtype}s across {anchor_1_name}, {anchor_2_name} and {anchor_3_name}",
            "list {target_subtype}s in {anchor_1_name}, {anchor_2_name} and {anchor_3_name}",
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
            "merge all {target_subtype}s in {anchor_name} into one geometry",
            "combined geometry of all {target_subtype}s in {anchor_name}",
            "union of all {target_subtype}s within {anchor_name}",
            "all {target_subtype}s of {anchor_name} merged together",
            "the overall extent of {target_subtype}s in {anchor_name}",
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
            "the northern half of {anchor_name}",
            "northern part of {anchor_name}",
            "the top half of {anchor_name}",
            "northern portion of {anchor_name}",
            "upper half of {anchor_name}",
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
            "the southern half of {anchor_name}",
            "southern part of {anchor_name}",
            "the bottom half of {anchor_name}",
            "southern portion of {anchor_name}",
            "lower half of {anchor_name}",
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
            "the eastern half of {anchor_name}",
            "eastern part of {anchor_name}",
            "the right half of {anchor_name}",
            "eastern portion of {anchor_name}",
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
            "the western half of {anchor_name}",
            "western part of {anchor_name}",
            "the left half of {anchor_name}",
            "western portion of {anchor_name}",
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
            "the part of {anchor_name} that overlaps the {clip_feature_name}",
            "{anchor_name} within the {clip_feature_name}",
            "the portion of {anchor_name} inside the {clip_feature_name}",
            "part of the {clip_feature_name} in {anchor_name}",
            "part of {anchor_name} in the {clip_feature_name}",
            "clip {anchor_name} to the {clip_feature_name}",
            "{anchor_name} clipped to the {clip_feature_name}",
            "{clip_feature_name} inside {anchor_name}",
            "parts of {anchor_name} covered by the {clip_feature_name}",
            "show me where {anchor_name} and the {clip_feature_name} overlap",
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
            "top {top_n} largest {target_subtype}s in {anchor_name}",
            "biggest {top_n} {target_subtype}s in {anchor_name}",
            "{top_n} largest {target_subtype}s inside {anchor_name}",
            "the {top_n} biggest {target_subtype}s within {anchor_name}",
            "largest {target_subtype} in {anchor_name}",
            "which {target_subtype} in {anchor_name} has the most area?",
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
            "top {top_n} smallest {target_subtype}s in {anchor_name}",
            "smallest {top_n} {target_subtype}s in {anchor_name}",
            "{top_n} smallest {target_subtype}s inside {anchor_name}",
            "the {top_n} tiniest {target_subtype}s within {anchor_name}",
            "smallest {target_subtype} in {anchor_name}",
            "which {target_subtype} in {anchor_name} has the least area?",
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
            "top {top_n} largest {target_subtype}s in {anchor_name}",
            "{top_n} biggest {target_subtype}s in {anchor_name}",
            "largest {top_n} {target_subtype}s in {anchor_name}",
            "the {top_n} largest {target_subtype}s in {anchor_name}",
            "biggest {target_subtype} in {anchor_name}",
            "which {target_subtype} in {anchor_name} is the largest?",
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
            "top {top_n} smallest {target_subtype}s in {anchor_name}",
            "{top_n} smallest {target_subtype}s in {anchor_name}",
            "smallest {top_n} {target_subtype}s in {anchor_name}",
            "the {top_n} smallest {target_subtype}s in {anchor_name}",
            "smallest {target_subtype} in {anchor_name}",
            "which {target_subtype} in {anchor_name} is the smallest?",
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
            "the largest {target_subtype} in each region of {anchor_name}",
            "biggest {target_subtype} per region in {anchor_name}",
            "largest {target_subtype} for every region of {anchor_name}",
            "the biggest {target_subtype} in each province of {anchor_name}",
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
            "the smallest {target_subtype} in each region of {anchor_name}",
            "smallest {target_subtype} per region in {anchor_name}",
            "tiniest {target_subtype} for every region of {anchor_name}",
            "the smallest {target_subtype} in each province of {anchor_name}",
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
            "island territories of {anchor_name}",
            "overseas island {target_subtype}s belonging to {anchor_name}",
            "which islands are part of {anchor_name}?",
            "land territories of {anchor_name}",
            "island possessions of {anchor_name}",
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
            "territorial {target_subtype}s of {anchor_name}",
            "official territorial divisions of {anchor_name}",
            "recognised territorial {target_subtype}s belonging to {anchor_name}",
            "which territorial regions does {anchor_name} have?",
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
            "land-based {target_subtype}s of {anchor_name}",
            "{target_subtype}s on the mainland of {anchor_name}",
            "all {target_subtype}s on land in {anchor_name}",
            "non-island {target_subtype}s of {anchor_name}",
        ],
    ),

    # ── NATURAL EARTH ADJACENCY ─────────────────────────────────────────────
    # Division anchor, natural_earth targets. Handler formats anchor_id and
    # target_subtype but the SQL hardcodes NE subtypes (like adj_03).

    SQLTemplate(
        template_id="adj_04",
        family="adjacency",
        sql_difficulty="medium",
        anchor_source="divisions_area",
        num_anchors=1,
        target_subtype="river",
        sql_template=(
            "WITH a AS ("
            "  SELECT geometry FROM read_parquet('divisions_area') WHERE id = '{anchor_id}'"
            ")"
            " SELECT n.id, n.names.\"primary\" AS name, n.subtype,"
            "        ST_AsGeoJSON(n.geometry) AS geometry"
            " FROM read_parquet('natural_earth') AS n, a"
            " WHERE n.subtype IN ('River', 'Lake', 'Basin')"
            "   AND ST_Intersects(a.geometry, n.geometry)"
        ),
        question_hints=[
            "what rivers or lakes are in {anchor_name}?",
            "natural water features of {anchor_name}",
            "which rivers flow through {anchor_name}?",
            "lakes and rivers within {anchor_name}",
            "water features inside {anchor_name}",
            "what bodies of water cross {anchor_name}?",
            "rivers of {anchor_name}",
            "show me the lakes in {anchor_name}",
        ],
    ),

    SQLTemplate(
        template_id="adj_05",
        family="adjacency",
        sql_difficulty="medium",
        anchor_source="divisions_area",
        num_anchors=1,
        target_subtype="range",
        sql_template=(
            "WITH a AS ("
            "  SELECT geometry FROM read_parquet('divisions_area') WHERE id = '{anchor_id}'"
            ")"
            " SELECT n.id, n.names.\"primary\" AS name, n.subtype,"
            "        ST_AsGeoJSON(n.geometry) AS geometry"
            " FROM read_parquet('natural_earth') AS n, a"
            " WHERE n.subtype IN ('Range/Mts', 'Terrain area', 'Peninsula', 'Depression')"
            "   AND ST_Intersects(a.geometry, n.geometry)"
        ),
        question_hints=[
            "what mountain ranges are in {anchor_name}?",
            "terrain features of {anchor_name}",
            "which mountain ranges cross {anchor_name}?",
            "landforms inside {anchor_name}",
            "peninsulas and ranges in {anchor_name}",
            "geographic features within {anchor_name}",
            "mountains of {anchor_name}",
            "what terrain does {anchor_name} contain?",
        ],
    ),

    # ── NATURAL EARTH INTERSECTION ──────────────────────────────────────────
    # intersect_03: NE anchor, finding overlapping regions (vs countries in
    # intersect_02). Uses cross_source_relations handler.
    # intersect_04: division anchor, finding NE features that overlap it.
    # Uses intersection_pairs handler (extra NE subtypes ignored in SQL).

    SQLTemplate(
        template_id="intersect_03",
        family="intersection",
        sql_difficulty="medium-hard",
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
            "   AND ST_Intersects(b.geometry, a.geometry)"
        ),
        question_hints=[
            "which regions does the {anchor_name} pass through?",
            "what admin regions overlap with the {anchor_name}?",
            "regions that the {anchor_name} crosses",
            "admin areas intersected by the {anchor_name}",
            "what provinces does the {anchor_name} span?",
            "regions along the {anchor_name}",
            "which provinces overlap the {anchor_name}?",
        ],
    ),

    SQLTemplate(
        template_id="intersect_04",
        family="intersection",
        sql_difficulty="medium-hard",
        anchor_source="divisions_area",
        num_anchors=1,
        target_subtype="region",
        sql_template=(
            "WITH a AS ("
            "  SELECT geometry FROM read_parquet('divisions_area') WHERE id = '{anchor_id}'"
            ")"
            " SELECT n.id, n.names.\"primary\" AS name, n.subtype,"
            "        ST_AsGeoJSON(n.geometry) AS geometry"
            " FROM read_parquet('natural_earth') AS n, a"
            " WHERE ST_Intersects(n.geometry, a.geometry)"
        ),
        question_hints=[
            "what natural features intersect {anchor_name}?",
            "natural features that overlap {anchor_name}",
            "which geographic features cross {anchor_name}?",
            "everything natural that touches {anchor_name}",
            "what geographic features does {anchor_name} contain?",
            "natural features within or crossing {anchor_name}",
        ],
    ),

    # ── NATURAL EARTH CHAINED ───────────────────────────────────────────────
    # chained_04: localities in a region that intersect a river or lake.
    # chained_05: localities in a region that lie on a mountain range.

    SQLTemplate(
        template_id="chained_04",
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
            "     WHERE n.subtype IN ('River', 'Lake', 'Basin')"
            "       AND ST_Intersects(b.geometry, n.geometry)"
            "   )"
        ),
        question_hints=[
            "riverside {target_subtype}s in {anchor_name}",
            "{target_subtype}s in {anchor_name} near a river or lake",
            "which {target_subtype}s in {anchor_name} are on a waterway?",
            "lakeside or riverside {target_subtype}s within {anchor_name}",
            "{target_subtype}s in {anchor_name} that touch a river",
            "which {target_subtype}s in {anchor_name} are on a lake?",
            "waterfront {target_subtype}s of {anchor_name}",
        ],
    ),

    SQLTemplate(
        template_id="chained_05",
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
            "     WHERE n.subtype IN ('Range/Mts', 'Depression')"
            "       AND ST_Intersects(b.geometry, n.geometry)"
            "   )"
        ),
        question_hints=[
            "mountain {target_subtype}s in {anchor_name}",
            "{target_subtype}s in {anchor_name} on a mountain range",
            "which {target_subtype}s in {anchor_name} are in the mountains?",
            "highland {target_subtype}s within {anchor_name}",
            "{target_subtype}s of {anchor_name} in mountainous terrain",
            "{target_subtype}s in {anchor_name} near a mountain range",
        ],
    ),

    # ── CHAINED (county-level) ──────────────────────────────────────────────
    # Same spatial patterns as chained_01..05 but targeting counties/districts
    # so the model learns "coastal districts of X", "riverside counties", etc.

    SQLTemplate(
        template_id="chained_06",
        family="chained",
        sql_difficulty="hard",
        anchor_source="divisions_area",
        num_anchors=1,
        target_subtype="county",
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
            "coastal {target_subtype}s of {anchor_name}",
            "which districts of {anchor_name} are on the coast?",
            "{target_subtype}s in {anchor_name} that border the sea",
            "seaside {target_subtype}s within {anchor_name}",
            "{target_subtype}s of {anchor_name} with ocean access",
            "which {target_subtype}s in {anchor_name} touch the sea?",
            "maritime {target_subtype}s of {anchor_name}",
        ],
    ),

    SQLTemplate(
        template_id="chained_07",
        family="chained",
        sql_difficulty="hard",
        anchor_source="divisions_area",
        num_anchors=1,
        target_subtype="county",
        sql_template=(
            "WITH region AS ("
            "  SELECT geometry FROM read_parquet('divisions_area') WHERE id = '{anchor_id}'"
            ")"
            " SELECT b.id, b.names.\"primary\" AS name, b.subtype, b.country,"
            "        ST_AsGeoJSON(b.geometry) AS geometry"
            " FROM read_parquet('divisions_area') AS b, region"
            " WHERE b.subtype = '{target_subtype}'"
            "   AND ST_Within(b.geometry, region.geometry)"
            "   AND NOT EXISTS ("
            "     SELECT 1 FROM read_parquet('natural_earth') AS n"
            "     WHERE n.subtype IN ('ocean', 'sea')"
            "       AND ST_Intersects(b.geometry, n.geometry)"
            "   )"
        ),
        question_hints=[
            "landlocked {target_subtype}s of {anchor_name}",
            "which districts of {anchor_name} have no coastline?",
            "interior {target_subtype}s within {anchor_name}",
            "{target_subtype}s in {anchor_name} with no sea access",
            "non-coastal {target_subtype}s of {anchor_name}",
            "inland {target_subtype}s of {anchor_name}",
        ],
    ),

    SQLTemplate(
        template_id="chained_08",
        family="chained",
        sql_difficulty="hard",
        anchor_source="divisions_area",
        num_anchors=1,
        target_subtype="county",
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
            "     WHERE n.subtype IN ('River', 'Lake', 'Basin')"
            "       AND ST_Intersects(b.geometry, n.geometry)"
            "   )"
        ),
        question_hints=[
            "riverside {target_subtype}s of {anchor_name}",
            "which districts of {anchor_name} have a river or lake?",
            "{target_subtype}s in {anchor_name} on a waterway",
            "lakeside {target_subtype}s within {anchor_name}",
            "{target_subtype}s of {anchor_name} along a river",
            "which {target_subtype}s in {anchor_name} border a lake?",
        ],
    ),

    SQLTemplate(
        template_id="chained_09",
        family="chained",
        sql_difficulty="hard",
        anchor_source="divisions_area",
        num_anchors=1,
        target_subtype="county",
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
            "     WHERE n.subtype IN ('Range/Mts', 'Depression')"
            "       AND ST_Intersects(b.geometry, n.geometry)"
            "   )"
        ),
        question_hints=[
            "mountain {target_subtype}s of {anchor_name}",
            "which districts of {anchor_name} are in the mountains?",
            "{target_subtype}s in {anchor_name} on a mountain range",
            "highland {target_subtype}s within {anchor_name}",
            "{target_subtype}s of {anchor_name} in mountainous terrain",
            "which {target_subtype}s in {anchor_name} have mountain ranges?",
        ],
    ),

    # chained_10 / chained_11: coastal and inland REGIONS of a country.
    # Same pattern as chained_06/07 but with target_subtype='region' and
    # container forced to a country so phrasings like "coastal states of
    # India" / "inland provinces of Kenya" work correctly.

    SQLTemplate(
        template_id="chained_10",
        family="chained",
        sql_difficulty="hard",
        anchor_source="divisions_area",
        num_anchors=1,
        target_subtype="region",
        sql_template=(
            "WITH country AS ("
            "  SELECT geometry FROM read_parquet('divisions_area') WHERE id = '{anchor_id}'"
            ")"
            " SELECT b.id, b.names.\"primary\" AS name, b.subtype, b.country,"
            "        ST_AsGeoJSON(b.geometry) AS geometry"
            " FROM read_parquet('divisions_area') AS b, country"
            " WHERE b.subtype = '{target_subtype}'"
            "   AND ST_Within(b.geometry, country.geometry)"
            "   AND EXISTS ("
            "     SELECT 1 FROM read_parquet('natural_earth') AS n"
            "     WHERE n.subtype IN ('ocean', 'sea')"
            "       AND ST_Intersects(b.geometry, n.geometry)"
            "   )"
        ),
        question_hints=[
            "coastal states of {anchor_name}",
            "coastal regions of {anchor_name}",
            "coastal provinces of {anchor_name}",
            "which states of {anchor_name} are on the coast?",
            "regions of {anchor_name} with sea access",
            "states of {anchor_name} that border the ocean",
            "maritime states of {anchor_name}",
            "seaside regions of {anchor_name}",
            "which provinces of {anchor_name} touch the sea?",
            "states of {anchor_name} along the coast",
        ],
    ),

    SQLTemplate(
        template_id="chained_11",
        family="chained",
        sql_difficulty="hard",
        anchor_source="divisions_area",
        num_anchors=1,
        target_subtype="region",
        sql_template=(
            "WITH country AS ("
            "  SELECT geometry FROM read_parquet('divisions_area') WHERE id = '{anchor_id}'"
            ")"
            " SELECT b.id, b.names.\"primary\" AS name, b.subtype, b.country,"
            "        ST_AsGeoJSON(b.geometry) AS geometry"
            " FROM read_parquet('divisions_area') AS b, country"
            " WHERE b.subtype = '{target_subtype}'"
            "   AND ST_Within(b.geometry, country.geometry)"
            "   AND NOT EXISTS ("
            "     SELECT 1 FROM read_parquet('natural_earth') AS n"
            "     WHERE n.subtype IN ('ocean', 'sea')"
            "       AND ST_Intersects(b.geometry, n.geometry)"
            "   )"
        ),
        question_hints=[
            "landlocked states of {anchor_name}",
            "inland regions of {anchor_name}",
            "non-coastal states of {anchor_name}",
            "which states of {anchor_name} have no coast?",
            "inland provinces of {anchor_name}",
            "regions of {anchor_name} without sea access",
            "interior states of {anchor_name}",
            "states of {anchor_name} that don't border the ocean",
        ],
    ),

    # ── NATURAL EARTH CONTAINMENT ───────────────────────────────────────────
    # contain_04: NE anchor (sea/gulf/bay), find countries that touch it.
    # Uses containment handler via containment_pairs.

    SQLTemplate(
        template_id="contain_04",
        family="containment",
        sql_difficulty="medium",
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
            "which countries border the {anchor_name}?",
            "what countries are along the {anchor_name}?",
            "countries surrounding the {anchor_name}",
            "nations on the {anchor_name}",
            "which countries touch the {anchor_name}?",
            "countries with coastline on the {anchor_name}",
            "what nations lie on the {anchor_name}?",
        ],
    ),

    # ── NATURAL EARTH BUFFER ────────────────────────────────────────────────
    # buffer_05: NE anchor, find other NE features within a buffer distance.
    # Uses buffer handler for natural_earth.

    SQLTemplate(
        template_id="buffer_05",
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
            " SELECT n.id, n.names.\"primary\" AS name, n.subtype,"
            "        ST_AsGeoJSON(n.geometry) AS geometry"
            " FROM read_parquet('natural_earth') AS n, a"
            " WHERE ST_Intersects(n.geometry, a.geom)"
        ),
        question_hints=[
            "natural features within {buffer_km} km of the {anchor_name}",
            "what's within {buffer_km} km of the {anchor_name}?",
            "geographic features near the {anchor_name} within {buffer_km} km",
            "everything within {buffer_km} km of the {anchor_name}",
            "what natural features are close to the {anchor_name}?",
            "{buffer_km} km radius around the {anchor_name}",
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
