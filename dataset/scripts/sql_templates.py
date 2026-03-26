"""
SQL template definitions for synthetic data generation.

Each template includes:
- Template ID
- Task family
- SQL difficulty level
- Required anchor types
- SQL template string with placeholders
- Question generation hints
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


# Template catalog
TEMPLATES = [
    # DIRECT LOOKUP (10 samples)
    SQLTemplate(
        template_id="lookup_01",
        family="direct_lookup",
        sql_difficulty="easy",
        anchor_source="divisions_area",
        num_anchors=1,
        sql_template="""SELECT geometry, names."primary" AS name, id, subtype FROM read_parquet('{DIVISIONS_AREA_PATH}') WHERE id = '{anchor_id}'""",
        question_hints=["Show me {anchor_name}", "Get the geometry of {anchor_name}", "Find {anchor_name}"]
    ),
    
    SQLTemplate(
        template_id="lookup_02",
        family="direct_lookup",
        sql_difficulty="easy",
        anchor_source="natural_earth",
        num_anchors=1,
        sql_template="""SELECT geometry, names."primary" AS name, id, subtype FROM read_parquet('{NATURAL_EARTH_PATH}') WHERE id = '{anchor_id}'""",
        question_hints=["Show me the {anchor_name}", "Get {anchor_name}", "Find the {anchor_name}"]
    ),
    
    # ADJACENCY (20 samples)
    SQLTemplate(
        template_id="adj_01",
        family="adjacency",
        sql_difficulty="medium",
        anchor_source="divisions_area",
        num_anchors=1,
        sql_template="""WITH a AS (SELECT geometry FROM read_parquet('{DIVISIONS_AREA_PATH}') WHERE id = '{anchor_id}') SELECT b.id, b.names."primary" AS name, b.geometry FROM read_parquet('{DIVISIONS_AREA_PATH}') AS b, a WHERE b.id != '{anchor_id}' AND ST_Touches(a.geometry, b.geometry)""",
        question_hints=["Which regions border {anchor_name}?", "What borders {anchor_name}?", "List places adjacent to {anchor_name}"]
    ),
    
    SQLTemplate(
        template_id="adj_02",
        family="adjacency",
        sql_difficulty="medium",
        anchor_source="divisions_area",
        num_anchors=1,
        target_subtype="region",
        sql_template="""WITH a AS (SELECT geometry FROM read_parquet('{DIVISIONS_AREA_PATH}') WHERE id = '{anchor_id}') SELECT b.id, b.names."primary" AS name, b.geometry FROM read_parquet('{DIVISIONS_AREA_PATH}') AS b, a WHERE b.id != '{anchor_id}' AND b.subtype = '{target_subtype}' AND ST_Touches(a.geometry, b.geometry)""",
        question_hints=["Which {target_subtype}s border {anchor_name}?", "What {target_subtype}s touch {anchor_name}?"]
    ),
    
    SQLTemplate(
        template_id="adj_03",
        family="adjacency",
        sql_difficulty="medium",
        anchor_source="divisions_area",
        num_anchors=1,
        target_subtype="sea",
        sql_template="""WITH a AS (SELECT geometry FROM read_parquet('{DIVISIONS_AREA_PATH}') WHERE id = '{anchor_id}') SELECT n.id, n.names."primary" AS name, n.geometry FROM read_parquet('{NATURAL_EARTH_PATH}') AS n, a WHERE n.subtype = '{target_subtype}' AND ST_Touches(a.geometry, n.geometry)""",
        question_hints=["Which {target_subtype}s touch {anchor_name}?", "What {target_subtype}s border {anchor_name}?"]
    ),
    
    # CONTAINMENT (15 samples)
    SQLTemplate(
        template_id="contain_01",
        family="containment",
        sql_difficulty="medium",
        anchor_source="divisions_area",
        num_anchors=1,
        target_subtype="locality",
        sql_template="""WITH a AS (SELECT geometry FROM read_parquet('{DIVISIONS_AREA_PATH}') WHERE id = '{anchor_id}') SELECT b.id, b.names."primary" AS name, b.geometry FROM read_parquet('{DIVISIONS_AREA_PATH}') AS b, a WHERE b.id != '{anchor_id}' AND b.subtype = '{target_subtype}' AND ST_Within(b.geometry, a.geometry)""",
        question_hints=["What {target_subtype}s are in {anchor_name}?", "Which {target_subtype}s are within {anchor_name}?"]
    ),
    
    SQLTemplate(
        template_id="contain_02",
        family="containment",
        sql_difficulty="medium",
        anchor_source="divisions_area",
        num_anchors=1,
        target_subtype="country",
        sql_template="""WITH a AS (SELECT geometry FROM read_parquet('{DIVISIONS_AREA_PATH}') WHERE id = '{anchor_id}') SELECT b.id, b.names."primary" AS name, b.geometry FROM read_parquet('{DIVISIONS_AREA_PATH}') AS b, a WHERE b.id != '{anchor_id}' AND b.subtype = '{target_subtype}' AND ST_Contains(b.geometry, a.geometry)""",
        question_hints=["What {target_subtype} contains {anchor_name}?", "Which {target_subtype} is {anchor_name} in?"]
    ),
    
    SQLTemplate(
        template_id="contain_03",
        family="containment",
        sql_difficulty="medium",
        anchor_source="natural_earth",
        num_anchors=1,
        target_subtype="region",
        sql_template="""WITH a AS (SELECT geometry FROM read_parquet('{NATURAL_EARTH_PATH}') WHERE id = '{anchor_id}') SELECT b.id, b.names."primary" AS name, b.geometry FROM read_parquet('{DIVISIONS_AREA_PATH}') AS b, a WHERE b.subtype = '{target_subtype}' AND ST_Within(b.geometry, a.geometry)""",
        question_hints=["Which {target_subtype}s are in the {anchor_name}?", "What {target_subtype}s fall within the {anchor_name}?"]
    ),
    
    # INTERSECTION (15 samples)
    SQLTemplate(
        template_id="intersect_01",
        family="intersection",
        sql_difficulty="medium-hard",
        anchor_source="divisions_area",
        num_anchors=1,
        target_subtype="region",
        sql_template="""WITH a AS (SELECT geometry FROM read_parquet('{DIVISIONS_AREA_PATH}') WHERE id = '{anchor_id}') SELECT b.id, b.names."primary" AS name, b.geometry FROM read_parquet('{DIVISIONS_AREA_PATH}') AS b, a WHERE b.id != '{anchor_id}' AND b.subtype = '{target_subtype}' AND ST_Intersects(b.geometry, a.geometry)""",
        question_hints=["Which {target_subtype}s intersect {anchor_name}?", "What {target_subtype}s overlap with {anchor_name}?"]
    ),
    
    SQLTemplate(
        template_id="intersect_02",
        family="intersection",
        sql_difficulty="medium-hard",
        anchor_source="natural_earth",
        num_anchors=1,
        target_subtype="country",
        sql_template="""WITH a AS (SELECT geometry FROM read_parquet('{NATURAL_EARTH_PATH}') WHERE id = '{anchor_id}') SELECT b.id, b.names."primary" AS name, b.geometry FROM read_parquet('{DIVISIONS_AREA_PATH}') AS b, a WHERE b.subtype = '{target_subtype}' AND ST_Intersects(b.geometry, a.geometry)""",
        question_hints=["Which {target_subtype}s intersect the {anchor_name}?", "What {target_subtype}s touch the {anchor_name}?"]
    ),
    
    # BUFFER OPERATIONS (10 samples)
    SQLTemplate(
        template_id="buffer_01",
        family="buffer",
        sql_difficulty="hard",
        anchor_source="divisions_area",
        num_anchors=1,
        requires_buffer=True,
        sql_template="""WITH a AS (SELECT ST_Buffer(geometry, {buffer_degrees}) AS geom FROM read_parquet('{DIVISIONS_AREA_PATH}') WHERE id = '{anchor_id}') SELECT b.id, b.names."primary" AS name, b.geometry FROM read_parquet('{DIVISIONS_AREA_PATH}') AS b, a WHERE b.id != '{anchor_id}' AND ST_Intersects(b.geometry, a.geom)""",
        question_hints=["A {buffer_degrees} degree buffer around {anchor_name}", "Features within {buffer_degrees} degrees of {anchor_name}"]
    ),
    
    SQLTemplate(
        template_id="buffer_02",
        family="buffer",
        sql_difficulty="hard",
        anchor_source="divisions_area",
        num_anchors=2,
        requires_buffer=True,
        sql_template="""WITH a AS (SELECT geometry AS g1 FROM read_parquet('{DIVISIONS_AREA_PATH}') WHERE id = '{anchor_id_1}'), b AS (SELECT geometry AS g2 FROM read_parquet('{DIVISIONS_AREA_PATH}') WHERE id = '{anchor_id_2}'), boundary AS (SELECT ST_Buffer(ST_Intersection(a.g1, b.g2), {buffer_degrees}) AS geom FROM a, b WHERE ST_Touches(a.g1, b.g2)) SELECT geom AS geometry FROM boundary""",
        question_hints=["A {buffer_degrees} degree buffer around the border between {anchor_1_name} and {anchor_2_name}"]
    ),
    
    # SET OPERATIONS (15 samples)
    SQLTemplate(
        template_id="union_01",
        family="set_operations",
        sql_difficulty="medium-hard",
        anchor_source="divisions_area",
        num_anchors=2,
        sql_template="""SELECT ST_Union_Agg(geometry) AS geometry, array_agg(names."primary") AS names FROM read_parquet('{DIVISIONS_AREA_PATH}') WHERE id IN ('{anchor_id_1}', '{anchor_id_2}')""",
        question_hints=["{anchor_1_name} and {anchor_2_name}", "The union of {anchor_1_name} and {anchor_2_name}"]
    ),
    
    # PARTIAL SELECTION (10 samples)
    SQLTemplate(
        template_id="partial_01",
        family="partial_selection",
        sql_difficulty="hard",
        anchor_source="divisions_area",
        num_anchors=1,
        sql_template="""WITH a AS (SELECT geometry FROM read_parquet('{DIVISIONS_AREA_PATH}') WHERE id = '{anchor_id}'), bbox AS (SELECT ST_XMin(geometry) AS xmin, ST_XMax(geometry) AS xmax, ST_YMin(geometry) AS ymin, ST_YMax(geometry) AS ymax FROM a), north_half AS (SELECT ST_MakeEnvelope(xmin, (ymin + ymax) / 2, xmax, ymax) AS half_geom FROM bbox) SELECT ST_Intersection(a.geometry, nh.half_geom) AS geometry FROM a, north_half AS nh""",
        question_hints=["The northern half of {anchor_name}", "Northern part of {anchor_name}"]
    ),
    
    SQLTemplate(
        template_id="partial_02",
        family="partial_selection",
        sql_difficulty="hard",
        anchor_source="divisions_area",
        num_anchors=1,
        sql_template="""WITH a AS (SELECT geometry FROM read_parquet('{DIVISIONS_AREA_PATH}') WHERE id = '{anchor_id}'), bbox AS (SELECT ST_XMin(geometry) AS xmin, ST_XMax(geometry) AS xmax, ST_YMin(geometry) AS ymin, ST_YMax(geometry) AS ymax FROM a), south_half AS (SELECT ST_MakeEnvelope(xmin, ymin, xmax, (ymin + ymax) / 2) AS half_geom FROM bbox) SELECT ST_Intersection(a.geometry, sh.half_geom) AS geometry FROM a, south_half AS sh""",
        question_hints=["The southern half of {anchor_name}", "Southern part of {anchor_name}"]
    ),
    
    SQLTemplate(
        template_id="partial_04",
        family="partial_selection",
        sql_difficulty="hard",
        anchor_source="divisions_area",
        num_anchors=1,
        sql_template="""WITH a AS (SELECT geometry FROM read_parquet('{DIVISIONS_AREA_PATH}') WHERE id = '{anchor_id}'), bbox AS (SELECT ST_XMin(geometry) AS xmin, ST_XMax(geometry) AS xmax, ST_YMin(geometry) AS ymin, ST_YMax(geometry) AS ymax FROM a), east_half AS (SELECT ST_MakeEnvelope((xmin + xmax) / 2, ymin, xmax, ymax) AS half_geom FROM bbox) SELECT ST_Intersection(a.geometry, eh.half_geom) AS geometry FROM a, east_half AS eh""",
        question_hints=["The eastern half of {anchor_name}", "Eastern part of {anchor_name}"]
    ),
    
    SQLTemplate(
        template_id="partial_05",
        family="partial_selection",
        sql_difficulty="hard",
        anchor_source="divisions_area",
        num_anchors=1,
        sql_template="""WITH a AS (SELECT geometry FROM read_parquet('{DIVISIONS_AREA_PATH}') WHERE id = '{anchor_id}'), bbox AS (SELECT ST_XMin(geometry) AS xmin, ST_XMax(geometry) AS xmax, ST_YMin(geometry) AS ymin, ST_YMax(geometry) AS ymax FROM a), west_half AS (SELECT ST_MakeEnvelope(xmin, ymin, (xmin + xmax) / 2, ymax) AS half_geom FROM bbox) SELECT ST_Intersection(a.geometry, wh.half_geom) AS geometry FROM a, west_half AS wh""",
        question_hints=["The western half of {anchor_name}", "Western part of {anchor_name}"]
    ),
    
    SQLTemplate(
        template_id="partial_03",
        family="partial_selection",
        sql_difficulty="hard",
        anchor_source="mixed",
        num_anchors=2,
        sql_template="""WITH a AS (SELECT geometry AS g1 FROM read_parquet('{DIVISIONS_AREA_PATH}') WHERE id = '{anchor_id}'), b AS (SELECT geometry AS g2 FROM read_parquet('{NATURAL_EARTH_PATH}') WHERE id = '{clip_feature_id}') SELECT ST_Intersection(a.g1, b.g2) AS geometry FROM a, b WHERE ST_Intersects(a.g1, b.g2)""",
        question_hints=["The part of {anchor_name} that is in the {clip_feature_name}", "{anchor_name} within the {clip_feature_name}"]
    ),
    
    # AGGREGATION (5 samples)
    SQLTemplate(
        template_id="agg_01",
        family="aggregation",
        sql_difficulty="hard",
        anchor_source="divisions_area",
        num_anchors=1,
        target_subtype="locality",
        requires_aggregation=True,
        sql_template="""WITH a AS (SELECT geometry FROM read_parquet('{DIVISIONS_AREA_PATH}') WHERE id = '{anchor_id}') SELECT b.id, b.names."primary" AS name, b.geometry, ST_Area(b.geometry) AS area FROM read_parquet('{DIVISIONS_AREA_PATH}') AS b, a WHERE ST_Within(b.geometry, a.geometry) AND b.subtype = '{target_subtype}' ORDER BY area DESC LIMIT {top_n}""",
        question_hints=["Top {top_n} largest {target_subtype}s in {anchor_name}", "Biggest {target_subtype}s in {anchor_name}", "{top_n} largest {target_subtype}s in {anchor_name}"]
    ),
    
    SQLTemplate(
        template_id="agg_02",
        family="aggregation",
        sql_difficulty="hard",
        anchor_source="divisions_area",
        num_anchors=1,
        target_subtype="locality",
        requires_aggregation=True,
        sql_template="""WITH a AS (SELECT geometry FROM read_parquet('{DIVISIONS_AREA_PATH}') WHERE id = '{anchor_id}') SELECT b.id, b.names."primary" AS name, b.geometry, ST_Area(b.geometry) AS area FROM read_parquet('{DIVISIONS_AREA_PATH}') AS b, a WHERE ST_Within(b.geometry, a.geometry) AND b.subtype = '{target_subtype}' ORDER BY area ASC LIMIT {top_n}""",
        question_hints=["Top {top_n} smallest {target_subtype}s in {anchor_name}", "Smallest {target_subtype}s in {anchor_name}", "{top_n} smallest {target_subtype}s in {anchor_name}"]
    ),
    
    SQLTemplate(
        template_id="agg_03",
        family="aggregation",
        sql_difficulty="hard",
        anchor_source="divisions_area",
        num_anchors=1,
        target_subtype="region",
        requires_aggregation=True,
        sql_template="""WITH a AS (SELECT geometry FROM read_parquet('{DIVISIONS_AREA_PATH}') WHERE id = '{anchor_id}') SELECT b.id, b.names."primary" AS name, b.geometry FROM read_parquet('{DIVISIONS_AREA_PATH}') AS b, a WHERE ST_Within(b.geometry, a.geometry) AND b.subtype = '{target_subtype}' ORDER BY RANDOM() LIMIT {top_n}""",
        question_hints=["{top_n} random {target_subtype}s in {anchor_name}", "Any {top_n} {target_subtype}s in {anchor_name}"]
    ),
    
    SQLTemplate(
        template_id="agg_04",
        family="aggregation",
        sql_difficulty="hard",
        anchor_source="divisions_area",
        num_anchors=1,
        target_subtype="locality",
        requires_aggregation=True,
        sql_template="""SELECT id, names."primary" AS name, geometry, ST_Area(geometry) AS area FROM read_parquet('{DIVISIONS_AREA_PATH}') WHERE country = '{country}' AND subtype = '{target_subtype}' ORDER BY area DESC LIMIT {top_n}""",
        question_hints=["Top {top_n} largest {target_subtype}s in {country}", "{top_n} biggest {target_subtype}s in {country}"]
    ),
    
    SQLTemplate(
        template_id="agg_05",
        family="aggregation",
        sql_difficulty="hard",
        anchor_source="divisions_area",
        num_anchors=1,
        target_subtype="locality",
        requires_aggregation=True,
        sql_template="""SELECT id, names."primary" AS name, geometry, ST_Area(geometry) AS area FROM read_parquet('{DIVISIONS_AREA_PATH}') WHERE country = '{country}' AND subtype = '{target_subtype}' ORDER BY area ASC LIMIT {top_n}""",
        question_hints=["Top {top_n} smallest {target_subtype}s in {country}", "{top_n} smallest {target_subtype}s in {country}"]
    ),
]


def get_templates_by_family(family: str) -> List[SQLTemplate]:
    """Get all templates for a specific family."""
    return [t for t in TEMPLATES if t.family == family]


def get_template_by_id(template_id: str) -> SQLTemplate:
    """Get a specific template by ID."""
    for t in TEMPLATES:
        if t.template_id == template_id:
            return t
    raise ValueError(f"Template {template_id} not found")


if __name__ == "__main__":
    # Print template summary
    families = {}
    for t in TEMPLATES:
        families[t.family] = families.get(t.family, 0) + 1
    
    print("SQL Template Catalog")
    print("=" * 60)
    for family, count in sorted(families.items()):
        print(f"{family:20s}: {count:2d} templates")
    print(f"{'TOTAL':20s}: {len(TEMPLATES):2d} templates")
