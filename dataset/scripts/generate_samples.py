"""
Generate synthetic training samples for text-to-SQL task.

This script:
1. Loads relation tables and entity inventories
2. For each SQL template, samples valid anchors
3. Renders and executes SQL to verify it works
4. Builds candidate lists with controlled distractors
5. Generates natural language questions using LLM
6. Saves complete training samples

Output:
- output/samples/sample_*.json (individual samples)
- output/dataset_raw.jsonl (all samples)
"""

import json
import random
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import duckdb
import pandas as pd
from pydantic import BaseModel

# Suppress warnings
warnings.filterwarnings('ignore')

from gazet.config import DIVISIONS_AREA_PATH, NATURAL_EARTH_PATH

# Configurable parameters (can be overridden by CLI)
TARGET_COUNTS = None  # Will be set in main() or by CLI
MAX_WORKERS = 8
RETRY_MULTIPLIER = 2
APPEND_MODE = False

# Import templates from same directory
from . import sql_templates
TEMPLATES = sql_templates.TEMPLATES
SQLTemplate = sql_templates.SQLTemplate
get_templates_by_family = sql_templates.get_templates_by_family


class Candidate(BaseModel):
    """Candidate entity for grounding."""
    candidate_id: str
    source: str
    id: str
    name: str
    subtype: Optional[str] = None
    country: Optional[str] = None
    region: Optional[str] = None
    admin_level: Optional[int] = None
    similarity: float = 0.0


class TrainingSample(BaseModel):
    """Complete training sample."""
    id: str
    question: str
    candidates: List[Candidate]
    target: Dict[str, Any]
    metadata: Dict[str, Any]


def load_relation_tables(intermediate_dir: Path, quiet: bool = False) -> Dict[str, pd.DataFrame]:
    """Load all precomputed relation tables."""
    tables = {}
    
    for file in intermediate_dir.glob("*.parquet"):
        name = file.stem
        tables[name] = pd.read_parquet(file)
        if not quiet:
            print(f"  {name}: {len(tables[name])} rows")
    
    return tables


def sample_adjacency_anchor(adjacency_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Sample a random adjacency pair."""
    if adjacency_df.empty:
        return None
    
    row = adjacency_df.sample(n=1).iloc[0]
    return {
        'anchor_id': row['anchor_id'],
        'anchor_name': row['anchor_name'],
        'anchor_subtype': row['anchor_subtype'],
        'anchor_country': row.get('anchor_country'),  # May not exist in all tables
        'target_subtype': row.get('target_subtype')
    }


def sample_intersection_anchor(intersection_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Sample a random intersection pair."""
    if intersection_df.empty:
        return None
    
    row = intersection_df.sample(n=1).iloc[0]
    return {
        'anchor_id': row['anchor_id'],
        'anchor_name': row['anchor_name'],
        'anchor_subtype': row['anchor_subtype'],
        'target_id': row.get('target_id'),
        'target_name': row.get('target_name'),
        'target_subtype': row.get('target_subtype')
    }


def sample_containment_anchor(containment_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Sample a random containment pair."""
    if containment_df.empty:
        return None
    
    row = containment_df.sample(n=1).iloc[0]
    return {
        'container_id': row['container_id'],
        'container_name': row['container_name'],
        'container_subtype': row['container_subtype'],
        'contained_subtype': row['contained_subtype']
    }


def sample_cross_source_anchor(cross_source_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Sample a random cross-source relation."""
    if cross_source_df.empty:
        return None
    
    row = cross_source_df.sample(n=1).iloc[0]
    return {
        'division_id': row['division_id'],
        'division_name': row['division_name'],
        'division_subtype': row['division_subtype'],
        'natural_id': row['natural_id'],
        'natural_name': row['natural_name'],
        'natural_subtype': row['natural_subtype'],
        'relation_type': row['relation_type']
    }


def build_candidate_list(
    con: duckdb.DuckDBPyConnection,
    anchor_id: str,
    anchor_name: str,
    anchor_source: str,
    num_candidates: int = 10,
    difficulty: str = "medium"
) -> List[Candidate]:
    """Build candidate list with true anchor + distractors."""
    
    # Helper to convert pandas NA to None
    def safe_get(row, key, default=None):
        val = row.get(key, default)
        return None if pd.isna(val) else val
    
    # Get the true anchor
    if anchor_source == "divisions_area":
        query = """
        SELECT 
            id,
            names."primary" AS name,
            subtype,
            country,
            region,
            admin_level
        FROM read_parquet(?)
        WHERE id = ?
        """
        anchor_row = con.execute(query, [DIVISIONS_AREA_PATH, anchor_id]).fetchdf().iloc[0]
    else:
        query = """
        SELECT 
            id,
            names."primary" AS name,
            subtype
        FROM read_parquet(?)
        WHERE id = ?
        """
        anchor_row = con.execute(query, [NATURAL_EARTH_PATH, anchor_id]).fetchdf().iloc[0]
    
    # Build true candidate
    true_candidate = Candidate(
        candidate_id="c1",
        source=anchor_source,
        id=anchor_id,
        name=safe_get(anchor_row, 'name'),
        subtype=safe_get(anchor_row, 'subtype'),
        country=safe_get(anchor_row, 'country'),
        region=safe_get(anchor_row, 'region'),
        admin_level=safe_get(anchor_row, 'admin_level'),
        similarity=1.0
    )
    
    # Build distractors based on difficulty
    distractors = build_distractors(
        con, 
        anchor_name, 
        anchor_source,
        anchor_id,
        num_candidates - 1,
        difficulty
    )
    
    # Combine and shuffle
    candidates = [true_candidate] + distractors
    random.shuffle(candidates)
    
    # Reassign candidate IDs after shuffling
    for i, cand in enumerate(candidates, 1):
        cand.candidate_id = f"c{i}"
    
    return candidates


def build_distractors(
    con: duckdb.DuckDBPyConnection,
    anchor_name: str,
    anchor_source: str,
    exclude_id: str,
    num_distractors: int,
    difficulty: str
) -> List[Candidate]:
    """Build distractor candidates using fuzzy search."""
    
    # Fuzzy search for similar names
    if anchor_source == "divisions_area":
        query = """
        SELECT 
            id,
            names."primary" AS name,
            subtype,
            country,
            region,
            admin_level,
            jaro_winkler_similarity(lower(names."primary"), lower(?)) AS similarity
        FROM read_parquet(?)
        WHERE id != ?
          AND names."primary" IS NOT NULL
        ORDER BY similarity DESC
        LIMIT ?
        """
        df = con.execute(query, [
            anchor_name, DIVISIONS_AREA_PATH, exclude_id, num_distractors
        ]).fetchdf()
        source = "divisions_area"
    else:
        query = """
        SELECT 
            id,
            names."primary" AS name,
            subtype,
            jaro_winkler_similarity(lower(names."primary"), lower(?)) AS similarity
        FROM read_parquet(?)
        WHERE id != ?
          AND names."primary" IS NOT NULL
        ORDER BY similarity DESC
        LIMIT ?
        """
        df = con.execute(query, [
            anchor_name, NATURAL_EARTH_PATH, exclude_id, num_distractors
        ]).fetchdf()
        source = "natural_earth"
    
    # Helper to convert pandas NA to None
    def safe_get(row, key, default=None):
        val = row.get(key, default)
        return None if pd.isna(val) else val
    
    distractors = []
    for _, row in df.iterrows():
        distractors.append(Candidate(
            candidate_id="temp",  # Will be reassigned
            source=source,
            id=row['id'],
            name=safe_get(row, 'name'),
            subtype=safe_get(row, 'subtype'),
            country=safe_get(row, 'country'),
            region=safe_get(row, 'region'),
            admin_level=safe_get(row, 'admin_level'),
            similarity=float(row['similarity'])
        ))
    
    return distractors


def generate_adjacency_sample(
    con: duckdb.DuckDBPyConnection,
    adjacency_df: pd.DataFrame,
    sample_id: str
) -> Optional[TrainingSample]:
    """Generate a sample for adjacency task."""
    
    anchor = sample_adjacency_anchor(adjacency_df)
    if not anchor:
        return None
    
    # Build SQL
    sql = f"""WITH a AS (
  SELECT geometry FROM read_parquet('{DIVISIONS_AREA_PATH}')
  WHERE id = '{anchor['anchor_id']}'
)
SELECT b.id, b.names."primary" AS name, b.geometry
FROM read_parquet('{DIVISIONS_AREA_PATH}') AS b, a
WHERE b.id != '{anchor['anchor_id']}'
  AND b.subtype = '{anchor['target_subtype']}'
  AND ST_Touches(a.geometry, b.geometry)"""
    
    # Execute to verify
    try:
        result = con.execute(sql).fetchdf()
        if result.empty:
            return None
    except Exception as e:
        print(f"SQL execution failed: {e}")
        return None
    
    # Build candidates
    candidates = build_candidate_list(
        con,
        anchor['anchor_id'],
        anchor['anchor_name'],
        "divisions_area",
        num_candidates=10,
        difficulty="medium"
    )
    
    # Find which candidate is the true anchor
    selected_candidate_ids = [c.candidate_id for c in candidates if c.id == anchor['anchor_id']]
    
    # Generate question
    question = f"Which {anchor['target_subtype']}s border {anchor['anchor_name']}?"
    
    return TrainingSample(
        id=sample_id,
        question=question,
        candidates=candidates,
        target={
            "selected_candidates": selected_candidate_ids,
            "sql": sql
        },
        metadata={
            "task_family": "adjacency",
            "sql_difficulty": "medium",
            "grounding_difficulty": "medium",
            "template_id": "adj_02",
            "num_candidates": len(candidates),
            "anchor_source": "divisions_area",
            "sql_verified": True
        }
    )


def generate_containment_sample(
    con: duckdb.DuckDBPyConnection,
    containment_df: pd.DataFrame,
    sample_id: str
) -> Optional[TrainingSample]:
    """Generate a sample for containment task."""
    
    anchor = sample_containment_anchor(containment_df)
    if not anchor:
        return None
    
    # Build SQL
    sql = f"""WITH a AS (
  SELECT geometry FROM read_parquet('{DIVISIONS_AREA_PATH}')
  WHERE id = '{anchor['container_id']}'
)
SELECT b.id, b.names."primary" AS name, b.geometry
FROM read_parquet('{DIVISIONS_AREA_PATH}') AS b, a
WHERE b.id != '{anchor['container_id']}'
  AND b.subtype = '{anchor['contained_subtype']}'
  AND ST_Within(b.geometry, a.geometry)"""
    
    # Execute to verify
    try:
        result = con.execute(sql).fetchdf()
        if result.empty:
            return None
    except Exception as e:
        print(f"SQL execution failed: {e}")
        return None
    
    # Build candidates
    candidates = build_candidate_list(
        con,
        anchor['container_id'],
        anchor['container_name'],
        "divisions_area",
        num_candidates=10,
        difficulty="medium"
    )
    
    # Find which candidate is the true anchor
    selected_candidate_ids = [c.candidate_id for c in candidates if c.id == anchor['container_id']]
    
    # Generate question
    question = f"What {anchor['contained_subtype']}s are in {anchor['container_name']}?"
    
    return TrainingSample(
        id=sample_id,
        question=question,
        candidates=candidates,
        target={
            "selected_candidates": selected_candidate_ids,
            "sql": sql
        },
        metadata={
            "task_family": "containment",
            "sql_difficulty": "medium",
            "grounding_difficulty": "medium",
            "template_id": "contain_01",
            "num_candidates": len(candidates),
            "anchor_source": "divisions_area",
            "sql_verified": True
        }
    )


def sample_random_entity(
    con: duckdb.DuckDBPyConnection,
    inventory_df: pd.DataFrame,
    source: str
) -> Optional[Dict[str, Any]]:
    """Sample a random entity from inventory."""
    if inventory_df.empty:
        return None
    
    row = inventory_df.sample(n=1).iloc[0]
    return {
        'id': row['id'],
        'name': row['name'],
        'subtype': row.get('subtype'),
        'country': row.get('country'),
        'source': source
    }


def generate_template_based_sample(
    con: duckdb.DuckDBPyConnection,
    template: SQLTemplate,
    tables: Dict[str, pd.DataFrame],
    sample_id: str
) -> Optional[TrainingSample]:
    """Generate a sample based on a SQL template."""
    
    # Sample anchor based on template requirements
    if template.family == "direct_lookup":
        # Just pick a random entity
        if template.anchor_source == "divisions_area":
            anchor = sample_random_entity(con, tables['divisions_area_inventory'], 'divisions_area')
        else:
            anchor = sample_random_entity(con, tables['natural_earth_inventory'], 'natural_earth')
        
        if not anchor:
            return None
        
        # Render SQL
        sql = template.sql_template.format(
            DIVISIONS_AREA_PATH=DIVISIONS_AREA_PATH,
            NATURAL_EARTH_PATH=NATURAL_EARTH_PATH,
            anchor_id=anchor['id']
        )
        
        # Build candidates
        candidates = build_candidate_list(
            con, anchor['id'], anchor['name'], anchor['source'],
            num_candidates=10, difficulty="easy"
        )
        
        # Question
        question = random.choice(template.question_hints).format(anchor_name=anchor['name'])
        
    elif template.family == "adjacency":
        anchor = sample_adjacency_anchor(tables['adjacency_pairs'])
        if not anchor:
            return None
        
        sql = template.sql_template.format(
            DIVISIONS_AREA_PATH=DIVISIONS_AREA_PATH,
            NATURAL_EARTH_PATH=NATURAL_EARTH_PATH,
            anchor_id=anchor['anchor_id'],
            target_subtype=anchor['target_subtype']
        )
        
        candidates = build_candidate_list(
            con, anchor['anchor_id'], anchor['anchor_name'], 'divisions_area',
            num_candidates=10, difficulty="medium"
        )
        
        question = random.choice(template.question_hints).format(
            anchor_name=anchor['anchor_name'],
            target_subtype=anchor['target_subtype']
        )
        
    elif template.family == "containment":
        anchor = sample_containment_anchor(tables['containment_pairs'])
        if not anchor:
            return None
        
        sql = template.sql_template.format(
            DIVISIONS_AREA_PATH=DIVISIONS_AREA_PATH,
            NATURAL_EARTH_PATH=NATURAL_EARTH_PATH,
            anchor_id=anchor['container_id'],
            target_subtype=anchor['contained_subtype']
        )
        
        candidates = build_candidate_list(
            con, anchor['container_id'], anchor['container_name'], 'divisions_area',
            num_candidates=10, difficulty="medium"
        )
        
        question = random.choice(template.question_hints).format(
            anchor_name=anchor['container_name'],
            target_subtype=anchor['contained_subtype']
        )
        
    elif template.family == "intersection":
        if template.anchor_source == "natural_earth":
            anchor = sample_cross_source_anchor(tables['cross_source_relations'])
            if not anchor:
                return None
            
            sql = template.sql_template.format(
                NATURAL_EARTH_PATH=NATURAL_EARTH_PATH,
                DIVISIONS_AREA_PATH=DIVISIONS_AREA_PATH,
                anchor_id=anchor['natural_id'],
                target_subtype='country'
            )
            
            candidates = build_candidate_list(
                con, anchor['natural_id'], anchor['natural_name'], 'natural_earth',
                num_candidates=10, difficulty="medium"
            )
            
            question = random.choice(template.question_hints).format(
                anchor_name=anchor['natural_name'],
                target_subtype='country'
            )
        else:
            # Same-source intersection
            anchor = sample_intersection_anchor(tables['intersection_pairs'])
            if not anchor:
                return None
            
            # Use a generic subtype if not available
            target_subtype = anchor.get('target_subtype') or 'region'
            
            sql = template.sql_template.format(
                DIVISIONS_AREA_PATH=DIVISIONS_AREA_PATH,
                NATURAL_EARTH_PATH=NATURAL_EARTH_PATH,
                anchor_id=anchor['anchor_id'],
                target_subtype=target_subtype
            )
            
            candidates = build_candidate_list(
                con, anchor['anchor_id'], anchor['anchor_name'], 'divisions_area',
                num_candidates=10, difficulty="medium"
            )
            
            question = random.choice(template.question_hints).format(
                anchor_name=anchor['anchor_name'],
                target_subtype=target_subtype
            )
    
    elif template.family == "set_operations":
        # Union of two entities
        anchor1 = sample_random_entity(con, tables['divisions_area_inventory'], 'divisions_area')
        anchor2 = sample_random_entity(con, tables['divisions_area_inventory'], 'divisions_area')
        
        if not anchor1 or not anchor2:
            return None
        
        sql = template.sql_template.format(
            DIVISIONS_AREA_PATH=DIVISIONS_AREA_PATH,
            NATURAL_EARTH_PATH=NATURAL_EARTH_PATH,
            anchor_id_1=anchor1['id'],
            anchor_id_2=anchor2['id']
        )
        
        # Build candidates for both anchors
        candidates1 = build_candidate_list(
            con, anchor1['id'], anchor1['name'], 'divisions_area',
            num_candidates=5, difficulty="medium"
        )
        candidates2 = build_candidate_list(
            con, anchor2['id'], anchor2['name'], 'divisions_area',
            num_candidates=5, difficulty="medium"
        )
        
        # Combine and deduplicate
        candidates = candidates1 + candidates2
        seen_ids = set()
        unique_candidates = []
        for c in candidates:
            if c.id not in seen_ids:
                unique_candidates.append(c)
                seen_ids.add(c.id)
        candidates = unique_candidates[:10]
        
        # Reassign IDs
        for i, c in enumerate(candidates, 1):
            c.candidate_id = f"c{i}"
        
        question = f"{anchor1['name']} and {anchor2['name']}"
    
    elif template.family == "buffer":
        # Buffer operations
        if template.num_anchors == 1:
            anchor = sample_random_entity(con, tables['divisions_area_inventory'], 'divisions_area')
            if not anchor:
                return None
            
            buffer_degrees = random.choice([0.1, 0.5, 1.0])
            
            sql = template.sql_template.format(
                DIVISIONS_AREA_PATH=DIVISIONS_AREA_PATH,
                NATURAL_EARTH_PATH=NATURAL_EARTH_PATH,
                anchor_id=anchor['id'],
                buffer_degrees=buffer_degrees
            )
            
            candidates = build_candidate_list(
                con, anchor['id'], anchor['name'], 'divisions_area',
                num_candidates=10, difficulty="medium"
            )
            
            question = random.choice(template.question_hints).format(
                anchor_name=anchor['name'],
                buffer_degrees=buffer_degrees
            )
        else:
            # Two anchor buffer
            anchor1 = sample_random_entity(con, tables['divisions_area_inventory'], 'divisions_area')
            anchor2 = sample_random_entity(con, tables['divisions_area_inventory'], 'divisions_area')
            
            if not anchor1 or not anchor2:
                return None
            
            buffer_degrees = random.choice([0.1, 0.5])
            
            sql = template.sql_template.format(
                DIVISIONS_AREA_PATH=DIVISIONS_AREA_PATH,
                NATURAL_EARTH_PATH=NATURAL_EARTH_PATH,
                anchor_id_1=anchor1['id'],
                anchor_id_2=anchor2['id'],
                buffer_degrees=buffer_degrees
            )
            
            candidates1 = build_candidate_list(
                con, anchor1['id'], anchor1['name'], 'divisions_area',
                num_candidates=5, difficulty="medium"
            )
            candidates2 = build_candidate_list(
                con, anchor2['id'], anchor2['name'], 'divisions_area',
                num_candidates=5, difficulty="medium"
            )
            
            candidates = candidates1 + candidates2
            seen_ids = set()
            unique_candidates = []
            for c in candidates:
                if c.id not in seen_ids:
                    unique_candidates.append(c)
                    seen_ids.add(c.id)
            candidates = unique_candidates[:10]
            
            for i, c in enumerate(candidates, 1):
                c.candidate_id = f"c{i}"
            
            question = random.choice(template.question_hints).format(
                anchor_1_name=anchor1['name'],
                anchor_2_name=anchor2['name'],
                buffer_degrees=buffer_degrees
            )
    
    elif template.family == "partial_selection":
        # Partial selection (northern half, clipping, etc.)
        anchor = sample_random_entity(con, tables['divisions_area_inventory'], 'divisions_area')
        if not anchor:
            return None
        
        if template.num_anchors == 1:
            sql = template.sql_template.format(
                DIVISIONS_AREA_PATH=DIVISIONS_AREA_PATH,
                NATURAL_EARTH_PATH=NATURAL_EARTH_PATH,
                anchor_id=anchor['id']
            )
            
            question = random.choice(template.question_hints).format(anchor_name=anchor['name'])
        else:
            # Mixed source clipping
            clip_feature = sample_random_entity(con, tables['natural_earth_inventory'], 'natural_earth')
            if not clip_feature:
                return None
            
            sql = template.sql_template.format(
                DIVISIONS_AREA_PATH=DIVISIONS_AREA_PATH,
                NATURAL_EARTH_PATH=NATURAL_EARTH_PATH,
                anchor_id=anchor['id'],
                clip_feature_id=clip_feature['id']
            )
            
            question = random.choice(template.question_hints).format(
                anchor_name=anchor['name'],
                clip_feature_name=clip_feature['name']
            )
        
        candidates = build_candidate_list(
            con, anchor['id'], anchor['name'], 'divisions_area',
            num_candidates=10, difficulty="hard"
        )
    
    elif template.family == "aggregation":
        # Aggregation queries (e.g., largest N localities in a region)
        top_n = random.choice([3, 5, 10])
        
        # Check if this is a country-level query (agg_04, agg_05)
        if template.template_id in ['agg_04', 'agg_05']:
            # Country-level aggregation
            anchor = sample_random_entity(con, tables['divisions_area_inventory'], 'divisions_area')
            if not anchor:
                return None
            
            country = anchor.get('country', 'EC')
            target_subtype = random.choice(['locality', 'region'])
            
            sql = template.sql_template.format(
                DIVISIONS_AREA_PATH=DIVISIONS_AREA_PATH,
                NATURAL_EARTH_PATH=NATURAL_EARTH_PATH,
                country=country,
                target_subtype=target_subtype,
                top_n=top_n
            )
            
            candidates = build_candidate_list(
                con, anchor['id'], anchor['name'], 'divisions_area',
                num_candidates=10, difficulty="hard"
            )
            
            question = random.choice(template.question_hints).format(
                top_n=top_n,
                target_subtype=target_subtype,
                country=country
            )
        else:
            # Container-based aggregation (within a region)
            anchor = sample_containment_anchor(tables['containment_pairs'])
            if not anchor:
                return None
            
            target_subtype = anchor.get('contained_subtype', 'locality')
            
            sql = template.sql_template.format(
                DIVISIONS_AREA_PATH=DIVISIONS_AREA_PATH,
                NATURAL_EARTH_PATH=NATURAL_EARTH_PATH,
                anchor_id=anchor['container_id'],
                target_subtype=target_subtype,
                top_n=top_n
            )
            
            candidates = build_candidate_list(
                con, anchor['container_id'], anchor['container_name'], 'divisions_area',
                num_candidates=10, difficulty="hard"
            )
            
            question = random.choice(template.question_hints).format(
                top_n=top_n,
                target_subtype=target_subtype,
                anchor_name=anchor['container_name']
            )
        
    else:
        # Skip unsupported families
        return None
    
    # Execute SQL to verify
    try:
        result = con.execute(sql).fetchdf()
        if result.empty:
            return None
    except Exception as e:
        # Errors are tracked in worker return, no need to print
        return None
    
    # Find selected candidates
    if template.family == "set_operations":
        selected_candidate_ids = [c.candidate_id for c in candidates if c.id in [anchor1['id'], anchor2['id']]]
    else:
        anchor_id_to_find = anchor.get('anchor_id') or anchor.get('container_id') or anchor.get('natural_id') or anchor.get('id')
        selected_candidate_ids = [c.candidate_id for c in candidates if c.id == anchor_id_to_find]
    
    return TrainingSample(
        id=sample_id,
        question=question,
        candidates=candidates,
        target={
            "selected_candidates": selected_candidate_ids,
            "sql": sql
        },
        metadata={
            "task_family": template.family,
            "sql_difficulty": template.sql_difficulty,
            "grounding_difficulty": "medium",
            "template_id": template.template_id,
            "num_candidates": len(candidates),
            "anchor_source": template.anchor_source,
            "sql_verified": True
        }
    )


def generate_cross_source_sample(
    con: duckdb.DuckDBPyConnection,
    cross_source_df: pd.DataFrame,
    sample_id: str
) -> Optional[TrainingSample]:
    """Generate a sample for cross-source intersection task."""
    
    anchor = sample_cross_source_anchor(cross_source_df)
    if not anchor:
        return None
    
    # Build SQL (natural feature -> divisions)
    sql = f"""WITH a AS (
  SELECT geometry FROM read_parquet('{NATURAL_EARTH_PATH}')
  WHERE id = '{anchor['natural_id']}'
)
SELECT b.id, b.names."primary" AS name, b.geometry
FROM read_parquet('{DIVISIONS_AREA_PATH}') AS b, a
WHERE b.subtype = 'country'
  AND ST_Intersects(b.geometry, a.geometry)"""
    
    # Execute to verify
    try:
        result = con.execute(sql).fetchdf()
        if result.empty:
            return None
    except Exception as e:
        print(f"SQL execution failed: {e}")
        return None
    
    # Build candidates for natural feature
    candidates = build_candidate_list(
        con,
        anchor['natural_id'],
        anchor['natural_name'],
        "natural_earth",
        num_candidates=10,
        difficulty="medium"
    )
    
    # Find which candidate is the true anchor
    selected_candidate_ids = [c.candidate_id for c in candidates if c.id == anchor['natural_id']]
    
    # Generate question
    question = f"Which countries intersect the {anchor['natural_name']}?"
    
    return TrainingSample(
        id=sample_id,
        question=question,
        candidates=candidates,
        target={
            "selected_candidates": selected_candidate_ids,
            "sql": sql
        },
        metadata={
            "task_family": "intersection",
            "sql_difficulty": "medium-hard",
            "grounding_difficulty": "medium",
            "template_id": "intersect_02",
            "num_candidates": len(candidates),
            "anchor_source": "natural_earth",
            "sql_verified": True
        }
    )


def generate_sample_batch_worker(args):
    """Worker function that processes a batch of work items with a single DuckDB connection.
    
    Initializes DuckDB, spatial extension, templates module, and relation tables
    ONCE per batch, then processes all items sequentially.
    """
    from pathlib import Path
    
    work_items, intermediate_dir_str = args
    
    # Convert string back to Path
    intermediate_dir = Path(intermediate_dir_str)
    
    # Initialize DuckDB ONCE for the entire batch
    con = duckdb.connect()
    con.execute("SET enable_progress_bar=false")
    con.execute("INSTALL spatial")
    con.execute("LOAD spatial")
    
    # Load relation tables ONCE
    tables = load_relation_tables(intermediate_dir, quiet=True)
    
    # Process all items in batch
    results = []
    for family, template_dict, sample_id, _ in work_items:
        # Reconstruct template from dict (sql_templates is already imported at module level)
        template = sql_templates.SQLTemplate(**template_dict)
        try:
            sample = generate_template_based_sample(con, template, tables, sample_id)
            if sample:
                results.append((sample, family, template.template_id, None))
            else:
                results.append((None, family, template.template_id, "Empty result"))
        except Exception as e:
            results.append((None, family, template_dict.get('template_id', 'unknown'), str(e)))
    
    con.close()
    return results


def generate_batch_core(
    work_items: List[tuple],
    intermediate_dir: str,
) -> List[Dict[str, Any]]:
    """Standalone batch worker usable from Modal or any remote context.
    
    Data paths are resolved via GAZET_DATA_DIR env var (set in Modal image).
    
    Args:
        work_items: List of (family, template_dict, sample_id, _) tuples
        intermediate_dir: Path to intermediate dir with relation parquets
        
    Returns:
        List of dicts with keys: sample (dict or None), family, template_id, error
    """
    from pathlib import Path as _Path
    intermediate = _Path(intermediate_dir)
    
    con = duckdb.connect()
    con.execute("SET enable_progress_bar=false")
    con.execute("INSTALL spatial")
    con.execute("LOAD spatial")
    
    tables = load_relation_tables(intermediate, quiet=True)
    
    results = []
    for family, template_dict, sample_id, _ in work_items:
        template = sql_templates.SQLTemplate(**template_dict)
        try:
            sample = generate_template_based_sample(con, template, tables, sample_id)
            if sample:
                results.append({
                    "sample": sample.model_dump(),
                    "family": family,
                    "template_id": template.template_id,
                    "error": None,
                })
            else:
                results.append({
                    "sample": None,
                    "family": family,
                    "template_id": template.template_id,
                    "error": "Empty result",
                })
        except Exception as e:
            results.append({
                "sample": None,
                "family": family,
                "template_id": template_dict.get('template_id', 'unknown'),
                "error": str(e),
            })
    
    con.close()
    return results


def prepare_work_items(
    target_counts: Dict[str, int],
    retry_multiplier: int = 2,
    start_counter: int = 1,
    intermediate_dir_str: str = "",
) -> List[tuple]:
    """Prepare shuffled work items for sample generation.
    
    Returns list of (family, template_dict, sample_id, intermediate_dir_str) tuples.
    Reusable by both local main() and Modal orchestrator.
    """
    work_items = []
    sample_counter = start_counter
    
    for family, target_count in target_counts.items():
        if target_count == 0:
            continue
        
        family_templates = [t for t in TEMPLATES if t.family == family]
        if not family_templates:
            print(f"No templates found for {family}, skipping...")
            continue
        
        for _ in range(target_count * retry_multiplier):
            template = random.choice(family_templates)
            template_dict = {
                'template_id': template.template_id,
                'family': template.family,
                'sql_difficulty': template.sql_difficulty,
                'anchor_source': template.anchor_source,
                'num_anchors': template.num_anchors,
                'sql_template': template.sql_template,
                'question_hints': template.question_hints,
                'target_subtype': template.target_subtype,
                'requires_buffer': template.requires_buffer,
                'requires_aggregation': template.requires_aggregation
            }
            work_items.append((
                family,
                template_dict,
                f"sample_{sample_counter:06d}",
                intermediate_dir_str,
            ))
            sample_counter += 1
    
    random.shuffle(work_items)
    return work_items


def main():
    """Generate training samples."""
    global TARGET_COUNTS, MAX_WORKERS, RETRY_MULTIPLIER, APPEND_MODE
    
    # Setup paths
    script_dir = Path(__file__).parent
    intermediate_dir = script_dir.parent / "intermediate"
    output_dir = script_dir.parent / "output"
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load relation tables once to check availability
    print("Loading relation tables...")
    tables = load_relation_tables(intermediate_dir, quiet=False)
    
    # Use configured target counts or defaults
    if TARGET_COUNTS is None:
        target_counts = {
            'direct_lookup': 100,
            'adjacency': 200,
            'containment': 100,
            'intersection': 150,
            'buffer': 100,
            'set_operations': 150,
            'partial_selection': 100,
            'aggregation': 100
        }
    else:
        target_counts = TARGET_COUNTS
    
    # Load existing samples if in append mode
    existing_samples = []
    existing_sample_ids = set()
    jsonl_file = output_dir / "dataset_raw.jsonl"
    
    if APPEND_MODE and jsonl_file.exists():
        print(f"\nAppend mode: Loading existing samples from {jsonl_file}")
        with open(jsonl_file, 'r') as f:
            for line in f:
                if line.strip():
                    sample_data = json.loads(line)
                    existing_samples.append(sample_data)
                    existing_sample_ids.add(sample_data['id'])
        print(f"  Found {len(existing_samples)} existing samples")
        
        # Determine starting sample counter
        max_existing_id = max([int(s['id'].split('_')[1]) for s in existing_samples if s['id'].startswith('sample_')], default=0)
        sample_counter = max_existing_id + 1
    else:
        sample_counter = 1
    
    # Prepare work items using shared helper
    work_items = prepare_work_items(
        target_counts=target_counts,
        retry_multiplier=RETRY_MULTIPLIER,
        start_counter=sample_counter,
        intermediate_dir_str=str(intermediate_dir),
    )
    starting_sample_counter = sample_counter
    
    # Partition work items into batches (one per worker)
    num_workers = min(MAX_WORKERS, len(work_items))
    if num_workers == 0:
        print("No work items to process")
        return
    batch_size = (len(work_items) + num_workers - 1) // num_workers
    batches = []
    for i in range(0, len(work_items), batch_size):
        batch = work_items[i:i + batch_size]
        batches.append((batch, str(intermediate_dir)))
    
    # Generate samples in parallel (one batch per worker)
    active_families = len([f for f in target_counts.values() if f > 0])
    print(f"\nGenerating {len(work_items)} samples across {active_families} families...")
    print(f"  Split into {len(batches)} batches of ~{batch_size} items (1 DuckDB init per batch)")
    if APPEND_MODE and existing_samples:
        print(f"Appending: starting from sample_{starting_sample_counter:03d}")
    
    all_samples = []
    family_progress = {f: {'success': 0, 'failed': 0} for f in target_counts.keys() if target_counts[f] > 0}
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit one batch per worker
        futures = {executor.submit(generate_sample_batch_worker, batch): i for i, batch in enumerate(batches)}
        
        # Collect results as batches complete
        batches_done = 0
        for future in as_completed(futures):
            try:
                batch_results = future.result()
                for sample, family, template_id, error in batch_results:
                    if sample:
                        all_samples.append(sample)
                        family_progress[family]['success'] += 1
                    else:
                        family_progress[family]['failed'] += 1
            except Exception as e:
                print(f"\n  Batch failed: {e}")
            
            batches_done += 1
            total_done = sum(p['success'] + p['failed'] for p in family_progress.values())
            print(f"\r  Progress: {total_done}/{len(work_items)} samples ({batches_done}/{len(batches)} batches) ", end='', flush=True)
        
        print()  # New line after progress
    
    # Show distribution (keep all samples, no filtering)
    print("\nResults by family:")
    for family in sorted(family_progress.keys()):
        success = family_progress[family]['success']
        failed = family_progress[family]['failed']
        target = target_counts.get(family, 0)
        total = success + failed
        success_rate = (success / total * 100) if total > 0 else 0
        print(f"  {family:20s}: {success:3d} success / {failed:3d} failed ({success_rate:5.1f}% success rate, target: {target})")
    
    # Save combined JSONL (skip individual JSON files for speed at scale)
    print(f"\nSaving {len(all_samples)} new samples...")
    if APPEND_MODE and existing_samples:
        # Append to existing dataset
        print(f"Appending to existing dataset ({len(existing_samples)} existing samples)")
        with open(jsonl_file, 'a') as f:
            for sample in all_samples:
                f.write(json.dumps(sample.model_dump()) + '\n')
        total_samples = len(existing_samples) + len(all_samples)
    else:
        # Overwrite dataset
        with open(jsonl_file, 'w') as f:
            for sample in all_samples:
                f.write(json.dumps(sample.model_dump()) + '\n')
        total_samples = len(all_samples)
    
    print(f"\nGenerated {len(all_samples)} new samples")
    print(f"Total dataset size: {total_samples} samples")
    print(f"  Dataset: {jsonl_file}")


if __name__ == "__main__":
    main()
