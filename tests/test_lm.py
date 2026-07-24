"""Tests for gazet.lm — prompt templates, postprocessing, and GGUF helpers."""


from gazet.lm import (
    _postprocess_sql,
    _PLACES_SYSTEM_PROMPT,
    _SYSTEM_PROMPT_TEMPLATE,
    _USER_PROMPT_TEMPLATE,
    PlacesResult,
)


class TestPostprocessSql:
    def test_plain_sql(self):
        assert _postprocess_sql("SELECT * FROM foo") == "SELECT * FROM foo"

    def test_sql_fence_with_lang(self):
        raw = "```sql\nSELECT id FROM bar\n```"
        assert _postprocess_sql(raw) == "SELECT id FROM bar"

    def test_sql_fence_no_lang(self):
        raw = "```\nSELECT 1\n```"
        assert _postprocess_sql(raw) == "SELECT 1"

    def test_fence_with_trailing_text(self):
        raw = "```sql\nSELECT * FROM foo\n```\nDone!"
        result = _postprocess_sql(raw)
        assert "SELECT" in result
        assert "FROM" in result

    def test_nested_backticks_preserved(self):
        raw = "```sql\nSELECT `col`\n```"
        assert _postprocess_sql(raw) == "SELECT `col`"

    def test_whitespace_only(self):
        assert _postprocess_sql("   ") == ""

    def test_empty_string(self):
        assert _postprocess_sql("") == ""

    def test_multiline_query(self):
        raw = """```sql
SELECT
    id,
    names."primary" AS name
FROM read_parquet('divisions_area')
WHERE country = 'IN'
```"""
        result = _postprocess_sql(raw)
        assert "SELECT" in result
        assert "WHERE" in result
        assert "```" not in result

    def test_no_fence_just_whitespace_surround(self):
        raw = "  SELECT * FROM table  "
        assert _postprocess_sql(raw) == "SELECT * FROM table"


class TestPromptTemplates:
    def test_system_prompt_has_schema_placeholder(self):
        assert "{schema}" in _SYSTEM_PROMPT_TEMPLATE

    def test_system_prompt_mentions_st_asgeojson(self):
        assert "ST_AsGeoJSON" in _SYSTEM_PROMPT_TEMPLATE

    def test_user_prompt_has_candidates_placeholder(self):
        assert "{candidates_csv}" in _USER_PROMPT_TEMPLATE

    def test_user_prompt_has_question_placeholder(self):
        assert "{question}" in _USER_PROMPT_TEMPLATE

    def test_system_prompt_formatting(self):
        formatted = _SYSTEM_PROMPT_TEMPLATE.format(schema="dummy_schema")
        assert "dummy_schema" in formatted
        assert "{schema}" not in formatted

    def test_user_prompt_formatting(self):
        formatted = _USER_PROMPT_TEMPLATE.format(
            candidates_csv="source,id,name",
            question="get boundary",
        )
        assert "source,id,name" in formatted
        assert "get boundary" in formatted

    def test_places_system_prompt_has_examples(self):
        assert "EXAMPLES:" in _PLACES_SYSTEM_PROMPT
        assert "Puri, Odisha" in _PLACES_SYSTEM_PROMPT
        assert "Amazon basin" in _PLACES_SYSTEM_PROMPT


class TestGeneratePlacesFallback:
    """Test that generate_places handles parse errors gracefully.
    We can't test the real llama-server path without a running server,
    but we can verify the PlacesResult model behavior used in fallback.
    """

    def test_places_result_from_raw_query(self):
        # Simulates the fallback path in generate_places
        query = "get me the boundary of Paris"
        result = PlacesResult(places=[{"place": query}])
        assert len(result.places) == 1
        assert result.places[0].place == query

    def test_places_result_validation(self):
        data = {"places": [{"place": "Chad"}, {"place": "Lake Chad"}]}
        r = PlacesResult.model_validate(data)
        assert len(r.places) == 2
