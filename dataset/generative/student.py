"""The student model's input format: one place for the prompt, shared by evaluation and the training export.

Order is schema cards -> value hints (optional) -> question, so llama.cpp can reuse a cached schema prefix.
"""
import consistency

# For models that were not fine-tuned on this data (reference ceilings, zero-shot baselines): rules spelled out.
SYSTEM_RULES = consistency.SYSTEM
# For the fine-tuned student: the rules are learned, so the prompt stays short.
SYSTEM_MINIMAL = "You write DuckDB spatial SQL. Answer with a single read-only SELECT statement and nothing else."
SYSTEMS = {"rules": SYSTEM_RULES, "minimal": SYSTEM_MINIMAL}

HINTS_HEADER = "VALUES matching the question:"


def user_prompt(cards, tables, question, hints=None):
    """`hints` is a list of 'table.column: 'v1', 'v2'' lines from hints.value_hints, or None / empty for no block."""
    parts = ["\n\n".join(cards[t] for t in tables)]
    if hints:
        parts.append(HINTS_HEADER + "\n" + "\n".join("  " + h for h in hints))
    parts.append(f"Question: {question}")
    return "\n\n".join(parts)
