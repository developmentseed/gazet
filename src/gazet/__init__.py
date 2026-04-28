"""gazet package.

Submodules are imported lazily so lightweight consumers (e.g. dataset
generation scripts that only need ``gazet.config``) don't pay the cost of
loading ``gazet.api`` and its heavy runtime deps (shapely, dspy, etc.).

Access ``gazet.api:app`` explicitly (uvicorn does this already).
"""
