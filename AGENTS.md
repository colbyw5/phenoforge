# AGENTS.md

Conventions for agentic coding tools working in this repository.
Read `DESIGN.md` before proposing architectural changes — those decisions are settled and
carry stated rationale.

## What this is

A retrieval and expansion engine for clinical value sets, exposed over MCP and consumed by a
LangGraph agent. Three layers: engine (the contribution), MCP server (thin), agent
(application). See `DESIGN.md`.

## Language and style

- Python 3.11+
- **PEP 8**, enforced by `ruff`
- **Sphinx-style docstrings** (`:param:`, `:type:`, `:returns:`, `:rtype:`, `:raises:`) on all
  public functions, classes, and methods
- Full type hints on every signature; `mypy --strict` on `src/`
- **Pydantic v2** for all data models — no bare dicts crossing module boundaries
- Prefer `pathlib` over `os.path`, `httpx` over `requests`

## Structure

```
src/phenoforge/
├── engine/       # retrieval, expansion, curated matching, provenance. No I/O to MCP or LLMs.
├── mcp/          # thin MCP server. Tool registration only — no business logic.
├── agent/        # LangGraph graph, state, nodes.
└── eval/         # ontology-aware metrics, benchmark harness.
tests/            # mirrors src/ layout
scripts/          # one-off loaders (vocabulary download, index build)
notebooks/        # Jupyter, exploration only — never pushed to production
data/             # gitignored — Athena downloads, built indices
```

**Layering rule:** `engine` imports nothing from `mcp` or `agent`. `mcp` and `agent` both
import `engine`. Violations of this break the whole architectural argument.

## Testing

- `pytest`, tests mirror the `src/` layout
- Every engine function gets a unit test; no exceptions
- Use a small fixture vocabulary (a few hundred concepts) — do not require the full Athena load
  to run the test suite
- Eval metrics get property-based tests where the invariant is clear (e.g. distance-weighted
  score is monotonic in tree distance)

## Commands

```bash
uv sync                      # install
ruff check src tests         # lint
ruff format src tests        # format
mypy src                     # types
pytest                       # tests
python scripts/load_vocab.py # build DuckDB from Athena download
python scripts/build_index.py # build LanceDB embedding index
jupyter lab notebooks/         # interactive exploration + eval walkthrough
```

## Things not to do

- Do not add a hosted vector database. DuckDB + LanceDB is a decision, not an oversight (D2).
- Do not add SNOMED. Licensing (see `DESIGN.md`, out of scope).
- Do not put business logic in `src/phenoforge/mcp/`. It is a transport adapter.
- Do not couple to stdio transport. It must stay flag-controlled (D5).
- Do not flatten `ConceptSet` into a bare code list. The provenance and unmappable-resolution
  fields are the anti-hallucination commitment and are expensive to retrofit.
- Do not commit anything from `data/`.

## Notes on MCP tool design

Tool names, parameter descriptions, and error messages are read by a model, not a human
developer. They are prompt engineering. If a tool gets called with wrong arguments, treat it as
a schema clarity bug first and a model failure second.

Generate tool schemas from the Pydantic models already defined in `engine/models.py` via
FastMCP — do not hand-write JSON schemas.
