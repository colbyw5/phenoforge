# phenoforge

> Semantic value set assembly for clinical cohort definitions, over MCP.

**Status: early prototype.** The vocabulary loader, hierarchy expansion, hybrid (BM25 + dense)
search, curated OHDSI Phenotype Library matching, an eval harness scoring all of it against
curated ground truth, a thin MCP server exposing them all, and a LangGraph agent that decomposes
a population description, checks curated first, and pauses for human confirmation before
including anything generated — all work end-to-end against a real Athena download. See Roadmap.

## Setup

Requires your own [OHDSI Athena](https://athena.ohdsi.org/) bulk download — vocabulary content
carries its own license terms, so it can't be bundled with this repo.

1. Create a free account at [athena.ohdsi.org](https://athena.ohdsi.org/).
2. Under Search, select at least **ICD10CM** and **SNOMED** as vocabularies, then click
   Download. SNOMED is used locally only — to resolve curated-set matching (`find_curated_definition`)
   — and is never shipped or exposed through any tool surface.
3. Unzip the download into `data/athena/` (this directory is gitignored — nothing under `data/`
   is ever committed or shipped). It contains OMOP CDM vocabulary tables (`CONCEPT.csv`,
   `CONCEPT_RELATIONSHIP.csv`, etc.) — Athena ships vocabulary content pre-shaped as OMOP, so no
   separate mapping step is needed.

```bash
uv sync
python scripts/load_vocab.py data/athena --output data/vocab.duckdb
phenoforge-mcp  # stdio MCP server; add to Claude Desktop's mcpServers config to try it
```

### Optional: curated phenotype library

`find_curated_definition` needs a small local cache of OHDSI Phenotype Library cohort
definitions. Like the Athena download, this content carries its own terms — the OHDSI
PhenotypeLibrary GitHub repo has no confirmed LICENSE file (its R package `DESCRIPTION` claims
Apache, but that's a manifest claim, not a verified grant for the cohort content itself) — so
it's fetched into a gitignored local directory, never committed to this repo. Without this step,
`find_curated_definition` still runs and reports why nothing matched.

```bash
python scripts/fetch_phenotype_library.py  # writes data/phenotype_library/
```

Bundles 8 hand-picked, diabetes/kidney-relevant cohorts (not the full library): Type 2/Type 1/
gestational diabetes, diabetic ketoacidosis, retinopathy, and chronic kidney disease.

### Optional: dense (semantic) search

`search_concepts` fuses lexical and semantic matching when a dense index has been built;
without one, it falls back to lexical-only search automatically.

```bash
python scripts/build_index.py  # writes data/concept_index.lance; downloads BioLORD-2023 on first run
```

### Optional: eval harness

Scores each retrieval method (BM25, dense, hybrid, hierarchy expansion) against the curated
demo cohorts as ground truth — hierarchical distance-weighted scoring, set-level coverage, and
an over-inclusion penalty (partial credit for near-misses under the same hierarchy parent, not
exact-match recall). Requires the phenotype library fetch step above; dense/hybrid scoring also
needs the built index.

```bash
python scripts/run_eval.py                                 # bm25 + expand_descendants
python scripts/run_eval.py --index data/concept_index.lance # + dense + hybrid
```

### Optional: the agent

The LangGraph agent decomposes a plain-English population description into seed terms, checks
each against the curated phenotype library first, and pauses on the command line for you to
accept or reject any generated (unverified) candidates before they're included. Requires an
Anthropic API key (decomposition is a real Claude call) and the `agent` extra.
`scripts/run_agent.py` loads a local `.env` automatically (gitignored — never commit real keys),
or export the variable directly.

```bash
uv sync --extra agent
echo 'ANTHROPIC_API_KEY=...' > .env   # or: export ANTHROPIC_API_KEY=...
python scripts/run_agent.py "adults with type 2 diabetes and diabetic nephropathy"
python scripts/run_agent.py "..." --index data/concept_index.lance  # + dense retrieval for generated candidates
```

### Interactive exploration

Both notebooks are exploration only, never pushed to production — reusable logic stays in
`src/phenoforge/`.

- `notebooks/explore.ipynb` calls the engine directly (no MCP transport) against your real
  built `data/vocab.duckdb`.
- `notebooks/evaluate.ipynb` runs the eval harness and walks through the metrics with
  explanatory text, a method-comparison chart, and a sortable per-cohort results table.

```bash
uv sync --extra dev
jupyter lab notebooks/explore.ipynb
```

## What it does

Turns a plain-English patient population description into a defensible set of ICD-10-CM codes,
where every code carries provenance — whether it came from a peer-reviewed phenotype
definition, from hierarchy expansion, or from semantic retrieval that a human should check.

```
"adults with type 2 diabetes and diabetic nephropathy"
  → decomposes into seed clinical terms
  → checks OHDSI Phenotype Library for a validated definition for each
  → falls back to hybrid retrieval for terms with no curated match,
    pausing for human confirmation before including anything generated
  → returns a ConceptSet with per-code provenance and citations
```

Run it for real: `python scripts/run_agent.py "adults with type 2 diabetes and diabetic
nephropathy"` (see Setup).

## Why not an existing terminology server

Several good MCP terminology servers exist. They solve *lookup* — "what is code X", "map X to
Y". This solves *set assembly*, which is the actual task in cohort definition. It also covers
US ICD-10-CM, which the existing servers do not, and uses semantic retrieval rather than
proxied keyword search, which fails when a population description and a code description share
no vocabulary.

## Architecture

Three layers — a retrieval engine, a thin MCP server, and a LangGraph agent. The MCP server
and the agent are independent consumers of the same engine.

## Roadmap

- [x] **v0.1 — vocabulary layer.** Athena loader, DuckDB schema, hierarchy queries
- [x] **v0.2 — retrieval.** BM25, BioLORD-2023 embeddings, LanceDB index, RRF hybrid scoring
- [x] **v0.3 — expansion + provenance.** `ConceptSet` model, descendant expansion, and OHDSI PL
      curated matching (small hand-picked demo set — see Setup)
- [x] **v0.4 — MCP server.** stdio transport, four tools (see below), Claude Desktop config
- [x] **v0.5 — eval harness.** Distance-weighted scoring, coverage, over-inclusion penalty;
      curated demo cohorts as ground truth. Decomposition accuracy deferred to `v0.7` (see
      `phenoforge.eval`) — nothing decomposes a population description yet
- [x] **v0.7 — LangGraph agent.** Decompose → check curated first → generate only for
      unresolved terms → human confirmation gate on anything generated → assemble. Real
      interactive CLI (`scripts/run_agent.py`)
- [ ] **v1.0 — packaging.** PyPI, docs, validation and limitations section

Skipped: `v0.6` (encoder benchmark — BioLORD vs SapBERT vs MedCPT). The agent was the more
demonstrable deliverable, so `v0.7` was built first; the encoder benchmark may return later.

Deferred: literature-derived phenotype algorithms (`published` tier), RxNorm and LOINC domains

## Tools

| Tool | Status | Purpose |
|------|--------|---------|
| `lookup_concept` | done | Exact ICD-10-CM code → name and metadata |
| `expand_hierarchy` | done | Seed code → full descendant expansion (`generated` provenance) |
| `search_concepts` | done | Hybrid BM25 + dense search over ICD-10-CM names, RRF-fused (`generated` provenance); falls back to BM25-only if no dense index is built |
| `find_curated_definition` | done | Search the bundled OHDSI Phenotype Library demo set before generating anything (`curated` provenance) |
| `explain_inclusion` | planned | Why is this code in this set, via which path, with what evidence |

`find_curated_definition` is the only source of `curated` provenance today, and only for the 8
bundled demo cohorts. Everything from `expand_hierarchy` and `search_concepts` is `generated`
provenance — ungrounded, structural or lexical/semantic only, and meant to be confirmed by a
human before use in a cohort definition.

## License

Apache 2.0. Vocabulary content (ICD-10-CM, SNOMED) carries its own license terms from OHDSI/
Athena, separate from this repo's license — nothing from `data/` is redistributed.

## Not a clinical decision tool

This produces code sets for research and analytics. It does not make clinical determinations
and has not been validated for patient care.
