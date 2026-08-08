# phenoforge

> Semantic value set assembly for clinical cohort definitions, over MCP.

**Status: early prototype.** The vocabulary loader, hierarchy expansion, hybrid (BM25 + dense)
search, curated OHDSI Phenotype Library matching, and a thin MCP server exposing them all work
end-to-end against a real Athena download. The LangGraph agent and eval harness are not built
yet — see Roadmap.

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

### Interactive exploration

`notebooks/explore.ipynb` calls the engine directly (no MCP transport) against your real
built `data/vocab.duckdb` — exploration only, never pushed to production; reusable logic stays
in `src/phenoforge/`.

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
  → checks OHDSI Phenotype Library for a validated definition
  → expands seed concepts through the ICD-10-CM hierarchy
  → returns a ConceptSet with per-code provenance and citations
```

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
- [ ] **v0.5 — eval harness.** Ontology-aware metrics, curated sets as ground truth
- [ ] **v0.6 — encoder benchmark.** BioLORD vs SapBERT vs MedCPT vs general-purpose
- [ ] **v0.7 — LangGraph agent.** Tiered trust routing, HITL confirmation gate
- [ ] **v1.0 — packaging.** PyPI, docs, validation and limitations section

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
