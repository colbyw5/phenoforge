# phenoforge

> Semantic value set assembly for clinical cohort definitions, over MCP.

**Status: early prototype.** The vocabulary loader, hierarchy expansion, BM25 lexical search, and
a thin MCP server exposing them all work end-to-end against a real Athena download. Dense
(semantic) retrieval, curated OHDSI Phenotype Library matching, and the LangGraph agent are not
built yet — see Roadmap.

## Setup

Requires your own [OHDSI Athena](https://athena.ohdsi.org/) bulk download (free account,
unzipped locally) — this cannot be bundled. See DESIGN.md D4 for why.

```bash
uv sync
python scripts/load_vocab.py path/to/unzipped/athena --output data/vocab.duckdb
phenoforge-mcp  # stdio MCP server; add to Claude Desktop's mcpServers config to try it
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

See [DESIGN.md](./DESIGN.md) for the full rationale.

## Architecture

Three layers — a retrieval engine, a thin MCP server, and a LangGraph agent. The MCP server
and the agent are independent consumers of the same engine.

## Roadmap

- [x] **v0.1 — vocabulary layer.** Athena loader, DuckDB schema, hierarchy queries
- [ ] **v0.2 — retrieval.** ~~BM25~~ done; BioLORD-2023 embeddings, LanceDB index, hybrid scoring
- [ ] **v0.3 — expansion + provenance.** `ConceptSet` model and descendant expansion done;
      OHDSI PL curated matching not started
- [x] **v0.4 — MCP server.** stdio transport, three tools (see below), Claude Desktop config
- [ ] **v0.5 — eval harness.** Ontology-aware metrics, curated sets as ground truth
- [ ] **v0.6 — encoder benchmark.** BioLORD vs SapBERT vs MedCPT vs general-purpose
- [ ] **v0.7 — LangGraph agent.** Tiered trust routing, HITL confirmation gate
- [ ] **v1.0 — packaging.** PyPI, docs, validation and limitations section

Deferred: literature-derived phenotype algorithms (`published` tier), RxNorm and LOINC domains,
temporal cohort logic, hosted HTTP transport.

## Tools

| Tool | Status | Purpose |
|------|--------|---------|
| `lookup_concept` | done | Exact ICD-10-CM code → name and metadata |
| `expand_hierarchy` | done | Seed code → full descendant expansion (`generated` provenance) |
| `search_concepts` | done (BM25 only) | Lexical search over ICD-10-CM names (`generated` provenance) |
| `find_curated_definition` | planned | Search OHDSI Phenotype Library before generating anything |
| `explain_inclusion` | planned | Why is this code in this set, via which path, with what evidence |

Every result from `expand_hierarchy` and `search_concepts` is tagged `generated` provenance —
ungrounded, structural or lexical only, and meant to be confirmed by a human before use in a
cohort definition. Nothing produces `curated` provenance yet.

## License

Apache 2.0. Vocabulary content carries its own terms — see `DESIGN.md`.

## Not a clinical decision tool

This produces code sets for research and analytics. It does not make clinical determinations
and has not been validated for patient care.
