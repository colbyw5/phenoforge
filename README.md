# phenoforge

[![CI](https://github.com/colbyw5/phenoforge/actions/workflows/ci.yml/badge.svg)](https://github.com/colbyw5/phenoforge/actions/workflows/ci.yml)
[![Docs](https://github.com/colbyw5/phenoforge/actions/workflows/docs.yml/badge.svg)](https://colbyw5.github.io/phenoforge/)

> Semantic value set assembly for clinical cohort definitions, over MCP.

**Status: early prototype.** The vocabulary loader, hierarchy expansion, hybrid (BM25 + dense)
search, curated OHDSI Phenotype Library matching, an eval harness scoring all of it against
curated ground truth, a thin MCP server exposing them all, and a LangGraph agent that decomposes
a population description, checks curated first, and pauses for human confirmation before
including anything generated — all work end-to-end against a real Athena download. See Roadmap.

## Try it now, no setup

```bash
git clone https://github.com/colbyw5/phenoforge && cd phenoforge
uv run python scripts/run_demo.py "diabetic nephropathy"   # or: make demo
```

Runs real hybrid search and hierarchy expansion against a small, bundled, redistributable
ICD-10-CM-only demo vocabulary (see `demo/README.md`) — no Athena account, no API key, no model
download. It only exercises the `generated` tier (BM25 + hierarchy expansion); the `curated`
tier, dense retrieval, and the full agent need the real Setup below.

## Setup

Published to PyPI as `py-phenoforge` (the bare `phenoforge` name is taken by an unrelated
package) — `pip install py-phenoforge` or `uv add py-phenoforge` gets you the library and the
`phenoforge-mcp` CLI, importable as `phenoforge`. That alone isn't runnable yet, though: every
tool needs the vocabulary database built below from your own Athena download, which can't be
redistributed on PyPI.

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

Or, once the Athena download is in place, `make setup` chains install + every optional data-build
step below (vocab, phenotype library, dense index) in one call. `make check` runs lint, type
checks, and the full test suite — the same checks CI runs on every push. See the `Makefile` for
the full target list.

<details>
<summary><strong>Optional: curated phenotype library</strong></summary>

`find_curated_definition` needs a small local cache of OHDSI Phenotype Library cohort
definitions. Like the Athena download, this content carries its own terms — the OHDSI
PhenotypeLibrary GitHub repo has no confirmed LICENSE file (its R package `DESCRIPTION` claims
Apache, but that's a manifest claim, not a verified grant for the cohort content itself) — so
it's fetched into a gitignored local directory, never committed to this repo. Without this step,
`find_curated_definition` still runs and reports why nothing matched.

```bash
python scripts/fetch_phenotype_library.py  # writes data/phenotype_library/
```

Bundles 23 hand-picked cohorts (not the full ~1,100-cohort library), spanning: diabetes (type
2/type 1/gestational, ketoacidosis, retinopathy), chronic kidney disease, cardiovascular (MI,
atrial fibrillation, heart failure), respiratory (pneumonia, asthma), mental health (depression,
anxiety), rheumatoid arthritis, neurological (epilepsy, migraine), sepsis, inflammatory bowel
disease, cirrhosis, hyperlipidemia, and obesity. See `scripts/fetch_phenotype_library.py` for
exact cohort ids and selection rationale.

</details>

<details>
<summary><strong>Optional: dense (semantic) search</strong></summary>

`search_concepts` fuses lexical and semantic matching when a dense index has been built;
without one, it falls back to lexical-only search automatically.

```bash
python scripts/build_index.py  # writes data/concept_index.lance; downloads BioLORD-2023 on first run
```

</details>

<details>
<summary><strong>Optional: eval harness</strong></summary>

Scores each retrieval method (BM25, dense, hybrid, hierarchy expansion) against the curated
demo cohorts as ground truth — hierarchical distance-weighted scoring, set-level coverage, and
an over-inclusion penalty (partial credit for near-misses under the same hierarchy parent, not
exact-match recall). Requires the phenotype library fetch step above; dense/hybrid scoring also
needs the built index.

```bash
python scripts/run_eval.py                                 # bm25 + expand_descendants
python scripts/run_eval.py --index data/concept_index.lance # + dense + hybrid
```

Separately, `scripts/eval_decomposition.py` scores the agent's `decompose` step itself (needs
`ANTHROPIC_API_KEY` and the `agent` extra — see Validation and limitations below):

```bash
python scripts/eval_decomposition.py --index data/concept_index.lance
```

</details>

<details>
<summary><strong>Optional: the agent</strong></summary>

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

</details>

<details>
<summary><strong>Optional: API docs</strong></summary>

Hosted at [colbyw5.github.io/phenoforge](https://colbyw5.github.io/phenoforge/), rebuilt on
every push to `main`. To build locally instead:

```bash
make docs  # writes docs/_build/index.html
```

</details>

<details>
<summary><strong>Optional: interactive exploration (notebooks)</strong></summary>

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

</details>

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

<details>
<summary><strong>Roadmap</strong></summary>

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
- [x] **v1.0 — packaging.** One-command setup (`make setup`), CI (lint/typecheck/test on every
      push), a PyPI release pipeline (tag-triggered, trusted publishing) as `py-phenoforge`, a
      zero-setup demo (`make demo`, no account/key/download needed), hosted API docs
      ([colbyw5.github.io/phenoforge](https://colbyw5.github.io/phenoforge/), built from the
      same docstrings every function already carries), and a validation and limitations
      section (see below)

Skipped: `v0.6` (encoder benchmark — BioLORD vs SapBERT vs MedCPT). The agent was the more
demonstrable deliverable, so `v0.7` was built first; the encoder benchmark may return later.

Deferred: literature-derived phenotype algorithms (`published` tier), RxNorm and LOINC domains

Watching: [TypeSafe AI's Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev), a
schema-constrained "System One Model" that could fit the `decompose` node's structured-output
call. Early access/waitlist, proprietary, no independent benchmarks yet as of 2026-10 — not
adopted; revisit if it leaves early access and the latency/cost tradeoff is verified
independently.

</details>

<details>
<summary><strong>Tools</strong></summary>

| Tool | Status | Purpose |
|------|--------|---------|
| `lookup_concept` | done | Exact ICD-10-CM code → name and metadata |
| `expand_hierarchy` | done | Seed code → full descendant expansion (`generated` provenance) |
| `search_concepts` | done | Hybrid BM25 + dense search over ICD-10-CM names, RRF-fused (`generated` provenance); falls back to BM25-only if no dense index is built |
| `find_curated_definition` | done | Search the bundled OHDSI Phenotype Library demo set before generating anything (`curated` provenance) |
| `explain_inclusion` | done | Why a code would be included, via which path(s), with what evidence — re-derives rather than reading back state, since nothing persists server-side |

`find_curated_definition` is the only source of `curated` provenance today, and only for the 23
bundled demo cohorts. Everything from `expand_hierarchy` and `search_concepts` is `generated`
provenance — ungrounded, structural or lexical/semantic only, and meant to be confirmed by a
human before use in a cohort definition.

</details>

<details>
<summary><strong>Validation and limitations</strong></summary>

The only quantitative evaluation that exists is `scripts/run_eval.py` scoring each retrieval
method against the 23 bundled OHDSI Phenotype Library demo cohorts (spanning diabetes/kidney,
cardiovascular, respiratory, mental health, and more — see Setup) as ground truth. Current
numbers on that set:

| method | coverage | over-inclusion | hierarchical |
|---|---|---|---|
| `bm25` | 0.759 | 0.441 | 0.589 |
| `expand_descendants` | 0.635 | 0.122 | 0.667 |
| `dense` | 0.828 | 0.541 | 0.577 |
| `hybrid` (0.5/0.5) | 0.873 | 0.436 | 0.654 |

(Earlier, measured against only the original 8 diabetes/kidney cohorts: `bm25` 0.535,
`expand_descendants` 0.697, `dense` 0.586, `hybrid` 0.692 hierarchical. Every method's
hierarchical score moved when the set broadened — `hybrid` down from 0.692 to 0.634 (since
corrected to 0.654, see below), `expand_descendants` down from 0.697 to 0.667 — concrete
evidence that the original 8-cohort numbers were narrower than they looked, not a general
ICD-10-CM retrieval benchmark.)

`hybrid`'s own numbers above reflect a later fix: `hybrid_search`'s candidate pool (how many
results each of BM25/dense considers *before* RRF fusion) used to default to `k` itself — the
same number controlling how many fused results come back — so a concept ranked just outside
`k` in one sub-retriever was invisible to fusion even when it would have won easily once
pooled. Found via cohort `775` (below); fixed by decoupling the default (`candidate_k` now
defaults to `5 * k`, still overridable). Measured before/after on this same benchmark: coverage
unchanged (`0.873`), over-inclusion down `0.480` → `0.436`, hierarchical score up `0.634` →
`0.654` — a real, measured improvement, not just a plausible-sounding default.

What this does and doesn't establish:

- **23 cohorts across several disease areas — still not a general benchmark.** Broadening past
  diabetes/kidney measurably moved every method's score (above), which is itself the point: a
  benchmark this size is sensitive to exactly which conditions are in it. It says nothing about
  performance on conditions still absent from the set, rare diseases, or codes with sparse/
  ambiguous natural-language descriptions.
- **Over-inclusion is the dominant error mode, not coverage.** Every method pulls in more
  codes than the curated ground truth (dense and hybrid worst: ~0.48-0.54 over-inclusion
  against ~0.83-0.87 coverage) — the system errs toward casting a wide net and relying on human
  review to reject false positives, not toward silently missing codes. Anyone using the
  `generated` tier should expect to reject a meaningful fraction of what it returns, not
  rubber-stamp it.
- **Decomposition now has a test case for every bundled cohort (23/23), measured at the
  same search depth the real pipeline actually uses, and on average it helps rather than
  hurts — but several real, distinct failure modes showed up along the way.**
  `scripts/eval_decomposition.py` (`phenoforge.eval.decomposition`) compares, per bundled
  cohort, "hybrid search using the cohort's own name directly" against "the real `decompose`
  step, then per-term resolution" — same ground truth, isolating what decomposition costs or
  saves (phrasing drawn from real ClinicalTrials.gov titles/inclusion criteria, not this
  project's own wording). An internal audit caught this eval defaulting to `k=10` (previously
  `k=25`) — the earlier number measured a wider search than `agent/nodes.py` ever actually
  uses, silently inflating how good decomposition looked. At the corrected depth: mean gap
  across the original 8 diabetes/kidney cases was a near-wash (`+0.019`, `+0.020` on rerun at
  the old depth); across all 23 at the corrected depth, it's **-0.125 to -0.165** across reruns
  — decomposition still usually *wins*, because a single clean condition name (e.g. "atrial
  fibrillation") curated-matches better than a cohort's own longer, more clinical display name
  (e.g. "Diabetes Mellitus Type 2 or history of diabetes") did as a search query. (The mean gap
  itself isn't perfectly reproducible run-to-run: `decompose_fn` is a real, non-deterministic
  Claude call, so a handful of individual cases' seed terms — and therefore scores — shift
  between reruns even with no code change at all; the *themes* below hold across reruns, the
  exact per-case numbers don't.) Individual cases still range widely, and the depth correction
  itself surfaced a new one:
    - **Near-duplicate cohorts cause `find_curated_definition` to deliberately refuse to
      guess.** The library has three "type 2 diabetes" cohorts (`40`, `288`, `503`); a generic
      decomposed term matching more than one of them falls through to ungrounded `generated`
      retrieval instead of picking one (by design — see the function's docstring), losing the
      specificity a longer query carried. Worst case seen: `+0.277` to `+0.404` across reruns
      (cohort `288`).
    - **A cohort's display name can be misleading relative to what it actually contains.**
      Cohort `602`, bundled under the name "Anxiety," resolves — via its real SNOMED-to-
      ICD-10-CM mapping — exclusively to *dementia-with-anxiety* codes (`F01.x4`/`F02.x4`/
      `F03.x4`), not primary anxiety disorders. The decomposed term `"generalized anxiety
      disorder"` correctly retrieves the real F41.1 code and its relatives — retrieval works
      correctly — but scores `0.000` against ground truth that doesn't actually contain it
      (and at the corrected depth, so does the baseline — both score `0.000` against this
      cohort now). A curated-library content-quality issue, not a decomposition or retrieval
      bug, and it likely already depressed `602`'s contribution to the retrieval-only numbers
      above before this was ever noticed.
    - **Search depth itself changes which cases look solved — and widening the candidate pool
      narrows this but doesn't close it.** Cohort `775` (inflammatory bowel disease) scored a
      perfect `1.000` at the old, wider `k=25` depth but `0.000` at the corrected `k=10`.
      Investigated directly: the seed term "inflammatory bowel disease" has *zero* lexical
      overlap with any ICD-10-CM concept name at all (the vocabulary only names these codes
      "Crohn's disease" / "ulcerative colitis") — BM25 contributes nothing, and dense
      retrieval alone ranks the correct code (`K50.9`) only 5th of 50 candidates, short of
      cracking the fused top 10 even after `hybrid_search`'s `candidate_k` default was widened
      from `k` to `5 * k` (above). Widening the pool measurably helped the *baseline* query for
      this cohort (`0.177` → `0.402`, since the full cohort name has more lexical signal than
      the bare term) but the decomposed path stays at `0.000` — a genuine vocabulary/
      terminology-coverage gap (umbrella clinical term vs. disease-specific vocabulary naming),
      not a bug, and not something a pooling default alone can fix. Left as a documented
      limitation rather than pursued further (e.g. query synonym expansion), since that's new
      scope with its own failure modes, not justified by one cohort.
  (Checked separately: none of the 23 bundled cohorts currently have a ground-truth code
  missing from the loaded vocabulary — `unresolvable_ground_truth` exists for exactly this
  case but had never actually been populated until now; see `eval/harness.py`.)
- **Curated-tier coverage is still narrow by construction.** 23 cohorts out of the library's
  ~1,100 are bundled; any population description outside them falls through entirely to the
  unverified `generated` tier. Separately, SNOMED-to-ICD-10-CM resolution deliberately drops
  concepts whose fan-out exceeds a configurable threshold (`find_high_fanout_snomed_concepts`)
  rather than including everything — a real cohort definition can lose codes this way.
- **No scale or concurrency testing.** Built and measured against one developer's own Athena
  download on one machine. No data on index build time, query latency, or memory at a larger
  vocabulary, or under concurrent MCP clients.
- **English, US ICD-10-CM only.** No other language, no RxNorm/LOINC/other domains (see Roadmap).
- **The human-confirmation gate has no audit trail.** `confirm` pauses for review, but nothing
  records who approved what or persists decisions across runs — each CLI invocation starts a
  fresh thread with no memory of past confirmations.

</details>

<details>
<summary><strong>Releasing</strong></summary>

`.github/workflows/publish.yml` builds and publishes to PyPI on any `v*.*.*` tag push, via
[PyPI trusted publishing](https://docs.pypi.org/trusted-publishers/) (OIDC — no stored API
token). One-time setup, done outside this repo:

1. On PyPI: Account settings → Publishing → add a pending publisher for `py-phenoforge`,
   repository `colbyw5/phenoforge`, workflow `publish.yml`, environment `pypi`.
2. On GitHub: Settings → Environments → create an environment named `pypi` (optionally with
   required reviewers, for a manual approval gate before each publish).

To cut a release: bump `version` in `pyproject.toml`, commit, then `git tag v0.1.0 && git push
origin v0.1.0`.

`.github/workflows/docs.yml` builds and deploys the API docs to GitHub Pages on every push to
`main`. One-time setup: on GitHub, Settings → Pages → set "Source" to "GitHub Actions" (no
separate environment to create — `deploy-pages` manages the `github-pages` environment itself).

</details>

## License

Apache 2.0. Vocabulary content (ICD-10-CM, SNOMED) carries its own license terms from OHDSI/
Athena, separate from this repo's license — nothing from `data/` is redistributed.

## Not a clinical decision tool

This produces code sets for research and analytics. It does not make clinical determinations
and has not been validated for patient care.
