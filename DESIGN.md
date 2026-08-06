# Design

## Problem

Building a patient cohort definition means turning a plain-English population description
into a defensible set of clinical codes. Today that is manual work in ATLAS or a spreadsheet,
and the resulting code sets rarely carry provenance — you cannot tell whether a code is there
because a validated phenotype algorithm included it, because it is a descendant of a seed
concept, or because someone guessed.

Existing MCP terminology servers (`medical-terminologies-mcp`, OMOPHub, FHIRfly) solve
*lookup*: "what is code X", "map X to Y". None solve *set assembly*: "give me the complete,
defensible code set for this population, and tell me where each code came from."

Two further gaps in the existing tools:

- **No US ICD-10-CM.** They cover ICD-11 (WHO) and, in one case, Brazilian CID-10. ICD-10-CM
  is the currency of US claims and phenotype libraries and is absent.
- **Text search only.** They proxy NLM/WHO relevance ranking, which fails on paraphrase.
  A population description and a code description rarely share vocabulary.

## Architecture

Three layers. The engine is the contribution; MCP and the agent are two independent consumers
of it.

```
┌─────────────────────┐   ┌──────────────────────┐
│  MCP server (2a)    │   │  LangGraph agent(2b) │
│  stdio, thin        │   │  tiered trust flow   │
└──────────┬──────────┘   └──────────┬───────────┘
           │                         │
           └───────────┬─────────────┘
                       ▼
        ┌──────────────────────────────┐
        │  Retrieval & expansion engine │  ← the actual contribution
        │  (1) semantic + BM25 + graph  │
        │      hierarchy expansion      │
        │      curated set matching     │
        │      provenance tracking      │
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐
        │  DuckDB (vocabulary)          │
        │  LanceDB (embeddings)         │
        └──────────────────────────────┘
```

**Why MCP rather than plain functions.** Two independent consumers, one of which is not ours
(Claude Desktop, Cursor, a colleague's agent). Independent deploy and versioning; tool
discovery without per-client integration code. If the agent were the only consumer, direct
function calls would be correct and MCP would be ceremony.

**Why the agent goes through MCP rather than calling the engine directly.** Dogfooding. If the
tool schemas are awkward for a model to use, we find out immediately. Cost is latency and
indirection, acceptable for a research tool. This would be the wrong choice for a GxP batch
pipeline, where audit trail and determinism argue for direct calls.

## Tiered provenance

Every concept in an output set carries a provenance tier. This is the structural
anti-hallucination commitment and the reason the workflow is agentic rather than linear.

| Tier | Source | Behaviour |
|------|--------|-----------|
| `curated` | OHDSI Phenotype Library | Use as-is, cite cohort ID + PheValuator operating characteristics |
| `published` | Validated algorithm from literature | Retrieve, extract code list, cite. **Deferred to v2.** |
| `generated` | Semantic retrieval + hierarchy expansion | Label clearly as ungrounded, require human confirmation |

Routing between tiers is what justifies LangGraph over a linear chain: check curated, branch on
miss, escalate on low confidence, interrupt for human confirmation.

## Decisions

### D1 — Vocabulary source: OHDSI Athena

Athena bulk download. Load `CONCEPT` and `CONCEPT_RELATIONSHIP`. **Do not load
`CONCEPT_ANCESTOR`** — it does not cover ICD-10-CM (see findings).

*Rationale:* NLM APIs are rate-limited (5–20 req/s) and cannot support bulk expansion or
embedding index construction. Athena is also the only path to OHDSI Phenotype Library
resolution (see D4).

**Findings — resolved against the actual download (v2026 vocabularies):**

| Query | Result | Consequence |
|-------|--------|-------------|
| ICD-10-CM rows in `CONCEPT_ANCESTOR` | **0** | Precomputed closure is unavailable for this vocabulary. Do not load the table. |
| ICD-10-CM ↔ ICD-10-CM in `CONCEPT_RELATIONSHIP` | **290,572 `Is a`** (and inverse `Subsumes`) | A full internal hierarchy exists as direct parent/child edges. |
| ICD-10-CM concepts reachable from SNOMED via `Mapped from` | **98,059** (~complete) | OHDSI PL → SNOMED → ICD-10-CM resolution is viable. D4 holds. |

**Expansion mechanism:** recursive CTE over `Is a` edges within `vocabulary_id='ICD10CM'`, not a
join against `CONCEPT_ANCESTOR`, and not prefix matching on the code string. This uses OMOP's
actual modeled hierarchy rather than string structure, which is the stronger choice. 290k edges
is trivial in DuckDB.

*Traversal caution:* `Is a` and `Subsumes` are inverse relations (identical counts). Traverse a
single direction — descendants follow `Subsumes` from the parent (equivalently, `Is a`
reversed). The recursive CTE requires a visited-set guard to prevent cycles regardless.

The prefix-matching fallback considered earlier is unnecessary and is dropped.

*Verified traversal (E11.21):* `Is a` from `E11.21` returns `E11.2` (Type 2 diabetes with
kidney complications) and `E11` (Type 2 diabetes mellitus) — a correct, semantically ordered
parent chain. This is the expected shape; a loader test should assert it.

### D2 — Storage: embedded, no server

DuckDB for vocabulary tables, LanceDB for the embedding index.

*Rationale:* `pip install` and go. Adoption dies at "first, install Postgres." ~70k ICD-10-CM
concepts does not need a hosted vector database; reaching for one would be a negative signal.

### D3 — Vocabulary scope: ICD-10-CM only for v1

*Rationale:* the currency of US claims data and phenotype libraries. Also legally simplest —
ICD-10-CM is US government content, unlike SNOMED. This keeps a future hosted endpoint viable
without licensing complications.

**Schema must carry a `domain` field from day one** (`condition | measurement | drug |
procedure`) so RxNorm and LOINC are additive rather than a rewrite. Many real phenotypes are
unreachable with conditions alone — "poorly controlled diabetes" is a LOINC code plus a
threshold; "on insulin" is RxNorm.

Reserve schema space for value constraints and temporal windows. Do not implement them.

### D4 — Curated source: OHDSI Phenotype Library primary, VSAC optional

*Rationale:* OHDSI PL is publicly accessible on GitHub, version-controlled with a DOI per
version, OMOP CDM conformant, and openly peer-reviewed. Its metadata records literature review
and PheValuator operating characteristics, which feeds `explain_inclusion` directly. Because
definitions reference OMOP concept IDs, they join against Athena.

**Cost, accepted:** OHDSI PL definitions are written against *standard* concepts, which for
conditions means SNOMED. Resolving them back to ICD-10-CM requires the SNOMED↔ICD-10-CM
mapping, which means Athena stays on the critical path and users must bring their own download.
This is standard practice in OHDSI tooling. It costs the `pip install`-and-go adoption story;
say so plainly in the README rather than papering over it.

VSAC is the authoritative source for CMS eCQM value sets but requires a UMLS license and API
key, and all its code systems fall under the UMLS Metathesaurus License Agreement. It therefore
**cannot be bundled** — it becomes an optional user-supplied-key path, which usefully
demonstrates auth handling without licensing risk.

*Caveats:* OHDSI PL entries are full cohort definitions including temporal logic; extract the
concept-set portion rather than using them wholesale. Coverage is hundreds of phenotypes, not
VSAC's thousands — acceptable, and arguably better as eval ground truth. **Verify the LICENSE
file before redistributing anything.**

**Resolution mechanics (verified against v2026):** SNOMED → ICD-10-CM via `Mapped from` fans
out one-to-many — min 1, median 2, **max 3,029**. Two consequences:

- The resolved mapping is many-to-many. `ConceptSet` holds a list of ICD-10-CM codes per source
  concept, never a scalar. Model the mapping as a join table.
- **Fan-out guard, required.** The 3,029 tail is structural, not clinical — a few high-level
  SNOMED groupers map to thousands of ICD-10-CM codes across unrelated chapters. Unguarded
  resolution of such a concept silently destroys value-set specificity. Cap or flag any single
  SNOMED concept resolving to more than ~50–100 ICD-10-CM codes (tunable); surface it for review
  rather than expanding blindly. This is the same instinct as the over-inclusion penalty and the
  CCSR-spread heuristic (D10), applied at the resolution stage — a post-resolution set spanning
  many unrelated CCSR categories is the symptom this prevents.

#### Rejected: AHRQ CCSR as curated ground truth

CCSR was evaluated as a fully redistributable, ICD-10-CM-native, SNOMED-free alternative. It
does not work, and the reason is instructive.

Profiling `DXCCSR_v2026-1` (75,725 codes → 496 default inpatient categories, median 33 codes
per category):

- All 95 `E11*` codes collapse into two categories: `END003` (with complication, 93 codes) and
  `END002` (without complication, 2 codes)
- `END003` also contains `E08`, `E09`, `E10`, and `E13` — **type 1 and type 2 diabetes are
  indistinguishable**
- All three diabetic nephropathy codes land in `END003` alongside every other complication

The motivating example on the README front page — "adults with type 2 diabetes and diabetic
nephropathy" — is not representable. Neither is "type 2 but not type 1."

The largest categories reveal the design intent: `XXX000 Unacceptable PDX` (10,843 codes),
`INJ073 Injury, sequela` (7,654). CCSR is a **reporting grouper** for summarizing hospital
discharges, not a phenotype library. Both produce "groups of ICD-10-CM codes"; only one encodes
clinical intent.

CCSR is still loaded, as a feature rather than ground truth — see D10.

Tuva value sets remain a candidate supplement pending per-value-set license review. Their
terminology sets derive from NCHS/CDC and are public domain, but the **quality measure value
sets draw on CPT (AMA-licensed) and SNOMED**, so redistributability must be checked per set,
not wholesale. Avoid APR-DRG entirely (Solventum/3M licensed). CMS Chronic Conditions Data
Warehouse (75 conditions, ICD-10-CM native, public domain) is the most promising bundled
supplement.

### D5 — Transport: stdio for v1, flag-controlled

*Rationale:* stdio works in Claude Desktop today with no hosting, auth, or CORS. Keep business
logic out of the transport layer so HTTP is a flag, not a rewrite.

*What would trigger HTTP (v2):* the Athena load plus embedding index is a multi-GB, multi-minute
setup step per user. That is a real adoption barrier, and the strongest argument for a hosted
instance — index once, serve many. D3 (ICD-10-CM only) is what keeps that legally viable.

### D6 — Repo layout: monorepo

Packages: `engine/`, `mcp/`, `agent/`, `eval/`.

*Rationale:* splitting later is easy; un-fragmenting a half-finished story is not. Revisit if
the eval harness gets independent traction.

### D7 — Embedding model: BioLORD-2023 default, swappable

*Rationale:* biomedical concept encoder, CC-BY, available on HuggingFace. SapBERT and MedCPT
are benchmarking alternatives — the encoder comparison is a deliverable, so the interface must
allow swapping.

### D8 — Retrieval: hybrid, three components

Dense (paraphrase), BM25 (exact strings, code fragments, rare literal terms), graph traversal
(hierarchy expansion). Each covers a distinct failure mode; publish the ablation.

### D9 — SNOMED: use locally, never redistribute

An earlier version of this document said "SNOMED is out of scope," which conflated two
different things and blocked a legitimate design.

- **Use** — loading SNOMED locally from a user's own Athena download, to resolve OHDSI PL
  concept IDs or build mappings. Permitted under the terms accepted at download.
- **Redistribution** — shipping SNOMED content in a package, or serving it from a public
  endpoint. Not permitted.

Operating rule: **load everything locally, expose and ship only ICD-10-CM.** SNOMED is never a
user-facing vocabulary, never a tool surface, never in a released artifact.

*Open question:* whether an embedding index or mapping table *derived* from SNOMED
relationships constitutes a derivative work. Unresolved — do not build anything that depends
on the answer.

### D10 — CCSR: retrieval and validation feature, not ground truth

Rejected as curated ground truth (see D4). Loaded anyway, because it provides a clinical
grouping over ICD-10-CM that the billing hierarchy does not, and it is fully redistributable.

Three uses:

- **Reranking prior.** Semantic search returns candidates; shared CCSR category is a cheap
  clinical signal for reranking.
- **Over-inclusion detector.** A generated set spanning many unrelated CCSR categories is
  almost certainly wrong. Feeds the over-inclusion penalty in the evaluation layer.
- **Coarse graph layer.** Gives the graph component of hybrid retrieval a clinical grouping
  that is orthogonal to prefix structure, which is exactly why it adds signal.

*Parsing gotchas for the loader:*

- **Mixed quoting.** Codes are single-quoted (`'A000'`), descriptions double-quoted. Standard
  CSV readers with `ignore_errors` silently drop ~86% of rows. Parse with `quotechar='"'`,
  then strip single quotes per field. Verify the row count is ~75,725, not ~10,854.
- **Many-to-many.** 8,958 codes carry more than one CCSR category. Model it as a join table,
  not a column on the concept.

## Evaluation

Standard recall@k is wrong for this task. Retrieving `E11.21` when the target is `E11.9` is a
near-miss under the same parent; retrieving a circulatory-chapter code is a total miss. Both
score zero today.

Metrics to implement:

- Hierarchical distance-weighted scoring (partial credit by tree distance)
- Set-level coverage (did the whole value set get assembled)
- Over-inclusion penalty (cohort definitions fail on false positives too)
- Decomposition accuracy scored separately from retrieval accuracy

**Ground truth comes free.** Curated value sets *are* the labels. No hand-annotation required —
which is usually what kills solo evaluation projects.

## Explicitly out of scope

- SNOMED CT as a *user-facing vocabulary or shipped artifact* (see D9 — local use is permitted
  and required for D4)
- General PubMed search (exists, undifferentiated, does not improve value sets)
- Multimodal / imaging
- Temporal cohort logic in v1 (schema space reserved only)
- Competing on breadth of vocabulary lookup — that space is served

## Open questions

- Does `CONCEPT_ANCESTOR` cover ICD-10-CM, or is prefix matching the expansion mechanism? (D1)
- What fraction of OHDSI PL concept sets resolve cleanly back to ICD-10-CM via `Mapped from`?
  If coverage is poor, D4 needs rethinking again.
- Exact extraction path for concept sets from OHDSI PL cohort JSON
- Whether a derived index built from SNOMED relationships is a derivative work (D9)
- Which Tuva value sets are free of CPT and SNOMED and therefore bundleable
- Confidence threshold for escalating a `generated` set to human confirmation