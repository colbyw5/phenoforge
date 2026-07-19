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

Athena bulk download (`CONCEPT`, `CONCEPT_ANCESTOR`, `CONCEPT_RELATIONSHIP`).

*Rationale:* the only option giving a real transitive hierarchy offline. `CONCEPT_ANCESTOR` is
precomputed transitive closure — descendant expansion becomes a single query rather than a
recursive walk. NLM APIs are rate-limited (5–20 req/s) and cannot support bulk expansion or
embedding index construction.

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
definitions reference OMOP concept IDs, they join against Athena with no mapping layer.

VSAC is the authoritative source for CMS eCQM value sets but requires a UMLS license and API
key, and all its code systems fall under the UMLS Metathesaurus License Agreement. It therefore
**cannot be bundled** — it becomes an optional user-supplied-key path, which usefully
demonstrates auth handling without licensing risk.

Tuva value sets are a candidate second bundled source pending a license check on seed artifacts.

*Caveats:* OHDSI PL entries are full cohort definitions including temporal logic; extract the
concept-set portion rather than using them wholesale. Coverage is hundreds of phenotypes, not
VSAC's thousands — acceptable, and arguably better as eval ground truth. **Verify the LICENSE
file before redistributing anything.**

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

- SNOMED CT (licensing; requires self-hosted Snowstorm)
- General PubMed search (exists, undifferentiated, does not improve value sets)
- Multimodal / imaging
- Temporal cohort logic in v1 (schema space reserved only)
- Competing on breadth of vocabulary lookup — that space is served

## Open questions

- Exact extraction path for concept sets from OHDSI PL cohort JSON
- Whether Tuva seed artifacts are redistributable
- Confidence threshold for escalating a `generated` set to human confirmation
