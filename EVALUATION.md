# Evaluation

Full methodology, numbers, and investigation notes for every quantitative claim about
phenoforge. The README's own "Validation and limitations" section is a short summary of this
file — read here for the real analysis, caveats, and dead ends.

Everything below is measured against the 23 bundled OHDSI Phenotype Library demo cohorts (see
the main README's Setup section) as ground truth, via `scripts/run_eval.py` /
`scripts/eval_decomposition.py` / `scripts/compare_encoders.py` and the shared harness in
`phenoforge.eval`.

## Retrieval method benchmark

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
same number controlling how many fused results come back — so a concept ranked just outside `k`
in one sub-retriever was invisible to fusion even when it would have won easily once pooled.
Found via cohort `775` (below); fixed by decoupling the default (`candidate_k` now defaults to
`5 * k`, still overridable). Measured before/after on this same benchmark: coverage unchanged
(`0.873`), over-inclusion down `0.480` → `0.436`, hierarchical score up `0.634` → `0.654` — a
real, measured improvement, not just a plausible-sounding default.

## What this does and doesn't establish

- **23 cohorts across several disease areas — still not a general benchmark.** Broadening past
  diabetes/kidney measurably moved every method's score (above), which is itself the point: a
  benchmark this size is sensitive to exactly which conditions are in it. It says nothing about
  performance on conditions still absent from the set, rare diseases, or codes with sparse/
  ambiguous natural-language descriptions.
- **Over-inclusion is the dominant error mode, not coverage.** Every method pulls in more codes
  than the curated ground truth (dense and hybrid worst: ~0.48-0.54 over-inclusion against
  ~0.83-0.87 coverage) — the system errs toward casting a wide net and relying on human review
  to reject false positives, not toward silently missing codes. Anyone using the `generated`
  tier should expect to reject a meaningful fraction of what it returns, not rubber-stamp it.
- **Curated-tier coverage is still narrow by construction.** 23 cohorts out of the library's
  ~1,100 are bundled; any population description outside them falls through entirely to the
  unverified `generated` tier. Separately, SNOMED-to-ICD-10-CM resolution deliberately drops
  concepts whose fan-out exceeds a configurable threshold (`find_high_fanout_snomed_concepts`)
  rather than including everything — a real cohort definition can lose codes this way.
- **No scale or concurrency testing.** Built and measured against one developer's own Athena
  download on one machine. No data on index build time, query latency, or memory at a larger
  vocabulary, or under concurrent MCP clients.
- **English, US ICD-10-CM only.** No other language, no RxNorm/LOINC/other domains.
- **The human-confirmation gate has no audit trail.** `confirm` pauses for review, but nothing
  records who approved what or persists decisions across runs — each CLI invocation starts a
  fresh thread with no memory of past confirmations.

## Decomposition eval

Decomposition now has a test case for every bundled cohort (23/23), measured at the same search
depth the real pipeline actually uses, and on average it helps rather than hurts — but several
real, distinct failure modes showed up along the way.

`scripts/eval_decomposition.py` (`phenoforge.eval.decomposition`) compares, per bundled cohort,
"hybrid search using the cohort's own name directly" against "the real `decompose` step, then
per-term resolution" — same ground truth, isolating what decomposition costs or saves (phrasing
drawn from real ClinicalTrials.gov titles/inclusion criteria, not this project's own wording).

An internal audit caught this eval defaulting to `k=10` (previously `k=25`) — the earlier number
measured a wider search than `agent/nodes.py` ever actually uses, silently inflating how good
decomposition looked. At the corrected depth: mean gap across the original 8 diabetes/kidney
cases was a near-wash (`+0.019`, `+0.020` on rerun at the old depth); across all 23 at the
corrected depth, it's **-0.125 to -0.165** across reruns — decomposition still usually *wins*,
because a single clean condition name (e.g. "atrial fibrillation") curated-matches better than a
cohort's own longer, more clinical display name (e.g. "Diabetes Mellitus Type 2 or history of
diabetes") did as a search query.

(The mean gap itself isn't perfectly reproducible run-to-run: `decompose_fn` is a real,
non-deterministic Claude call, so a handful of individual cases' seed terms — and therefore
scores — shift between reruns even with no code change at all; the *themes* below hold across
reruns, the exact per-case numbers don't.)

Individual cases still range widely, and the depth correction itself surfaced a new one:

- **Near-duplicate cohorts cause `find_curated_definition` to deliberately refuse to guess.**
  The library has three "type 2 diabetes" cohorts (`40`, `288`, `503`); a generic decomposed
  term matching more than one of them falls through to ungrounded `generated` retrieval instead
  of picking one (by design — see the function's docstring), losing the specificity a longer
  query carried. Worst case seen: `+0.277` to `+0.404` across reruns (cohort `288`).
- **A cohort's display name can be misleading relative to what it actually contains.** Cohort
  `602`, bundled under the name "Anxiety," resolves — via its real SNOMED-to-ICD-10-CM mapping —
  exclusively to *dementia-with-anxiety* codes (`F01.x4`/`F02.x4`/`F03.x4`), not primary anxiety
  disorders. The decomposed term `"generalized anxiety disorder"` correctly retrieves the real
  F41.1 code and its relatives — retrieval works correctly — but scores `0.000` against ground
  truth that doesn't actually contain it (and at the corrected depth, so does the baseline — both
  score `0.000` against this cohort now). A curated-library content-quality issue, not a
  decomposition or retrieval bug, and it likely already depressed `602`'s contribution to the
  retrieval-only numbers above before this was ever noticed.
- **Search depth itself changes which cases look solved — and widening the candidate pool
  narrows this but doesn't close it.** Cohort `775` (inflammatory bowel disease) scored a perfect
  `1.000` at the old, wider `k=25` depth but `0.000` at the corrected `k=10`. Investigated
  directly: the seed term "inflammatory bowel disease" has *zero* lexical overlap with any
  ICD-10-CM concept name at all (the vocabulary only names these codes "Crohn's disease" /
  "ulcerative colitis") — BM25 contributes nothing, and dense retrieval alone ranks the correct
  code (`K50.9`) only 5th of 50 candidates, short of cracking the fused top 10 even after
  `hybrid_search`'s `candidate_k` default was widened from `k` to `5 * k` (above). Widening the
  pool measurably helped the *baseline* query for this cohort (`0.177` → `0.402`, since the full
  cohort name has more lexical signal than the bare term) but the decomposed path stays at
  `0.000` — a genuine vocabulary/terminology-coverage gap (umbrella clinical term vs.
  disease-specific vocabulary naming), not a bug, and not something a pooling default alone can
  fix. Left as a documented limitation rather than pursued further (e.g. query synonym
  expansion), since that's new scope with its own failure modes, not justified by one cohort.

(Checked separately: none of the 23 bundled cohorts currently have a ground-truth code missing
from the loaded vocabulary — `unresolvable_ground_truth` exists for exactly this case but had
never actually been populated until now; see `eval/harness.py`.)

## Encoder benchmark (`v0.6`)

`scripts/compare_encoders.py` builds a fresh LanceDB index per candidate embedding model
(re-embedding every ICD-10-CM concept name for real — no caching across embedders, since
different models produce different-dimension, incomparable vectors) and scores `dense`/`hybrid`
against the same 23 bundled cohorts, to check whether BioLORD-2023 (the shipped default) is
actually the best choice rather than just the first one tried.

Three candidates: **BioLORD-2023** (current default, symmetric), **SapBERT**
(`cambridgeltl/SapBERT-from-PubMedBERT-fulltext`, symmetric), and **MedCPT**
(`ncbi/MedCPT-Query-Encoder` + `ncbi/MedCPT-Article-Encoder`, asymmetric — a separate query
encoder and article encoder trained specifically for query-to-article retrieval, wired through
`DenseRetriever`'s `query_embed_fn` so the comparison doesn't misrepresent how MedCPT is meant to
be used). SapBERT and MedCPT both require explicit `[CLS]`-token pooling via raw `transformers`
(per their model cards' own reference code) rather than `sentence-transformers`'s default mean
pooling, which would silently not match either model's documented usage.

| embedder | dense coverage | dense over-inclusion | dense hierarchical | hybrid coverage | hybrid over-inclusion | hybrid hierarchical |
|---|---|---|---|---|---|---|
| **biolord** (current default) | 0.828 | 0.541 | 0.577 | 0.873 | 0.436 | **0.654** |
| sapbert | 0.847 | 0.589 | 0.538 | 0.854 | 0.486 | 0.599 |
| medcpt | 0.809 | 0.399 | 0.634 | 0.803 | 0.423 | 0.617 |

**Result: BioLORD stays the default.** Not a close call on the metric that actually matters —
`hybrid` is what every tool surface uses, and BioLORD wins there (0.654 vs. MedCPT's 0.617 and
SapBERT's 0.599).

The more interesting finding is the split between dense-only and hybrid: MedCPT is actually the
*best dense-only* retriever of the three by a clear margin (0.634 vs. BioLORD's 0.577), and
meaningfully less over-inclusive standalone (0.399 vs. 0.541) — its purpose-built query/article
asymmetry pays off in isolation. But once fused with BM25 into `hybrid`, BioLORD comes out ahead
again — its errors apparently complement BM25's gaps better than MedCPT's already-tighter
results do, so the fusion has less room to add value for MedCPT. SapBERT is the weakest of the
three on both dense and hybrid here. Worth revisiting if `hybrid`'s fusion weights or RRF
parameters change, since that's the exact mechanism behind the dense-vs-hybrid flip.
