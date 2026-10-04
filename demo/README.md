# Demo vocabulary

A tiny, hand-built, **ICD-10-CM-only** vocabulary subset (22 codes: the diabetes/CKD cluster from
the README's own example, plus a few unrelated codes for retrieval contrast), shaped exactly like
an Athena bulk export (`CONCEPT.csv` / `CONCEPT_RELATIONSHIP.csv`, tab-delimited) so
`scripts/load_vocab.py`'s real loader builds it with no special-casing.

Committed to the repo (unlike everything under `data/`) because, unlike a real Athena download,
every byte of it is safe to redistribute:

- **ICD-10-CM codes and descriptions are U.S. public domain** (CMS/NCHS) — no license at all,
  unlike SNOMED CT or the OHDSI Phenotype Library content this project otherwise keeps
  gitignored and never ships (see root `README.md`'s Setup section and `AGENTS.md`).
- **`concept_id` values are invented for this file**, not copied from OHDSI's compiled
  vocabulary numbering — sidesteps any question about redistributing *their* compilation, not
  just the underlying public-domain codes.
- **Contains no SNOMED rows at all.** That means `find_curated_definition` (which resolves
  curated cohorts through SNOMED) has nothing to resolve against here — this vocabulary only
  demonstrates the `generated` tier (BM25, dense, hierarchy expansion), not `curated`. See
  `scripts/run_demo.py` and the root README's Setup section for the real thing.

Used by `scripts/run_demo.py` to build a throwaway DuckDB on the fly — never written back here.
