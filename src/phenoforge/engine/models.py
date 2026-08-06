"""Pydantic models shared across the engine, MCP server, and agent.

Every code set carries provenance: a concept set is never a bare list of
codes, every concept carries where it came from, and terms that could not
be resolved are reported rather than silently dropped. This is the
project's anti-hallucination commitment — retrofitting it after the fact
would mean changing every caller's return type.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class ProvenanceTier(StrEnum):
    """Trust tier for a concept's inclusion in a set.

    :cvar CURATED: From the OHDSI Phenotype Library. Use as-is, citing the
        cohort ID and operating characteristics. Not yet produced by
        anything in this codebase.
    :cvar PUBLISHED: From a validated published algorithm in the literature.
        Deferred to a later version — not yet produced by this codebase.
    :cvar GENERATED: From semantic/BM25 retrieval or hierarchy expansion.
        Ungrounded — requires human confirmation before use.
    """

    CURATED = "curated"
    PUBLISHED = "published"
    GENERATED = "generated"


class Concept(BaseModel):
    """A single vocabulary concept, as loaded into ``vocab.duckdb``.

    :ivar concept_id: OMOP concept id.
    :ivar concept_code: Vocabulary-native code, e.g. ``"E11.21"``.
    :ivar concept_name: Human-readable concept name.
    :ivar domain_id: OMOP domain, e.g. ``"Condition"``.
    :ivar vocabulary_id: Source vocabulary. SNOMED rows are loaded locally
        for future curated-set resolution but are never a lookup or search
        target — only ``"ICD10CM"`` is ever surfaced outside the engine.
    """

    concept_id: int
    concept_code: str
    concept_name: str
    domain_id: str
    vocabulary_id: str


class ConceptWithProvenance(Concept):
    """A concept annotated with why it is in a result set.

    :ivar tier: The trust tier this concept was included under.
    :ivar source: Free-text description of the retrieval step that produced
        this concept, e.g. ``"hierarchy_expansion:E11"`` or
        ``"bm25:diabetic nephropathy"``. Not machine-parsed; exists for
        human review and debugging.
    """

    tier: ProvenanceTier
    source: str


class UnmappableTerm(BaseModel):
    """A query term that could not be resolved to any concept.

    :ivar term: The term or phrase that failed to resolve.
    :ivar reason: Why it failed, e.g. ``"no BM25 match above threshold"``.
    """

    term: str
    reason: str


class ConceptSet(BaseModel):
    """A retrieval or expansion result.

    Deliberately not a bare list of codes (AGENTS.md): every concept carries
    provenance, and unresolved terms are surfaced rather than dropped.

    :ivar concepts: Concepts included in the set, each with its provenance.
    :ivar unmappable: Terms that could not be resolved to any concept.
    """

    concepts: list[ConceptWithProvenance] = Field(default_factory=list)
    unmappable: list[UnmappableTerm] = Field(default_factory=list)
