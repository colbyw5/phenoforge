"""LangGraph agent: tiered-trust cohort assembly.

Realizes the tiered-provenance model (``engine/models.py``'s
``ProvenanceTier``) as an actual workflow rather than just a data-model
convention: decompose a plain-English population description into seed
clinical terms, check each against the curated OHDSI Phenotype Library
first (``curated`` — used as-is), and only fall back to generated retrieval
(BM25/dense/hybrid — ``generated``) for terms with no curated match. Every
generated candidate is staged for human confirmation before it can appear
in the final result — the tiered-trust discipline enforced as a real pause
in the graph, not just a label on the data.

This module is what ``eval/__init__.py`` named as the deferred capability:
"decomposition accuracy... that's the not-yet-built LangGraph agent's job."
Decomposition (``agent/nodes.py``'s ``decompose`` node) is a real LLM call,
not a rule-based stub.

Flow: ``decompose -> check_curated -> [generate -> confirm] -> assemble``,
the bracketed segment skipped when every term resolves curated. See
``agent/graph.py``'s module docstring for the exact graph topology and the
interrupt/resume payload contract that powers the confirmation gate.

Per AGENTS.md's layering rule, this package imports ``engine`` directly —
it is not an MCP client. ``mcp`` and ``agent`` are independent consumers of
the same engine.
"""
