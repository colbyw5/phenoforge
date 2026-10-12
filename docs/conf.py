"""Sphinx configuration for phenoforge's API reference.

Docstrings throughout src/phenoforge/ are already Sphinx-style
(:param:, :returns:, :rtype:, :raises: — see AGENTS.md), so this needs
only autodoc, not napoleon (which is for Google/NumPy-style docstrings).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

project = "phenoforge"
copyright = "2026, Colby Wilkinson"
author = "Colby Wilkinson"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

# Sphinx autodoc imports every documented module with try_reload=True —
# re-executing a module that defines Pydantic BaseModel subclasses (e.g.
# langchain_core.messages.BaseMessage) during that reload corrupts
# pydantic's schema-building state (a real, reproduced failure: "the type
# annotation for __pydantic_extra__ must be dict[str, ...]" — a plain,
# non-reloaded `import phenoforge.agent.nodes` works fine). Mocking these
# avoids the reload entirely; our own modules' docstrings/signatures still
# render, just without resolving these packages' own internal types.
autodoc_mock_imports = ["langgraph", "langchain_core", "langchain_anthropic", "langsmith", "mcp"]

# Cosmetic only: sphinx-autodoc-typehints trips on pydantic.Field's own
# internal type hints while documenting a class attribute that uses
# Field(...) as a default value — nothing wrong in this project's code.
suppress_warnings = ["sphinx_autodoc_typehints.forward_reference"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

html_theme = "furo"
html_title = "phenoforge"
