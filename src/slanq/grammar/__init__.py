"""Loading of the grammar definition.

The .lark file is read through the `importlib.resources` API rather than by a
relative file path, so the compiler finds the grammar when the package runs
installed, and from any working directory.
"""

from __future__ import annotations

from importlib import resources

GRAMMAR_FILENAME = "slanq.lark"


def grammar_text() -> str:
    """Return the Slanq grammar source as text."""
    return resources.files(__package__).joinpath(GRAMMAR_FILENAME).read_text(encoding="utf-8")


__all__ = ["GRAMMAR_FILENAME", "grammar_text"]
