"""Package-level smoke tests: the package imports, the grammar ships with the wheel."""

from __future__ import annotations

import slanq
from slanq.grammar import grammar_text


def test_version_is_exposed() -> None:
    assert slanq.__version__


def test_grammar_is_packaged() -> None:
    """The .lark file must be part of the package, not just of the source tree.

    This test fails if the grammar file is missing from the built wheel.
    """
    text = grammar_text()
    assert "start: statement*" in text
