"""Syntactic analysis: Slanq source -> Lark parse tree.

Tokenization and parsing are performed by Lark using the `grammar/slanq.lark`
definition. The `transformer` module turns the parse tree into our own AST.
"""

from __future__ import annotations

from functools import lru_cache

from lark import Lark, ParseTree
from lark.exceptions import UnexpectedInput

from slanq.diagnostics import SlanqError
from slanq.grammar import grammar_text


@lru_cache(maxsize=1)
def get_parser() -> Lark:
    """Return the Lark parser instance (cached, since building it is expensive)."""
    return Lark(
        grammar_text(),
        start="start",
        parser="earley",
        propagate_positions=True,
    )


def parse_source(source: str) -> ParseTree:
    """Parse Slanq source code and return the Lark parse tree.

    Raises SlanqError if the source is syntactically invalid.
    """
    try:
        return get_parser().parse(source)
    except UnexpectedInput as exc:
        raise SlanqError(f"Syntax error at line {exc.line}, column {exc.column}") from exc


__all__ = ["get_parser", "parse_source"]
