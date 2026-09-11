"""Syntactic analysis: Slanq source -> Lark parse tree.

Tokenization and parsing are performed by Lark using the `grammar/slanq.lark`
definition. The `transformer` module turns the parse tree into our own AST.
"""

from __future__ import annotations

from functools import lru_cache

from lark import Lark, ParseTree
from lark.exceptions import UnexpectedCharacters, UnexpectedInput

from slanq.diagnostics import SlanqError
from slanq.grammar import grammar_text

_PAIRS = {"(": ")", "[": "]", "{": "}"}


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
        line, column, detail = _describe(exc, source)
        raise SlanqError(f"Syntax error at line {line}, column {column}: {detail}") from exc


def _describe(exc: UnexpectedInput, source: str) -> tuple[int, int, str]:
    """Where the error is and what was found there.

    Lark reports running out of input as line -1, column -1 and an empty
    token, which points at nothing. The end of the text is the honest position
    for that, and an unclosed bracket -- the usual cause -- can say where it
    was opened, which is the part worth reading.
    """
    if isinstance(exc, UnexpectedCharacters):
        found = source.splitlines()[exc.line - 1][exc.column - 1]
        return exc.line, exc.column, f"unexpected {found!r}"

    line, column = _end_of_text(source)
    unclosed = _unclosed_bracket(source)
    if unclosed is None:
        return line, column, "unexpected end of file"
    bracket, open_line, open_column = unclosed
    return (
        line,
        column,
        f"unexpected end of file: {bracket!r} at line {open_line}, "
        f"column {open_column} is never closed",
    )


def _end_of_text(source: str) -> tuple[int, int]:
    """Just past the last character that is not trailing whitespace, so the
    position lands where the text stops rather than on a blank final line."""
    stripped = source.rstrip()
    if not stripped:
        return 1, 1
    lines = stripped.splitlines()
    return len(lines), len(lines[-1]) + 1


def _unclosed_bracket(source: str) -> tuple[str, int, int] | None:
    """The innermost bracket still open at the end of the source. Comments are
    skipped; the language has no string literals, so nothing else can hide a
    bracket from a plain scan."""
    stack: list[tuple[str, int, int]] = []
    for line_number, line in enumerate(source.splitlines(), start=1):
        text = line.split("//", 1)[0]
        for column, character in enumerate(text, start=1):
            if character in _PAIRS:
                stack.append((character, line_number, column))
            elif character in _PAIRS.values():
                if not stack or _PAIRS[stack[-1][0]] != character:
                    return None  # mismatched: the caller's position is better
                stack.pop()
    return stack[-1] if stack else None


__all__ = ["get_parser", "parse_source"]
