"""Names the language provides itself, and compile-time evaluation of them.

`const_value` is pure: it never mutates the tree, so both the analysis and the
lowering may call it. It deliberately covers only numeric operations -- exactly
the set the code generator knows how to print back out as Python. Keeping the
two in step means no expression can reach the generated file that we cannot
render faithfully.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Callable

from slanq.ast_nodes import BinaryOp, Call, Expression, Literal, Name, UnaryOp

BUILTIN_GATES: frozenset[str] = frozenset(
    {
        "H", "X", "Y", "Z", "S", "T", "Sdg", "Tdg",
        "RX", "RY", "RZ",
        "CX", "CY", "CZ", "CH", "SWAP", "CCX",
        "phase",
        "measure",
    }
)


def _round_half_away(value: float) -> int:
    """Slanq rounds a half away from zero; Python's `round` rounds to even."""
    return math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)


BUILTIN_CONSTANTS: dict[str, float] = {"PI": math.pi}

BUILTIN_FUNCTIONS: dict[str, Callable[[float], int]] = {
    "floor": math.floor,
    "ceil": math.ceil,
    "round": _round_half_away,
}

# Recognised only as the iterable of a `for`, never as a value.
RANGE = "range"

BUILTIN_NAMES: frozenset[str] = frozenset(
    BUILTIN_GATES | BUILTIN_CONSTANTS.keys() | BUILTIN_FUNCTIONS.keys() | {RANGE}
)

BINARY_OPS: dict[str, Callable[[object, object], object]] = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "%": operator.mod,
    "**": operator.pow,
    "&": operator.and_,
    "|": operator.or_,
    "^": operator.xor,
}

UNARY_OPS: dict[str, Callable[[object], object]] = {
    "-": operator.neg,
    "~": operator.invert,
}


class ConstEvalError(Exception):
    """The expression is constant but cannot be evaluated (division by zero,
    a bitwise operator on a float). Distinct from "not constant", which is a
    `None` result and not an error on its own."""


def const_value(expression: Expression) -> int | float | bool | None:
    """The compile-time value of `expression`, or None if it has none."""
    if isinstance(expression, Literal):
        return expression.value

    if isinstance(expression, Name):
        return BUILTIN_CONSTANTS.get(expression.name)

    if isinstance(expression, UnaryOp):
        operand = const_value(expression.operand)
        function = UNARY_OPS.get(expression.op)
        if operand is None or function is None:
            return None
        return _apply(function, (operand,), expression.op)

    if isinstance(expression, BinaryOp):
        function = BINARY_OPS.get(expression.op)
        if function is None:
            return None
        left = const_value(expression.left)
        right = const_value(expression.right)
        if left is None or right is None:
            return None
        return _apply(function, (left, right), expression.op)

    if isinstance(expression, Call):
        function = BUILTIN_FUNCTIONS.get(expression.callee.name)
        if function is None or len(expression.args) != 1:
            return None
        argument = const_value(expression.args[0])
        if argument is None:
            return None
        return _apply(function, (argument,), expression.callee.name)

    return None


def _apply(function, arguments, what: str):
    try:
        return function(*arguments)
    except ZeroDivisionError:
        raise ConstEvalError("division by zero") from None
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConstEvalError(f"'{what}' cannot be evaluated here: {exc}") from None


def const_int(expression: Expression) -> int | None:
    """The compile-time value when it is a whole number, otherwise None.
    A bool is rejected: `true` is not an index."""
    value = const_value(expression)
    if type(value) is int:
        return value
    return None


__all__ = [
    "BINARY_OPS",
    "BUILTIN_CONSTANTS",
    "BUILTIN_FUNCTIONS",
    "BUILTIN_GATES",
    "BUILTIN_NAMES",
    "RANGE",
    "UNARY_OPS",
    "ConstEvalError",
    "const_int",
    "const_value",
]
