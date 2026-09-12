"""Names the language provides itself, and compile-time evaluation of them.

`const_value` is pure: it never mutates the tree, so both the analysis and the
lowering may call it. It deliberately covers only numeric operations -- exactly
the set the code generator knows how to print back out as Python. Keeping the
two in step means no expression can reach the generated file that we cannot
render faithfully.
"""

from __future__ import annotations

import cmath
import math
import operator
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from slanq.ast_nodes import (
    BinaryOp,
    BuiltinDecl,
    Call,
    ClassicalDecl,
    Expression,
    FloatType,
    IntType,
    Literal,
    Name,
    Span,
    Type,
    UnaryOp,
)

BUILTIN_GATES: frozenset[str] = frozenset(
    {
        "H", "X", "Y", "Z", "S", "T", "Sdg", "Tdg",
        "RX", "RY", "RZ",
        "CX", "CY", "CZ", "CH", "SWAP", "CCX",
        "phase",
    }
)


class ArgKind(Enum):
    ANGLE = "angle"
    # A qubit operand: either a single indexed bit or a whole register.
    QUBITS = "qubits"
    # A whole quantum variable; an index is not accepted.
    QVAR = "qvar"


@dataclass(frozen=True, kw_only=True)
class Signature:
    args: tuple[ArgKind, ...]
    returns: Type | None = None
    # Qiskit broadcasts a register operand itself for one- and two-qubit gates;
    # for CCX it refuses, so the lowering has to expand the call.
    qiskit_broadcasts: bool = True


def _gates(names: str, *args: ArgKind, **extra) -> dict[str, Signature]:
    return {name: Signature(args=args, **extra) for name in names.split()}


BUILTIN_SIGNATURES: dict[str, Signature] = {
    **_gates("H X Y Z S T Sdg Tdg", ArgKind.QUBITS),
    **_gates("RX RY RZ", ArgKind.ANGLE, ArgKind.QUBITS),
    **_gates("CX CY CZ CH SWAP", ArgKind.QUBITS, ArgKind.QUBITS),
    "CCX": Signature(
        args=(ArgKind.QUBITS, ArgKind.QUBITS, ArgKind.QUBITS),
        qiskit_broadcasts=False,
    ),
    "phase": Signature(args=(ArgKind.ANGLE,)),
    "measure": Signature(args=(ArgKind.QVAR,), returns=IntType()),
    "reset": Signature(args=(ArgKind.QUBITS,)),
    **_gates("floor ceil round", ArgKind.ANGLE, returns=IntType()),
    "sqrt": Signature(args=(ArgKind.ANGLE,), returns=FloatType()),
}


def _round_half_away(value: float) -> int:
    """Slanq rounds a half away from zero; Python's `round` rounds to even."""
    return math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)


def _sqrt(value: float | complex) -> float | complex:
    """A negative real or complex input promotes to a complex result (with the
    expected `i`); a nonnegative real input stays a plain float."""
    if isinstance(value, complex) or value < 0:
        return cmath.sqrt(value)
    return math.sqrt(value)


BUILTIN_CONSTANTS: dict[str, float] = {"PI": math.pi}

BUILTIN_FUNCTIONS: dict[str, Callable[[Any], Any]] = {
    "floor": math.floor,
    "ceil": math.ceil,
    "round": _round_half_away,
    "sqrt": _sqrt,
}

# Recognised only as the iterable of a `for`, never as a value.
RANGE = "range"

BUILTIN_NAMES: frozenset[str] = frozenset(
    BUILTIN_SIGNATURES.keys() | BUILTIN_CONSTANTS.keys() | {RANGE}
)

# Builtins are not declared in the source, so they share one placeholder span.
# Every diagnostic about a builtin points at the use site, never at this.
BUILTIN_SPAN = Span(start_line=0, start_col=0, end_line=0, end_col=0)

BUILTIN_SCOPE: dict[str, BuiltinDecl] = {
    **{
        name: BuiltinDecl(span=BUILTIN_SPAN, name=name, signature=signature)
        for name, signature in BUILTIN_SIGNATURES.items()
    },
    **{
        name: BuiltinDecl(span=BUILTIN_SPAN, name=name, declared_type=FloatType())
        for name in BUILTIN_CONSTANTS
    },
    # No signature: `range` is variadic and is only meaningful as the iterable
    # of a for loop, which the loop checker handles on its own.
    RANGE: BuiltinDecl(span=BUILTIN_SPAN, name=RANGE),
}

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


# Only `int ** positive int` can blow up memory: `float`/`complex` power is
# fixed-precision (uses a C log/exp implementation, never grows). A right-
# associative chain (`2 ** 2 ** 2 ** 20`) explodes at whichever step first
# exceeds this, so checking each `**` application individually, before it
# runs, is enough to stop a tower regardless of depth -- the exponent itself
# may already be a huge int object by the outer steps, but computing its bit
# count and multiplying by the base's is cheap even then.
_MAX_POWER_RESULT_BITS = 4096


def _guard_power(base: object, exponent: object) -> None:
    if type(base) is not int or type(exponent) is not int:
        return
    if exponent <= 1 or abs(base) <= 1:
        return
    if base.bit_length() * exponent > _MAX_POWER_RESULT_BITS:
        raise ConstEvalError(
            f"'**' result would need more than {_MAX_POWER_RESULT_BITS} bits to "
            "represent -- this is not evaluated at compile time"
        )


def const_value(expression: Expression) -> int | float | bool | complex | None:
    """The compile-time value of `expression`, or None if it has none."""
    if isinstance(expression, Literal):
        return expression.value

    if isinstance(expression, Name):
        if expression.name in BUILTIN_CONSTANTS:
            return BUILTIN_CONSTANTS[expression.name]
        if isinstance(expression.resolved_symbol, ClassicalDecl):
            return const_value(expression.resolved_symbol.initializer)
        return None

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
        if expression.op == "**":
            _guard_power(left, right)
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
    "BUILTIN_SCOPE",
    "BUILTIN_SIGNATURES",
    "BUILTIN_SPAN",
    "BUILTIN_CONSTANTS",
    "BUILTIN_FUNCTIONS",
    "BUILTIN_GATES",
    "BUILTIN_NAMES",
    "RANGE",
    "UNARY_OPS",
    "ArgKind",
    "ConstEvalError",
    "Signature",
    "const_int",
    "const_value",
]
