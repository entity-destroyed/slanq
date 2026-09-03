from __future__ import annotations

import math
from collections.abc import Callable

import pytest

from slanq.ast_nodes import Expression, IntType, Program
from slanq.builtin import (
    BUILTIN_GATES,
    BUILTIN_NAMES,
    BUILTIN_SCOPE,
    BUILTIN_SIGNATURES,
    ArgKind,
    ConstEvalError,
    const_int,
    const_value,
)

BuildAst = Callable[[str], Program]


@pytest.fixture
def value_of(build_ast: BuildAst) -> Callable[[str], object]:
    def _value_of(source: str) -> object:
        (declaration,) = build_ast(f"float r = {source};").statements
        return const_value(declaration.initializer)

    return _value_of


@pytest.fixture
def expression_of(build_ast: BuildAst) -> Callable[[str], Expression]:
    def _expression_of(source: str) -> Expression:
        (declaration,) = build_ast(f"float r = {source};").statements
        return declaration.initializer

    return _expression_of


def test_pi_is_a_constant(value_of) -> None:
    assert value_of("PI") == math.pi
    assert value_of("PI / 4") == pytest.approx(math.pi / 4)


def test_division_always_yields_a_float(value_of) -> None:
    assert value_of("7 / 2") == 3.5
    assert isinstance(value_of("4 / 2"), float)


def test_modulo_follows_python_on_negatives(value_of) -> None:
    assert value_of("-7 % 3") == 2


def test_power_is_right_associative(value_of) -> None:
    assert value_of("2 ** 3 ** 2") == 512


def test_unary_minus_binds_looser_than_power(value_of) -> None:
    assert value_of("-2 ** 2") == -4
    assert value_of("(-2) ** 2") == 4


def test_bitwise_binds_tighter_than_comparison(build_ast: BuildAst) -> None:
    """Python's precedence, not C's: `1 & 2 == 2` is `(1 & 2) == 2`."""
    (declaration,) = build_ast("int r = 1 & 2 == 2;").statements
    assert declaration.initializer.op == "=="
    assert declaration.initializer.left.op == "&"


def test_round_goes_away_from_zero(value_of) -> None:
    """Unlike Python's round(), which rounds a half to the even neighbour."""
    assert value_of("round(2.5)") == 3
    assert value_of("round(-2.5)") == -3
    assert value_of("round(1.5)") == 2


def test_floor_and_ceil(value_of) -> None:
    assert value_of("floor(7 / 2)") == 3
    assert value_of("ceil(7 / 2)") == 4
    assert value_of("floor(-2.5)") == -3


def test_complex_arithmetic(value_of) -> None:
    assert value_of("0.5+0.3i") == complex(0.5, 0.3)
    assert value_of("1i * 1i") == complex(-1, 0)
    assert value_of("-1i") == complex(0, -1)
    assert value_of("(1+1i) * (1-1i)") == complex(2, 0)


def test_unsupported_operators_on_complex_are_errors(expression_of) -> None:
    for source in ("1i % 2", "1i & 1", "1i | 1", "1i ^ 1", "~1i"):
        with pytest.raises(ConstEvalError):
            const_value(expression_of(source))


def test_sqrt_of_a_nonnegative_real_stays_a_float(value_of) -> None:
    assert value_of("sqrt(4)") == 2.0
    assert isinstance(value_of("sqrt(4)"), float)
    assert value_of("sqrt(2)") == pytest.approx(2**0.5)


def test_sqrt_of_a_negative_real_promotes_to_complex(value_of) -> None:
    """`sqrt(-4)` gives exactly `2i`, not an error -- the whole reason `sqrt`
    exists alongside the complex type."""
    assert value_of("sqrt(-4)") == complex(0, 2)


def test_sqrt_of_a_complex_number(value_of) -> None:
    result = value_of("sqrt(1i)")
    assert isinstance(result, complex)
    assert result * result == pytest.approx(complex(0, 1))


def test_non_constant_expression_has_no_value(value_of) -> None:
    assert value_of("n + 1") is None


def test_comparisons_are_not_constant_folded(value_of) -> None:
    """Only what the code generator can print back out is evaluated here."""
    assert value_of("1 < 2") is None


def test_division_by_zero_is_an_error(expression_of) -> None:
    with pytest.raises(ConstEvalError):
        const_value(expression_of("1 / 0"))


def test_bitwise_on_a_float_is_an_error(expression_of) -> None:
    with pytest.raises(ConstEvalError):
        const_value(expression_of("1.5 & 2"))


def test_const_int_rejects_a_bool(expression_of) -> None:
    """`true` is not an index, even though Python calls a bool an int."""
    assert const_int(expression_of("true")) is None
    assert const_int(expression_of("1 + 1")) == 2


def test_const_int_rejects_a_fraction(expression_of) -> None:
    assert const_int(expression_of("3 / 2")) is None


def test_builtin_names_cover_gates_constants_and_functions() -> None:
    assert {"H", "measure", "PI", "floor", "ceil", "round", "sqrt", "range"} <= BUILTIN_NAMES


def test_every_gate_has_a_signature() -> None:
    assert not BUILTIN_GATES - BUILTIN_SIGNATURES.keys()


def test_returning_builtins_declare_their_type() -> None:
    for name in ("measure", "floor", "ceil", "round"):
        assert isinstance(BUILTIN_SIGNATURES[name].returns, IntType), name


def test_gates_return_nothing() -> None:
    assert all(BUILTIN_SIGNATURES[name].returns is None for name in BUILTIN_GATES)


def test_ccx_is_broadcast_by_slanq_not_qiskit() -> None:
    """Measured: Qiskit refuses a register operand for ccx, so the lowering
    has to unroll it; every other gate Qiskit spreads on its own."""
    assert not BUILTIN_SIGNATURES["CCX"].qiskit_broadcasts
    assert BUILTIN_SIGNATURES["CX"].qiskit_broadcasts


def test_measure_takes_a_whole_variable() -> None:
    assert BUILTIN_SIGNATURES["measure"].args == (ArgKind.QVAR,)


def test_rotation_takes_the_angle_first() -> None:
    assert BUILTIN_SIGNATURES["RX"].args == (ArgKind.ANGLE, ArgKind.QUBITS)


def test_builtin_scope_covers_every_builtin_name() -> None:
    assert BUILTIN_SCOPE.keys() == set(BUILTIN_NAMES)
