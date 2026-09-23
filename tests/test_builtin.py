from __future__ import annotations

import math
import time
from collections.abc import Callable

import pytest

from slanq.analysis import analyze
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
from slanq.diagnostics import DiagnosticBag

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


@pytest.fixture
def last_value_of(build_ast: BuildAst) -> Callable[[str], object]:
    """Like `value_of`, but for multi-statement source, evaluating the last
    statement's initializer -- needed to test Name resolution, which depends
    on `resolved_symbol` from a real `analyze()` pass, not just parsing."""

    def _last_value_of(source: str) -> object:
        ast = build_ast(source)
        analyze(ast, DiagnosticBag())
        return const_value(ast.statements[-1].initializer)

    return _last_value_of


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


def test_classical_name_resolves_to_its_initializer(last_value_of) -> None:
    assert last_value_of("complex c = 0.5+0.3i; complex d = c;") == complex(0.5, 0.3)
    assert last_value_of("float t = PI / 4; float u = t * 2;") == pytest.approx(math.pi / 2)


def test_classical_name_resolution_is_transitive(last_value_of) -> None:
    assert last_value_of("int a = 2; int b = a; int c = b + 1;") == 3


def test_param_name_is_not_a_compile_time_constant(last_value_of) -> None:
    assert last_value_of("param float p; float x = p;") is None


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("1 < 2", True),
        ("2 == 3", False),
        ("2 != 3", True),
        ("3 >= 3", True),
        ("1 < 2 && 3 > 4", False),
        ("1 < 2 || 3 > 4", True),
        ("!(1 < 2)", False),
    ],
)
def test_comparisons_and_logic_are_evaluated(value_of, source, expected) -> None:
    assert value_of(source) is expected


def test_division_by_zero_is_an_error(expression_of) -> None:
    with pytest.raises(ConstEvalError):
        const_value(expression_of("1 / 0"))


def test_huge_power_is_rejected(expression_of) -> None:
    with pytest.raises(ConstEvalError, match="bits to represent"):
        const_value(expression_of("2 ** 10000000"))


def test_power_tower_is_rejected_without_hanging(expression_of) -> None:
    """Right-associativity means this is 2 ** (2 ** (2 ** 20)) -- the outer
    step alone would need far more memory than exists to even represent the
    result, so the guard must catch it before `operator.pow` ever runs."""
    with pytest.raises(ConstEvalError, match="bits to represent"):
        const_value(expression_of("2 ** 2 ** 2 ** 20"))


def test_power_guard_does_not_reject_legitimate_constants(value_of) -> None:
    assert value_of("2 ** 100") == 2**100
    assert value_of("1 ** 999999999999") == 1
    assert value_of("(-1) ** 999999999999") == -1
    assert value_of("2 ** -1000000") == 0.0
    assert value_of("2.5 ** 3") == 15.625


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


def test_reset_is_a_quantum_builtin_that_is_not_a_gate() -> None:
    """The qif body rule reads `BUILTIN_GATES` as the unitary set, so reset
    staying out of it is what makes that rule reject reset."""
    assert "reset" not in BUILTIN_GATES
    assert BUILTIN_SIGNATURES["reset"].args == (ArgKind.QUBITS,)
    assert BUILTIN_SIGNATURES["reset"].returns is None


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


def test_a_long_definition_chain_is_evaluated_once_per_link(last_value_of) -> None:
    """Each link reads the one before it twice, so walking the chain again
    at every reference doubles the work per link. The bound is what has
    teeth: this chain is measured at about 0.1 s with the memo and about
    27 s without, so the value alone would not notice it going missing."""
    lines = ["float v0 = 1.0;"]
    lines += [f"float v{k} = v{k - 1} + v{k - 1};" for k in range(1, 25)]
    program = "\n".join(lines) + "\nfloat last = v24;\n"
    started = time.perf_counter()
    value = last_value_of(program)
    assert value == 2.0**24
    assert time.perf_counter() - started < 5.0
