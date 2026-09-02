"""End-to-end: Slanq source is compiled, executed and run on a simulator."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector

from slanq.compiler import compile_source

SHOTS = 1024
SEED = 1234


def _build_circuit(source: str) -> QuantumCircuit:
    result = compile_source(source, source_name="test.slanq")
    assert not result.diagnostics.has_errors
    assert result.qiskit_source is not None

    namespace: dict[str, Any] = {}
    exec(result.qiskit_source, namespace)  # noqa: S102
    return namespace["build_circuit"]()


def _counts(source: str, register: str = "result") -> dict[str, int]:
    circuit = _build_circuit(source)
    result = StatevectorSampler(seed=SEED).run([circuit], shots=SHOTS).result()
    return getattr(result[0].data, register).get_counts()


def test_mvp_a_circuit_shape(mvp_a_source: str) -> None:
    circuit = _build_circuit(mvp_a_source)
    assert circuit.num_qubits == 1
    assert circuit.num_clbits == 1


def test_hadamard_gives_both_outcomes(mvp_a_source: str) -> None:
    counts = _counts(mvp_a_source)
    assert set(counts) == {"0", "1"}
    assert all(SHOTS * 0.3 < value < SHOTS * 0.7 for value in counts.values())


def test_false_initializer_always_measures_zero() -> None:
    assert _counts("qbool q = false;\nint result = measure(q);\n") == {"0": SHOTS}


def test_true_initializer_always_measures_one() -> None:
    assert _counts("qbool q = true;\nint result = measure(q);\n") == {"1": SHOTS}


def test_double_x_returns_to_zero() -> None:
    source = "qbool q = false;\nX(q);\nX(q);\nint result = measure(q);\n"
    assert _counts(source) == {"0": SHOTS}


def test_bell_state_outcomes_are_correlated(mvp_b_source: str) -> None:
    counts = _counts(mvp_b_source)
    assert set(counts) == {"00", "11"}
    assert all(SHOTS * 0.3 < value < SHOTS * 0.7 for value in counts.values())


def test_qint_constant_survives_the_endianness_split() -> None:
    """The canary for B1: a mirrored index or bit order would change the value."""
    counts = _counts("qint<3> a = 5; int result = measure(a);")
    assert counts == {"101": SHOTS}


def test_indexed_gate_targets_the_intended_qubit() -> None:
    """Slanq a[0] is the most significant qubit, so flipping it gives 4, not 1."""
    counts = _counts("qint<3> a = 0; X(a[0]); int result = measure(a);")
    assert counts == {"100": SHOTS}


def test_last_index_is_the_least_significant_qubit() -> None:
    counts = _counts("qint<3> a = 0; X(a[2]); int result = measure(a);")
    assert counts == {"001": SHOTS}


def test_pi_rotation_flips_the_qubit() -> None:
    """RX(PI) is a bit flip up to phase; a half of it would not be deterministic."""
    source = "qbool q = false;\nRX(PI, q);\nint result = measure(q);\n"
    assert _counts(source) == {"1": SHOTS}


def test_a_computed_angle_reaches_the_circuit_unfolded() -> None:
    source = "qbool q = false;\nRX(round(2.5) * PI / floor(7 / 2), q);\n"
    result = compile_source(source, source_name="test.slanq")
    assert result.qiskit_source is not None
    assert "_round(2.5) * np.pi / math.floor(7 / 2)" in result.qiskit_source

    counts = _counts(source + "int result = measure(q);\n")
    assert counts == {"1": SHOTS}


def test_ccx_broadcast_computes_a_bitwise_and() -> None:
    """CCX over whole registers is one Toffoli per bit, so c becomes a & b."""
    source = (
        "qint<2> a = 3;\nqint<2> b = 2;\nqint<2> c = 0;\n"
        "CCX(a, b, c);\nint result = measure(c);\n"
    )
    assert _counts(source) == {"10": SHOTS}


def test_empty_probability_list_is_a_cheap_equal_superposition() -> None:
    """[] is meant as Hadamards on every qubit, not a general StatePreparation."""
    source = "qint<2> a = [];\nint result = measure(a);\n"
    result = compile_source(source, source_name="test.slanq")
    assert result.qiskit_source is not None
    assert "StatePreparation" not in result.qiskit_source

    counts = _counts(source)
    assert set(counts) == {"00", "01", "10", "11"}
    assert all(SHOTS * 0.15 < value < SHOTS * 0.35 for value in counts.values())


def test_uneven_probability_list_matches_the_given_distribution() -> None:
    """Slanq's index i and Qiskit's StatePreparation amplitude index i name the
    same basis state with no permutation -- the same rule as InitOp and
    measurement. Would fail if that mapping ever needed a mirror."""
    source = "qint<2> b = [0, 0, 0.8, 0.2];\nint result = measure(b);\n"
    counts = _counts(source)
    assert set(counts) == {"10", "11"}
    assert counts["10"] > counts["11"]
    assert SHOTS * 0.7 < counts["10"] < SHOTS * 0.9


def test_probability_list_needing_normalization_still_runs() -> None:
    """A warning does not stop compilation, and the normalized amplitudes
    still describe a valid, correctly-proportioned distribution."""
    source = "qint<2> c = [0.1, 0.1, 0.1, 0.1];\nint result = measure(c);\n"
    result = compile_source(source, source_name="test.slanq")
    assert result.diagnostics.warnings
    assert not result.diagnostics.has_errors

    counts = _counts(source)
    assert set(counts) == {"00", "01", "10", "11"}
    assert all(SHOTS * 0.15 < value < SHOTS * 0.35 for value in counts.values())


def test_qbool_probability_list() -> None:
    source = "qbool q = [0.2, 0.8];\nint result = measure(q);\n"
    counts = _counts(source)
    assert set(counts) == {"0", "1"}
    assert counts["1"] > counts["0"]


def test_qif_equality_condition_fires_only_on_match() -> None:
    for a_value, expected in [(0, "0"), (1, "0"), (2, "1"), (3, "0")]:
        source = (
            f"qint<2> a = {a_value};\nqbool out = false;\n"
            "qif(a == 2) { X(out); }\nint result = measure(out);\n"
        )
        assert _counts(source) == {expected: SHOTS}


def test_qif_negated_condition_fires_on_mismatch() -> None:
    for a_value, expected in [(0, "1"), (1, "1"), (2, "0"), (3, "1")]:
        source = (
            f"qint<2> a = {a_value};\nqbool out = false;\n"
            "qif(!(a == 2)) { X(out); }\nint result = measure(out);\n"
        )
        assert _counts(source) == {expected: SHOTS}


def test_qif_and_chain_across_two_registers() -> None:
    source = (
        "qint<2> a = 2;\nqbool flag = true;\nqbool out = false;\n"
        "qif(a == 2 && flag) { X(out); }\nint result = measure(out);\n"
    )
    assert _counts(source) == {"1": SHOTS}

    source_false = (
        "qint<2> a = 2;\nqbool flag = false;\nqbool out = false;\n"
        "qif(a == 2 && flag) { X(out); }\nint result = measure(out);\n"
    )
    assert _counts(source_false) == {"0": SHOTS}


def test_qif_two_qubit_gate_in_body_respects_endianness() -> None:
    """CX(a[0], b[0]) inside a qif body must mirror exactly as it would at the
    top level -- this is the canary for the sub-circuit reusing the same
    operand renderer as the outer one."""
    source = (
        "qint<2> a = 2;\nqint<2> b = 0;\nqbool ctrl = true;\n"
        "qif(ctrl) { CX(a[0], b[0]); }\nint result = measure(b);\n"
    )
    # a=2 is '10' MSB-first, so a[0] (Slanq MSB) is 1: the control fires.
    assert _counts(source) == {"10": SHOTS}


def test_qif_phase_shifts_only_the_matching_branch() -> None:
    """Counts cannot see a relative phase, so this reads the statevector
    directly instead."""
    source = "qint<2> a = [];\nqif(a == 2) { phase(1.5707963267948966); }\n"
    circuit = _build_circuit(source)
    statevector = Statevector.from_instruction(circuit)

    for index, amplitude in enumerate(statevector.data):
        if abs(amplitude) < 1e-9:
            continue
        assert abs(amplitude) == pytest.approx(0.5)
        expected_phase = np.pi / 2 if index == 2 else 0.0
        assert np.angle(amplitude) == pytest.approx(expected_phase, abs=1e-9)


def test_qif_negated_phase_leaves_the_ancilla_clean() -> None:
    source = "qint<2> a = [];\nqif(!(a == 2)) { phase(1.5707963267948966); }\n"
    circuit = _build_circuit(source)
    statevector = Statevector.from_instruction(circuit)

    seen_a_values = set()
    for index, amplitude in enumerate(statevector.data):
        if abs(amplitude) < 1e-9:
            continue
        ancilla_bit = index >> 2
        assert ancilla_bit == 0, "the ancilla must return to |0> on every branch"
        a_value = index & 0b11
        seen_a_values.add(a_value)
        expected_phase = 0.0 if a_value == 2 else np.pi / 2
        assert np.angle(amplitude) == pytest.approx(expected_phase, abs=1e-9)
    assert seen_a_values == {0, 1, 2, 3}


def test_qif_not_equal_matches_the_negated_condition() -> None:
    for a_value, expected in [(0, "1"), (1, "1"), (2, "0"), (3, "1")]:
        source = (
            f"qint<2> a = {a_value};\nqbool out = false;\n"
            "qif(a != 2) { X(out); }\nint result = measure(out);\n"
        )
        assert _counts(source) == {expected: SHOTS}


def test_qif_equality_accepts_reversed_operands() -> None:
    source = (
        "qint<2> a = 2;\nqbool out = false;\n"
        "qif(2 == a) { X(out); }\nint result = measure(out);\n"
    )
    assert _counts(source) == {"1": SHOTS}


def test_qif_multiple_negated_clauses_combine_via_and() -> None:
    for a_value, b_value, flag, expected in [
        (2, 3, "true", "0"),
        (1, 1, "true", "1"),
        (1, 1, "false", "0"),
    ]:
        source = (
            f"qint<2> a = {a_value};\nqint<2> b = {b_value};\nqbool flag = {flag};\n"
            "qbool out = false;\n"
            "qif(!(a == 2) && !(b == 3) && flag) { X(out); }\n"
            "int result = measure(out);\n"
        )
        assert _counts(source) == {expected: SHOTS}


def test_phase_outside_qif_sets_the_global_phase() -> None:
    """No relative-phase claim to verify here -- just that it compiles and
    the circuit carries the requested global phase."""
    source = "qbool q = false;\nphase(1.0);\n"
    circuit = _build_circuit(source)
    assert circuit.global_phase == pytest.approx(1.0)


def test_mvp_c_matches_the_condition(mvp_c_source: str) -> None:
    """qif(a == 2) flips out on exactly one of the four equally-likely
    branches of an equal superposition -- ~25% '1'."""
    counts = _counts(mvp_c_source)
    assert set(counts) == {"0", "1"}
    assert SHOTS * 0.15 < counts["1"] < SHOTS * 0.35


def test_quantum_addition_equal_width() -> None:
    source = "qint<2> a = 1;\nqint<2> b = 2;\na += b;\nint result = measure(a);\n"
    assert _counts(source) == {"11": SHOTS}


def test_quantum_addition_wraps_modulo_the_target_width() -> None:
    source = "qint<2> a = 3;\nqint<2> b = 2;\na += b;\nint result = measure(a);\n"
    assert _counts(source) == {"01": SHOTS}


def test_quantum_addition_wider_target_pads_the_narrower_addend() -> None:
    """The padding ancilla must come back clean regardless of the target's
    higher bits, verified here by running the full compiled circuit."""
    source = "qint<3> a = 5;\nqint<2> b = 3;\na += b;\nint result = measure(a);\n"
    assert _counts(source) == {"000": SHOTS}


def test_quantum_addition_narrower_target_uses_the_addends_low_bits() -> None:
    source = "qint<2> a = 1;\nqint<3> b = 7;\na += b;\nint result = measure(a);\n"
    assert _counts(source) == {"00": SHOTS}


def test_quantum_subtraction() -> None:
    source = "qint<2> a = 3;\nqint<2> b = 1;\na -= b;\nint result = measure(a);\n"
    assert _counts(source) == {"10": SHOTS}


def test_quantum_addition_with_a_compile_time_constant() -> None:
    source = "qint<2> a = 1;\na += 3;\nint result = measure(a);\n"
    assert _counts(source) == {"00": SHOTS}


def test_quantum_addition_constant_wraps_silently_on_overflow() -> None:
    """The decision is a silent mod 2^n, the same rule as a too-wide addend."""
    source = "qint<2> a = 1;\na += 6;\nint result = measure(a);\n"
    assert _counts(source) == {"11": SHOTS}


def test_quantum_multiplication_declaration() -> None:
    source = "qint<2> a = 3;\nqint<2> b = 2;\nqint<4> c = a * b;\nint result = measure(c);\n"
    assert _counts(source) == {"0110": SHOTS}


def test_quantum_multiplication_is_truncated_to_the_declared_width() -> None:
    source = "qint<2> a = 3;\nqint<2> b = 3;\nqint<2> c = a * b;\nint result = measure(c);\n"
    assert _counts(source) == {"01": SHOTS}


def test_multiply_accumulate_leaves_the_temp_register_clean() -> None:
    """d += a * b must uncompute its own temporary product -- checked here on
    the statevector, since counts alone cannot see a leftover ancilla."""
    source = "qint<2> a = 2;\nqint<2> b = 3;\nqint<3> d = 1;\nd += a * b;\n"
    circuit = _build_circuit(source)
    statevector = Statevector.from_instruction(circuit)

    (index,) = [i for i, amp in enumerate(statevector.data) if abs(amp) > 1e-9]
    a_bits = index & 0b11
    b_bits = (index >> 2) & 0b11
    d_bits = (index >> 4) & 0b111
    remainder = index >> 7
    assert a_bits == 2
    assert b_bits == 3
    assert d_bits == (1 + 2 * 3) % 8
    assert remainder == 0, "every ancilla (multiplier temp and both helpers) must be |0>"


def test_param_scalar_angle_binds_at_runtime() -> None:
    source = "param float theta;\nqbool q = false;\nRX(theta, q);\n"
    namespace: dict[str, Any] = {}
    result = compile_source(source, source_name="test.slanq")
    exec(result.qiskit_source, namespace)  # noqa: S102
    bound = namespace["build_bound_circuit"](theta=np.pi)
    statevector = Statevector.from_instruction(bound)
    assert abs(statevector.data[0]) == pytest.approx(0.0, abs=1e-9)
    assert abs(statevector.data[1]) == pytest.approx(1.0)


def test_param_array_indexed_angle_binds_at_runtime() -> None:
    source = "param int gamma[2];\nqbool q = false;\nRX(gamma[1], q);\n"
    result = compile_source(source, source_name="test.slanq")
    assert not result.diagnostics.has_errors
    namespace: dict[str, Any] = {}
    exec(result.qiskit_source, namespace)  # noqa: S102
    bound = namespace["build_bound_circuit"](gamma=[0, 3])
    statevector = Statevector.from_instruction(bound)
    expected = Statevector.from_instruction(
        namespace["build_circuit"]().assign_parameters({"gamma_1": 3})
    )
    assert statevector.equiv(expected)


def test_hello_example_leaves_only_the_param_arithmetic_limitation(
    hello_source: str,
) -> None:
    """Classical arithmetic and params eliminate every limitation from
    hello.slanq except adding a runtime param to a quantum variable."""
    result = compile_source(hello_source, source_name="hello.slanq")
    messages = [d.message for d in result.diagnostics.errors]
    assert len(messages) == 1
    assert "runtime parameter" in messages[0]


def test_qif_top_level_negation_composes_with_a_clause_ancilla() -> None:
    """!(a==2 && !(b==3)) == (a!=2) || (b==3) -- De Morgan gives OR-shaped
    conditions for free here, even without general `||` support: the
    top-level ancilla is computed from a pattern that already includes
    another clause's own ancilla."""
    for a_value, b_value in [(2, 0), (2, 3), (1, 1)]:
        expected = (a_value != 2) or (b_value == 3)
        source = (
            f"qint<2> a = {a_value};\nqint<2> b = {b_value};\nqbool out = false;\n"
            "qif(!(a == 2 && !(b == 3))) { X(out); }\nint result = measure(out);\n"
        )
        assert _counts(source) == {"1" if expected else "0": SHOTS}
