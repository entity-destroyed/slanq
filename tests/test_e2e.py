"""End-to-end: Slanq source is compiled, executed and run on a simulator."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Operator, Statevector

from slanq.compiler import compile_source

SHOTS = 1024
SEED = 1234


def _build_circuit(source: str) -> QuantumCircuit:
    result = compile_source(source, source_name="test.slanq")
    assert not result.diagnostics.has_errors
    assert result.qiskit_source is not None

    namespace: dict[str, Any] = {}
    exec(result.qiskit_source, namespace)  # noqa: S102
    circuit = namespace["build_circuit"]()
    # The generated file's own __main__ draws the circuit, so a shape that
    # builds but cannot be drawn is still a program the user cannot run.
    str(circuit)
    return circuit


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
    assert _counts("qbool q = false;\nrt int<> result = measure(q);\n") == {"0": SHOTS}


def test_true_initializer_always_measures_one() -> None:
    assert _counts("qbool q = true;\nrt int<> result = measure(q);\n") == {"1": SHOTS}


def test_double_x_returns_to_zero() -> None:
    source = "qbool q = false;\nX(q);\nX(q);\nrt int<> result = measure(q);\n"
    assert _counts(source) == {"0": SHOTS}


def test_reset_returns_a_flipped_qubit_to_zero() -> None:
    source = "qbool q = true;\nreset(q);\nrt int<> result = measure(q);\n"
    assert _counts(source) == {"0": SHOTS}


def test_reset_collapses_a_superposition_to_zero() -> None:
    """The one thing no unitary can do: the outcome is certain afterwards
    whatever the state was."""
    source = "qbool q = false;\nH(q);\nreset(q);\nrt int<> result = measure(q);\n"
    assert _counts(source) == {"0": SHOTS}


def test_reset_of_one_qubit_leaves_the_others_alone() -> None:
    """Slanq is big-endian: a[0] is the most significant qubit, so resetting
    it takes 3 (0b11) to 1."""
    source = "qint<2> a = 3;\nreset(a[0]);\nrt int<> result = measure(a);\n"
    assert _counts(source) == {"01": SHOTS}


def test_bell_state_outcomes_are_correlated(mvp_b_source: str) -> None:
    counts = _counts(mvp_b_source)
    assert set(counts) == {"00", "11"}
    assert all(SHOTS * 0.3 < value < SHOTS * 0.7 for value in counts.values())


def test_qint_constant_survives_the_endianness_split() -> None:
    """The canary for B1: a mirrored index or bit order would change the value."""
    counts = _counts("qint<3> a = 5; rt int<> result = measure(a);")
    assert counts == {"101": SHOTS}


def test_indexed_gate_targets_the_intended_qubit() -> None:
    """Slanq a[0] is the most significant qubit, so flipping it gives 4, not 1."""
    counts = _counts("qint<3> a = 0; X(a[0]); rt int<> result = measure(a);")
    assert counts == {"100": SHOTS}


def test_last_index_is_the_least_significant_qubit() -> None:
    counts = _counts("qint<3> a = 0; X(a[2]); rt int<> result = measure(a);")
    assert counts == {"001": SHOTS}


def test_pi_rotation_flips_the_qubit() -> None:
    """RX(PI) is a bit flip up to phase; a half of it would not be deterministic."""
    source = "qbool q = false;\nRX(PI, q);\nrt int<> result = measure(q);\n"
    assert _counts(source) == {"1": SHOTS}


def test_a_computed_angle_reaches_the_circuit_unfolded() -> None:
    source = "qbool q = false;\nRX(round(2.5) * PI / floor(7 / 2), q);\n"
    result = compile_source(source, source_name="test.slanq")
    assert result.qiskit_source is not None
    assert "_round(2.5) * np.pi / math.floor(7 / 2)" in result.qiskit_source

    counts = _counts(source + "rt int<> result = measure(q);\n")
    assert counts == {"1": SHOTS}


def test_ccx_broadcast_computes_a_bitwise_and() -> None:
    """CCX over whole registers is one Toffoli per bit, so c becomes a & b."""
    source = (
        "qint<2> a = 3;\nqint<2> b = 2;\nqint<2> c = 0;\n"
        "CCX(a, b, c);\nrt int<> result = measure(c);\n"
    )
    assert _counts(source) == {"10": SHOTS}


def test_empty_probability_list_is_a_cheap_equal_superposition() -> None:
    """[] is meant as Hadamards on every qubit, not a general StatePreparation."""
    source = "qint<2> a = [];\nrt int<> result = measure(a);\n"
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
    source = "qint<2> b = [0, 0, 0.8, 0.2];\nrt int<> result = measure(b);\n"
    counts = _counts(source)
    assert set(counts) == {"10", "11"}
    assert counts["10"] > counts["11"]
    assert SHOTS * 0.7 < counts["10"] < SHOTS * 0.9


def test_probability_list_needing_normalization_still_runs() -> None:
    """A warning does not stop compilation, and the normalized amplitudes
    still describe a valid, correctly-proportioned distribution."""
    source = "qint<2> c = [0.1, 0.1, 0.1, 0.1];\nrt int<> result = measure(c);\n"
    result = compile_source(source, source_name="test.slanq")
    assert result.diagnostics.warnings
    assert not result.diagnostics.has_errors

    counts = _counts(source)
    assert set(counts) == {"00", "01", "10", "11"}
    assert all(SHOTS * 0.15 < value < SHOTS * 0.35 for value in counts.values())


def test_qbool_probability_list() -> None:
    source = "qbool q = [0.2, 0.8];\nrt int<> result = measure(q);\n"
    counts = _counts(source)
    assert set(counts) == {"0", "1"}
    assert counts["1"] > counts["0"]


def test_amplitude_list_produces_the_exact_complex_state() -> None:
    """Unlike a [] probability list, {} can encode a relative phase -- counts
    alone cannot see this, so the statevector is checked directly."""
    source = "qbool q = {1/sqrt(2), 1i/sqrt(2)};\n"
    circuit = _build_circuit(source)
    statevector = Statevector.from_instruction(circuit)
    expected = np.array([1 / np.sqrt(2), 1j / np.sqrt(2)])
    assert np.allclose(statevector.data, expected)


def test_amplitude_list_with_negative_real_amplitude() -> None:
    """[] cannot express this state at all -- a negative amplitude has no
    corresponding (nonnegative) probability representation."""
    source = "qbool q = {1/sqrt(2), -1/sqrt(2)};\n"
    circuit = _build_circuit(source)
    statevector = Statevector.from_instruction(circuit)
    expected = np.array([1 / np.sqrt(2), -1 / np.sqrt(2)])
    assert np.allclose(statevector.data, expected)


def test_empty_amplitude_list_is_equal_superposition() -> None:
    source = "qint<2> a = {};\nrt int<> result = measure(a);\n"
    result = compile_source(source, source_name="test.slanq")
    assert result.qiskit_source is not None
    assert "StatePreparation" not in result.qiskit_source

    counts = _counts(source)
    assert set(counts) == {"00", "01", "10", "11"}
    assert all(SHOTS * 0.15 < value < SHOTS * 0.35 for value in counts.values())


def test_amplitude_list_needing_normalization_still_runs() -> None:
    source = "qbool q = {0.6, 0.6};\nrt int<> result = measure(q);\n"
    result = compile_source(source, source_name="test.slanq")
    assert result.diagnostics.warnings
    assert not result.diagnostics.has_errors

    counts = _counts(source)
    assert set(counts) == {"0", "1"}
    assert SHOTS * 0.35 < counts["0"] < SHOTS * 0.65


def test_amplitude_list_needing_normalization_with_a_complex_value() -> None:
    """The normalization warning and the correction it triggers must both work
    when the list contains a complex value, not just an all-real one."""
    source = "qbool q = {0.6i, 0.6};\nrt int<> result = measure(q);\n"
    result = compile_source(source, source_name="test.slanq")
    assert result.diagnostics.warnings
    assert not result.diagnostics.has_errors

    counts = _counts(source)
    assert set(counts) == {"0", "1"}
    assert SHOTS * 0.35 < counts["0"] < SHOTS * 0.65


def test_amplitude_list_element_from_a_classical_name() -> None:
    """The motivating gap for the const_value Name-resolution fix: a
    classical constant, and arithmetic on it, used as a {} element."""
    source = "complex c = 0.5+0.3i;\nqbool q = {c, 1 - c};\n"
    result = compile_source(source, source_name="test.slanq")
    assert result.diagnostics.warnings  # not normalized
    assert not result.diagnostics.has_errors

    circuit = _build_circuit(source)
    statevector = Statevector.from_instruction(circuit)
    raw = np.array([0.5 + 0.3j, 0.5 - 0.3j])
    expected = raw / np.linalg.norm(raw)
    assert np.allclose(statevector.data, expected)


def test_amplitude_list_wider_than_two_elements() -> None:
    """Every other {} test uses a 2-element (qbool) list -- this checks a
    4-element (qint<2>) list with a mix of real and complex amplitudes."""
    source = "qint<2> a = {0.5, 0.5i, 0.5, 0.5i};\n"
    circuit = _build_circuit(source)
    statevector = Statevector.from_instruction(circuit)
    expected = np.array([0.5, 0.5j, 0.5, 0.5j])
    assert np.allclose(statevector.data, expected)


def test_qif_equality_condition_fires_only_on_match() -> None:
    for a_value, expected in [(0, "0"), (1, "0"), (2, "1"), (3, "0")]:
        source = (
            f"qint<2> a = {a_value};\nqbool out = false;\n"
            "qif(a == 2) { X(out); }\nrt int<> result = measure(out);\n"
        )
        assert _counts(source) == {expected: SHOTS}


def test_qif_negated_condition_fires_on_mismatch() -> None:
    for a_value, expected in [(0, "1"), (1, "1"), (2, "0"), (3, "1")]:
        source = (
            f"qint<2> a = {a_value};\nqbool out = false;\n"
            "qif(!(a == 2)) { X(out); }\nrt int<> result = measure(out);\n"
        )
        assert _counts(source) == {expected: SHOTS}


def test_qif_and_chain_across_two_registers() -> None:
    source = (
        "qint<2> a = 2;\nqbool flag = true;\nqbool out = false;\n"
        "qif(a == 2 && flag) { X(out); }\nrt int<> result = measure(out);\n"
    )
    assert _counts(source) == {"1": SHOTS}

    source_false = (
        "qint<2> a = 2;\nqbool flag = false;\nqbool out = false;\n"
        "qif(a == 2 && flag) { X(out); }\nrt int<> result = measure(out);\n"
    )
    assert _counts(source_false) == {"0": SHOTS}


def test_qif_two_qubit_gate_in_body_respects_endianness() -> None:
    """CX(a[0], b[0]) inside a qif body must mirror exactly as it would at the
    top level -- this is the canary for the sub-circuit reusing the same
    operand renderer as the outer one."""
    source = (
        "qint<2> a = 2;\nqint<2> b = 0;\nqbool ctrl = true;\n"
        "qif(ctrl) { CX(a[0], b[0]); }\nrt int<> result = measure(b);\n"
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


def test_an_empty_qif_body_adds_nothing_to_the_circuit() -> None:
    circuit = _build_circuit("qint<2> a = 2;\nqif(a == 2) { }\n")
    assert [instruction.operation.name for instruction in circuit.data] == ["x"]


def test_a_qif_phase_body_draws() -> None:
    """A controlled global phase used to become a 0-wire sub-circuit, which
    builds and runs but cannot be drawn -- and drawing is what the generated
    file does when it is run."""
    circuit = _build_circuit("qint<2> a = 2;\nqif(a == 2) { phase(1.5); }\n")
    assert "qif_body" not in str(circuit)


def _qif_body_source(body: str) -> str:
    return f"qint<2> a = 2;\nqbool t = false;\nqif(a == 2) {{ {body} }}\n"


def _qif_body_reference(items: list[tuple[str, float | None]]) -> QuantumCircuit:
    """`qint<2> a = 2; qbool t = false; qif(a == 2) {...}` with every body item
    controlled on its own, in written order -- so a phase sits exactly where
    the source puts it. A phase is controlled here the way Qiskit itself does
    it, by controlling a circuit whose only content is a global phase: that
    gate cannot be drawn, which is why the compiler does not emit it, but its
    semantics are the definition the compiler's phase gate has to match."""
    a = QuantumRegister(2, "a")
    t = QuantumRegister(1, "t")
    circuit = QuantumCircuit(a, t)
    circuit.x(a[1])
    for name, angle in items:
        if name == "phase":
            phase = QuantumCircuit(0, global_phase=angle)
            circuit.append(
                phase.to_gate().control(2, ctrl_state=2, annotated=False), [*a]
            )
            continue
        body = QuantumCircuit(1)
        getattr(body, name)(0)
        circuit.append(
            body.to_gate().control(2, ctrl_state=2, annotated=False), [*a, t[0]]
        )
    return circuit


@pytest.mark.parametrize(
    ("body", "items"),
    [
        ("X(t); phase(PI / 3);", [("x", None), ("phase", np.pi / 3)]),
        ("phase(PI / 3); X(t);", [("phase", np.pi / 3), ("x", None)]),
        (
            "T(t); phase(PI / 3); S(t);",
            [("t", None), ("phase", np.pi / 3), ("s", None)],
        ),
        (
            "H(t); X(t); phase(PI / 3);",
            [("h", None), ("x", None), ("phase", np.pi / 3)],
        ),
        (
            "X(t); phase(PI / 3); H(t); phase(PI / 5);",
            [
                ("x", None),
                ("phase", np.pi / 3),
                ("h", None),
                ("phase", np.pi / 5),
            ],
        ),
    ],
)
def test_a_qif_body_phase_acts_where_it_is_written(
    body: str, items: list[tuple[str, float | None]]
) -> None:
    """The phase is emitted after the body's sub-circuit whatever the body
    looks like, so it has to agree with a reference that applies it in place.
    The gates around it do not commute (T then S, H then X), so the body's own
    order shows up here too. Where the phase sits cannot change the operator
    -- the next test says why."""
    compiled = Operator(_build_circuit(_qif_body_source(body)))
    assert compiled == Operator(_qif_body_reference(items))


def test_a_qif_body_phase_may_sit_anywhere_in_the_body() -> None:
    """Moving the phase inside the body cannot change the circuit. A phase
    commutes with anything that maps the states its condition matches onto
    itself, and a body gate could only fail that by targeting a condition
    qubit -- which a qif body may not do. The hoisting rests on that rule,
    which is why it is worth pinning down here."""
    circuits = [
        Operator(_build_circuit(_qif_body_source(body)))
        for body in (
            "phase(PI / 3); H(t); X(t);",
            "H(t); phase(PI / 3); X(t);",
            "H(t); X(t); phase(PI / 3);",
        )
    ]
    assert circuits[0] == circuits[1] == circuits[2]


def test_qif_not_equal_matches_the_negated_condition() -> None:
    for a_value, expected in [(0, "1"), (1, "1"), (2, "0"), (3, "1")]:
        source = (
            f"qint<2> a = {a_value};\nqbool out = false;\n"
            "qif(a != 2) { X(out); }\nrt int<> result = measure(out);\n"
        )
        assert _counts(source) == {expected: SHOTS}


def test_qif_equality_accepts_reversed_operands() -> None:
    source = (
        "qint<2> a = 2;\nqbool out = false;\n"
        "qif(2 == a) { X(out); }\nrt int<> result = measure(out);\n"
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
            "rt int<> result = measure(out);\n"
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
    source = "qint<2> a = 1;\nqint<2> b = 2;\na += b;\nrt int<> result = measure(a);\n"
    assert _counts(source) == {"11": SHOTS}


def test_quantum_addition_wraps_modulo_the_target_width() -> None:
    source = "qint<2> a = 3;\nqint<2> b = 2;\na += b;\nrt int<> result = measure(a);\n"
    assert _counts(source) == {"01": SHOTS}


def test_quantum_addition_wider_target_pads_the_narrower_addend() -> None:
    """The padding ancilla must come back clean regardless of the target's
    higher bits, verified here by running the full compiled circuit."""
    source = "qint<3> a = 5;\nqint<2> b = 3;\na += b;\nrt int<> result = measure(a);\n"
    assert _counts(source) == {"000": SHOTS}


def test_quantum_addition_narrower_target_uses_the_addends_low_bits() -> None:
    source = "qint<2> a = 1;\nqint<3> b = 7;\na += b;\nrt int<> result = measure(a);\n"
    assert _counts(source) == {"00": SHOTS}


def test_quantum_subtraction() -> None:
    source = "qint<2> a = 3;\nqint<2> b = 1;\na -= b;\nrt int<> result = measure(a);\n"
    assert _counts(source) == {"10": SHOTS}


def test_quantum_addition_with_a_compile_time_constant() -> None:
    source = "qint<2> a = 1;\na += 3;\nrt int<> result = measure(a);\n"
    assert _counts(source) == {"00": SHOTS}


def test_quantum_addition_constant_wraps_silently_on_overflow() -> None:
    """The decision is a silent mod 2^n, the same rule as a too-wide addend."""
    source = "qint<2> a = 1;\na += 6;\nrt int<> result = measure(a);\n"
    assert _counts(source) == {"11": SHOTS}


def test_quantum_multiplication_declaration() -> None:
    source = "qint<2> a = 3;\nqint<2> b = 2;\nqint<4> c = a * b;\nrt int<> result = measure(c);\n"
    assert _counts(source) == {"0110": SHOTS}


def test_quantum_multiplication_is_truncated_to_the_declared_width() -> None:
    source = "qint<2> a = 3;\nqint<2> b = 3;\nqint<2> c = a * b;\nrt int<> result = measure(c);\n"
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


@pytest.mark.parametrize("width", [2, 3, 4, 5])
def test_multiply_accumulate_works_at_every_target_width(width: int) -> None:
    program = (
        f"qint<2> a = 2;\nqint<2> b = 3;\nqint<{width}> c = 0;\n"
        "c += a * b;\nrt int<> result = measure(c);\n"
    )
    expected = format((2 * 3) % (1 << width), f"0{width}b")
    assert _counts(program) == {expected: SHOTS}


def test_repeated_multiply_accumulate_reuses_its_registers() -> None:
    """The second `c += a * b` allocates its own ancillas; nothing may be
    declared twice, and every temporary must still come back to |0>."""
    circuit = _build_circuit(
        "qint<1> a = 1;\nqint<1> b = 1;\nqint<2> c = 0;\nc += a * b;\nc += a * b;\n"
    )
    statevector = Statevector.from_instruction(circuit)
    (index,) = [i for i, amp in enumerate(statevector.data) if abs(amp) > 1e-9]
    assert index & 0b1 == 1
    assert (index >> 1) & 0b1 == 1
    assert (index >> 2) & 0b11 == 2
    assert index >> 4 == 0, "every ancilla of both statements must be |0>"


def test_a_non_ascii_name_survives_into_the_circuit() -> None:
    """A name beyond ASCII has to reach the whole way: the registers in the
    generated file are named exactly as the program wrote them."""
    source = (
        "qint<3> ψ = 5;\nfor(int i in range(3)) { X(ψ[i]); }\n"
        "rt int<> result = measure(ψ);\n"
    )
    assert _counts(source) == {"010": SHOTS}


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


def test_process_call_runs_the_body_on_the_arguments() -> None:
    """The section 3.4 example, end to end: 1 + 2 == 3."""
    source = (
        "process add(qint x, qint y) { x += y; }\n"
        "qint<2> num1 = 1;\nqint<2> num2 = 2;\nadd(num1, num2);\n"
        "rt int<> result = measure(num1);\n"
    )
    assert _counts(source) == {"11": SHOTS}


def test_a_process_is_equivalent_to_writing_the_body_out() -> None:
    inlined = _build_circuit(
        "process add(qint x, qint y) { x += y; }\n"
        "qint<3> a = 5;\nqint<2> b = 2;\nadd(a, b);\n"
    )
    written = _build_circuit("qint<3> a = 5;\nqint<2> b = 2;\na += b;\n")
    assert Statevector.from_instruction(inlined).equiv(
        Statevector.from_instruction(written)
    )


def test_two_calls_of_one_process_both_take_effect() -> None:
    source = (
        "process bump(qint x) { x += 1; }\n"
        "qint<2> a = 0;\nbump(a);\nbump(a);\nrt int<> result = measure(a);\n"
    )
    assert _counts(source) == {"10": SHOTS}


def test_a_process_parameter_may_be_a_single_qubit() -> None:
    source = (
        "process flip(qbool b) { X(b); }\n"
        "qint<2> a = 0;\nflip(a[1]);\nrt int<> result = measure(a);\n"
    )
    assert _counts(source) == {"01": SHOTS}


def test_a_classical_process_parameter_reaches_the_generated_file_unfolded() -> None:
    """A classical parameter is substituted as an expression, so the "never
    fold a compile-time expression" rule still holds inside a body."""
    result = compile_source(
        "process rot(qbool q, float angle) { RX(angle, q); }\n"
        "qbool t = false;\nrot(t, PI / 2);\n",
        source_name="test.slanq",
    )
    assert result.qiskit_source is not None
    assert "np.pi / 2" in result.qiskit_source


def test_a_qif_inside_a_process_body_still_controls_the_body() -> None:
    source = (
        "process guarded(qint a, qbool out) { qif(a == 2) { X(out); } }\n"
        "qint<2> a = 2;\nqbool out = false;\nguarded(a, out);\n"
        "rt int<> result = measure(out);\n"
    )
    assert _counts(source) == {"1": SHOTS}


def test_a_process_call_inside_a_qif_body_is_controlled() -> None:
    """Expansion turns the call into plain gates before lowering sees the qif
    body, which is what lets a qif body contain a call at all -- a qif body
    must always resolve to gates, since a control-flow op cannot be a gate."""
    source = (
        "process flip(qbool b) { X(b); }\n"
        "qint<2> a = 2;\nqbool out = false;\nqif(a == 2) { flip(out); }\n"
        "rt int<> result = measure(out);\n"
    )
    assert _counts(source) == {"1": SHOTS}


def test_a_classical_parameter_can_index_a_qubit() -> None:
    source = (
        "process poke(qint q, int i) { X(q[i]); }\n"
        "qint<2> a = 0;\npoke(a, 1);\nrt int<> result = measure(a);\n"
    )
    assert _counts(source) == {"01": SHOTS}


def test_a_nested_process_call_reaches_the_circuit() -> None:
    source = (
        "process inner(qbool x) { X(x); }\nprocess outer(qbool y) { inner(y); }\n"
        "qbool q = false;\nouter(q);\nrt int<> result = measure(q);\n"
    )
    assert _counts(source) == {"1": SHOTS}


def test_qif_controls_one_qubit_of_a_register_on_another() -> None:
    """Condition and body use different qubits of the SAME register --
    legal, and the compiler used to emit a file that died on `duplicate bit
    arguments` when the generated circuit was built."""
    for value, expected in [(2, "11"), (0, "00")]:
        source = (
            f"qint<2> a = {value};\n"
            "qif(a[0]) { X(a[1]); }\n"
            "rt int<> result = measure(a);\n"
        )
        assert _counts(source) == {expected: SHOTS}


def test_qif_controls_across_a_wider_register() -> None:
    source = (
        "qint<3> a = 4;\n"
        "qif(a[0]) { X(a[2]); }\n"
        "rt int<> result = measure(a);\n"
    )
    assert _counts(source) == {"101": SHOTS}


def test_qif_top_level_negation_composes_with_a_clause_ancilla() -> None:
    """!(a==2 && !(b==3)) == (a!=2) || (b==3) -- De Morgan gives OR-shaped
    conditions for free here, even without general `||` support: the
    top-level ancilla is computed from a pattern that already includes
    another clause's own ancilla."""
    for a_value, b_value in [(2, 0), (2, 3), (1, 1)]:
        expected = (a_value != 2) or (b_value == 3)
        source = (
            f"qint<2> a = {a_value};\nqint<2> b = {b_value};\nqbool out = false;\n"
            "qif(!(a == 2 && !(b == 3))) { X(out); }\nrt int<> result = measure(out);\n"
        )
        assert _counts(source) == {"1" if expected else "0": SHOTS}


def test_a_loop_flips_every_qubit_it_indexes() -> None:
    source = (
        "qint<3> a = 0;\nfor(int i in range(3)) { X(a[i]); }\n"
        "rt int<> result = measure(a);\n"
    )
    assert _counts(source) == {"111": SHOTS}


def test_a_loop_is_equivalent_to_writing_the_body_out() -> None:
    """Bit-identical circuits, which is why unrolling costs nothing the Qiskit
    transpiler could have saved: it sees the same flat circuit either way."""
    looped = _build_circuit("qint<3> a = 0;\nfor(int i in range(3)) { X(a[i]); }\n")
    written = _build_circuit("qint<3> a = 0;\nX(a[0]);\nX(a[1]);\nX(a[2]);\n")
    assert looped == written


def test_arithmetic_accumulates_over_the_iterations() -> None:
    source = (
        "qint<4> num = 0;\nfor(int i in range(1, 4)) { num += i; }\n"
        "rt int<> result = measure(num);\n"
    )
    assert _counts(source) == {"0110": SHOTS}


def test_a_triangular_loop_repeats_the_right_number_of_times() -> None:
    """i = 0 adds nothing, i = 1 flips a[0], i = 2 flips a[0] and a[1] -- so
    a[0] is flipped twice and cancels, leaving the value 2."""
    source = (
        "qint<3> a = 0;\n"
        "for(int i in range(3)) { for(int j in range(i)) { X(a[j]); } }\n"
        "rt int<> result = measure(a);\n"
    )
    assert _counts(source) == {"010": SHOTS}


def test_a_process_called_in_a_loop_runs_once_per_iteration() -> None:
    source = (
        "process bump(qint x) { x += 1; }\nqint<3> num = 0;\n"
        "for(int i in range(2)) { bump(num); }\nrt int<> result = measure(num);\n"
    )
    assert _counts(source) == {"010": SHOTS}


def test_a_loop_inside_a_process_body_reaches_the_circuit() -> None:
    source = (
        "process flip(qint x) { for(int i in range(3)) { X(x[i]); } }\n"
        "qint<3> a = 0;\nflip(a);\nrt int<> result = measure(a);\n"
    )
    assert _counts(source) == {"111": SHOTS}


def test_a_loop_in_a_qif_body_is_controlled_by_the_condition() -> None:
    on = (
        "qint<2> c = 3;\nqint<2> t = 0;\n"
        "qif(c == 3) { for(int i in range(2)) { X(t[i]); } }\n"
        "rt int<> result = measure(t);\n"
    )
    off = on.replace("qint<2> c = 3;", "qint<2> c = 1;")
    assert _counts(on) == {"11": SHOTS}
    assert _counts(off) == {"00": SHOTS}


def test_the_loop_value_reaches_an_angle_unfolded() -> None:
    """`i * PI / 8` keeps its shape in the generated file, the same way a
    process argument does -- the compiler substitutes, it does not evaluate."""
    result = compile_source("qbool q = false;\nfor(int i in range(3)) { RZ(i * PI / 8, q); }\n")
    assert result.qiskit_source is not None
    assert "circuit.rz(0 * np.pi / 8, q[0])" in result.qiskit_source
    assert "circuit.rz(1 * np.pi / 8, q[0])" in result.qiskit_source
    assert "circuit.rz(2 * np.pi / 8, q[0])" in result.qiskit_source


def test_a_classical_parameter_can_drive_a_loop_in_a_process_body() -> None:
    source = (
        "process flip(int n, qint x) { for(int i in range(n)) { X(x[i]); } }\n"
        "qint<3> a = 0;\nflip(3, a);\nrt int<> result = measure(a);\n"
    )
    assert _counts(source) == {"111": SHOTS}


def test_two_calls_may_ask_for_different_iteration_counts() -> None:
    """Each call site unrolls with its own count, which a single shared loop in
    the generated circuit could not express. Slanq indexes big-endian, so one
    flipped qubit is the most significant bit."""
    source = (
        "process flip(int n, qint x) { for(int i in range(n)) { X(x[i]); } }\n"
        "qint<3> a = 0;\nqint<3> b = 0;\nflip(1, a);\nflip(3, b);\n"
        "rt int<> result = measure(a);\nrt int<> other = measure(b);\n"
    )
    assert _counts(source) == {"100": SHOTS}
    assert _counts(source, register="other") == {"111": SHOTS}


def test_the_taken_branch_is_the_only_one_built() -> None:
    source = (
        "qbool q = false;\nbool t = true;\n"
        "if (t) { X(q); } else { H(q); }\nrt int<> result = measure(q);\n"
    )
    assert _counts(source) == {"1": SHOTS}


def test_the_untaken_branch_contributes_nothing() -> None:
    source = (
        "qbool q = false;\nbool t = false;\n"
        "if (t) { X(q); }\nrt int<> result = measure(q);\n"
    )
    assert _counts(source) == {"0": SHOTS}


def test_an_else_if_chain_picks_one_branch() -> None:
    source = (
        "qint<2> a = 0;\nint n = 5;\n"
        "if (n > 9) { X(a[0]); } else if (n > 3) { X(a[1]); } else { H(a); }\n"
        "rt int<> result = measure(a);\n"
    )
    assert _counts(source) == {"01": SHOTS}


_PHASE_DECLS = "qint<2> a = 0;\nqbool f = false;\nbool t = true;\n"


@pytest.mark.parametrize(
    ("written", "equivalent"),
    [
        (
            "qif(a == 2) { if (t) { phase(PI/3); } }",
            "qif(a == 2) { phase(PI/3); }",
        ),
        (
            "qif(a == 2) { X(f); if (!t) { phase(PI/3); } }",
            "qif(a == 2) { X(f); }",
        ),
        (
            "qif(a == 2) { if (!t) { phase(PI/3); } else { phase(PI/4); } }",
            "qif(a == 2) { phase(PI/4); }",
        ),
        (
            "qif(a == 2) { phase(PI/6); if (t) { phase(PI/3); } }",
            "qif(a == 2) { phase(PI/6 + PI/3); }",
        ),
        (
            "qif(!(a == 2)) { X(f); if (t) { phase(PI/3); } }",
            "qif(!(a == 2)) { X(f); phase(PI/3); }",
        ),
    ],
    ids=["taken", "untaken", "else", "added to an unconditional one", "negated qif"],
)
def test_a_conditional_phase_matches_the_branch_it_describes(
    written: str, equivalent: str
) -> None:
    """Exact operator equality, not equivalence: a phase inside a qif is a
    relative phase, so an overall factor would be a different circuit."""
    assert Operator(_build_circuit(_PHASE_DECLS + written + "\n")) == Operator(
        _build_circuit(_PHASE_DECLS + equivalent + "\n")
    )


TELEPORTATION = """qbool msg = true;
qbool alice = false;
qbool bob = false;
H(alice);
CX(alice, bob);
CX(msg, alice);
H(msg);
rt int<> m1 = measure(msg);
rt int<> m2 = measure(alice);
rt if (m2 == 1) { X(bob); }
rt if (m1 == 1) { Z(bob); }
rt int<> result = measure(bob);
"""


def _realtime_counts(source: str) -> dict[str, int]:
    """Run on Aer rather than StatevectorSampler: a feedforward branch is not
    a unitary, so it needs a simulator that executes control flow."""
    circuit = _build_circuit(source)
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    simulator = AerSimulator()
    counts = simulator.run(transpile(circuit, simulator), shots=SHOTS).result().get_counts()
    return {key.split()[0]: value for key, value in counts.items()}


def test_teleportation_delivers_the_state() -> None:
    """The two Pauli corrections are the textbook use of a feedforward branch:
    whichever pair of outcomes the measurements give, bob ends up in |1>."""
    assert set(_realtime_counts(TELEPORTATION)) == {"1"}


@pytest.mark.parametrize(
    ("program", "expected"),
    [
        (
            "qint<2> a = 0;\nqbool f = false;\nrt int<> m = measure(a);\n"
            "rt if (m == 3) { X(f); } else { X(f); X(f); }\n"
            "rt int<> result = measure(f);\n",
            "0",
        ),
        (
            "qint<2> a = 2;\nqint<2> out = 0;\nrt int<> m = measure(a);\n"
            "rt if (m == 0) { X(out[0]); } else rt if (m == 2) { X(out[1]); }"
            " else { H(out); }\nrt int<> result = measure(out);\n",
            "01",
        ),
        (
            "qint<2> a = 3;\nqbool f = false;\nrt int<> m = measure(a);\n"
            "rt if (m > 1 && m < 4) { X(f); }\nrt int<> result = measure(f);\n",
            "1",
        ),
        (
            "qint<2> a = 3;\nqbool f = false;\nbool off = false;\n"
            "rt int<> m = measure(a);\nrt if (m > 1 && off) { X(f); }\n"
            "rt int<> result = measure(f);\n",
            "0",
        ),
        (
            "qint<2> a = 3;\nqbool f = false;\nrt int<> m = measure(a);\n"
            "rt if (m < 100) { X(f); }\nrt int<> result = measure(f);\n",
            "1",
        ),
        (
            "qbool q = true;\nqint<2> a = 3;\nrt int<> m = measure(a);\n"
            "rt if (m == 3) { reset(q); }\nrt int<> result = measure(q);\n",
            "0",
        ),
    ],
    ids=[
        "else branch",
        "chain picks one arm",
        "and of two real-time tests",
        "a build-time clause narrows it",
        "a literal wider than the register",
        "reset inside a branch",
    ],
)
def test_a_real_time_branch_runs_the_arm_it_describes(
    program: str, expected: str
) -> None:
    assert set(_realtime_counts(program)) == {expected}


def test_a_tracked_index_addresses_the_qubit_it_names() -> None:
    """A hand-rolled loop: each `X` must land on the qubit the index held at
    that point, not on the one the declaration started with."""
    source = (
        "qint<3> a = 0;\nint i = 0;\n"
        "X(a[i]);\ni = i + 1;\nX(a[i]);\n"
        "rt int<> result = measure(a);\n"
    )
    assert _counts(source) == {"110": SHOTS}


def test_an_assigned_angle_reaches_the_gate() -> None:
    source = (
        "qbool q = false;\nfloat g = PI;\ng = g / 2;\nRX(g, q);\n"
        "rt int<> result = measure(q);\n"
    )
    counts = _counts(source)
    assert set(counts) == {"0", "1"}
    assert all(SHOTS * 0.3 < value < SHOTS * 0.7 for value in counts.values())


def test_re_measuring_overwrites_the_same_register() -> None:
    source = (
        "qbool q = false;\nrt int<> result = measure(q);\nX(q);\n"
        "result = measure(q);\n"
    )
    assert _realtime_counts(source) == {"1": SHOTS}


def test_a_build_time_loop_matches_the_same_loop_written_as_a_for() -> None:
    """The two ways of writing a counted repetition must give the same circuit,
    so a fault in one is not hidden by the same fault in the other."""
    written_as_for = _build_circuit(
        "qbool q = false;\nfor(int i in range(3)) { H(q); }\n"
    )
    written_as_while = _build_circuit(
        "qbool q = false;\nint i = 0;\nwhile (i < 3) { H(q); i = i + 1; }\n"
    )
    assert Operator(written_as_for) == Operator(written_as_while)


def test_a_build_time_loop_repeats_its_body() -> None:
    """Three X gates on one qubit leave it flipped; two would not."""
    source = (
        "qbool q = false;\nint i = 0;\nwhile (i < 3) { X(q); i = i + 1; }\n"
        "rt int<> result = measure(q);\n"
    )
    assert _counts(source) == {"1": SHOTS}


def test_a_build_time_break_stops_the_loop() -> None:
    """Two passes instead of three, so the qubit comes back to zero."""
    source = (
        "qbool q = false;\nint i = 0;\n"
        "while (i < 3) { X(q); i = i + 1; if (i > 1) { break; } }\n"
        "rt int<> result = measure(q);\n"
    )
    assert _counts(source) == {"0": SHOTS}


def test_a_build_time_continue_skips_the_rest_of_the_body() -> None:
    """Three passes, one of them skipping the gate: two X gates, back to zero."""
    source = (
        "qbool q = false;\nint i = 0;\n"
        "while (i < 3) { i = i + 1; if (i == 2) { continue; } X(q); }\n"
        "rt int<> result = measure(q);\n"
    )
    assert _counts(source) == {"0": SHOTS}


def test_a_build_time_loop_cannot_index_with_its_counter() -> None:
    """Unlike `for`, which is unrolled into one copy per iteration, a `while`
    body is written once and repeated -- so the counter has no one value there,
    and the compiler says which loop took it away."""
    result = compile_source(
        "qint<3> a = 0;\nint i = 0;\nwhile (i < 3) { X(a[i]); i = i + 1; }\n",
        source_name="test.slanq",
    )
    (diagnostic,) = result.diagnostics.errors
    assert diagnostic.message == (
        "a quantum register index must be an integer the compiler can compute"
        " -- 'i': its value here depends on a branch"
    )


REPEAT_UNTIL_SUCCESS = """qbool coin = false;
H(coin);
rt int<> m = measure(coin);
rt while (m == 1) {
    reset(coin);
    H(coin);
    m = measure(coin);
}
rt int<> result = measure(coin);
"""


def test_repeat_until_success_always_ends_on_the_wanted_outcome() -> None:
    """The loop keeps flipping until the coin lands on 0, so no shot can end
    anywhere else -- the thing a unitary circuit cannot express."""
    assert set(_realtime_counts(REPEAT_UNTIL_SUCCESS)) == {"0"}


def test_a_real_time_break_leaves_the_loop() -> None:
    source = (
        "qbool coin = false;\nH(coin);\nrt int<> m = measure(coin);\n"
        "rt while (m == 1) { reset(coin); H(coin); m = measure(coin);"
        " rt if (m == 0) { break; } }\n"
        "rt int<> result = measure(coin);\n"
    )
    assert set(_realtime_counts(source)) == {"0"}
