"""End-to-end: Slanq source is compiled, executed and run on a simulator."""

from __future__ import annotations

from typing import Any

from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler

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


def _counts(source: str) -> dict[str, int]:
    circuit = _build_circuit(source)
    result = StatevectorSampler(seed=SEED).run([circuit], shots=SHOTS).result()
    return result[0].data.result.get_counts()


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
