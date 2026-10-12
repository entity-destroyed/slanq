# Slanq

**A high-level quantum programming language that compiles to Qiskit.**

Slanq asks what a quantum program would look like if it followed the programmer's
intent instead of the shape of the circuit. Two ideas carry the design:

**Phase-aware control flow.** A branch that shapes the circuit, a branch coherently
controlled by a quantum register, and a branch that reacts to a measurement within
coherence time are three different things that look identical in a circuit-level
description. Slanq gives each its own syntax, so the one you meant is the one you get.

**An affine type discipline on quantum variables.** The no-cloning theorem is enforced
by the type system rather than by convention, and uncomputing ancilla qubits - the step
whose omission silently corrupts a result, with no error anywhere - is the compiler's
job, not the programmer's.

> **Status.** A working compiler, under active development. The full pipeline runs
> end to end - source to runnable Qiskit - and is covered by 1,100+ automated tests,
> but the language is still growing and the surface is not stable yet.

> **Paper.** This work was submitted to the Students' Scientific Conference (TDK) at
> the Budapest University of Technology and Economics, presented on 18 November 2026.
> A preprint will follow on arXiv (cs.PL), and the link will be posted here.

## The language in two examples

### Control a block on the state of a quantum register

<table>
<tr><th align="left">Slanq</th><th align="left">Compiles to</th></tr>
<tr valign="top">
<td>

```slanq
qint<3> a = [];   // all 8 values
qbool out = false;

qif (a == 5) { X(out); }

rt int<> result = measure(out);
```

</td>
<td>

```
          ┌───┐
     a_0: ┤ H ├─────■─────────
          ├───┤     │
     a_1: ┤ H ├─────o─────────
          ├───┤     │
     a_2: ┤ H ├─────■─────────
          └───┘┌────┴─────┐┌─┐
     out: ─────┤ qif_body ├┤M├
               └──────────┘└╥┘
result: 1/══════════════════╩═
                            0
```

</td>
</tr>
</table>

`qif`'s condition is a control-state pattern over physical qubits, and its body is restricted to unitary
transformations - so the compiler can turn the whole block into a single controlled gate
instead of a branch. `measure` and `reset` inside a `qif` are rejected with a diagnostic
that says so permanently: they aren't unitary.

### React to a measurement, inside coherence time

```slanq
process entangle(qubit[] register) {
    H(register[0]);
    for (int i in range(2)) {
        CX(register[i], register[i + 1]);
    }
}

qubit[3] ghz;
entangle(ghz);

rt bool[] outcome = measure(ghz);
rt if (outcome[0]) {          // emitted as a Qiskit if_test block
    X(ghz[0]);
    X(ghz[1]);
    X(ghz[2]);
}
```

The `rt` marker means *quantum runtime*: this branch runs on the classical processor
beside the QPU, within a single shot. The unmarked `for` above it is decided at compile
time and never reaches the circuit - it is already unrolled by the time the Qiskit module
is written out.

## How it compiles

```
parser → transformer → analysis → lowering → (analysis / passes over the IR) → codegen
```

Lark/EBNF grammar to parse tree, parse tree to AST, semantic analysis, lowering to an
intermediate representation, optimisation passes over that IR, then Qiskit Python source.

**Architectural rule.** `analysis.py` only derives facts about the IR; all IR mutation
happens in `passes.py`. The split is enforced by convention rather than by a directory
boundary, so it can later be promoted to a Qiskit-style `PassManager` without
restructuring any code.

A few decisions that shape the output:

- **The target is today's noisy hardware.** Adding a constant to a 5-qubit
  register costs 5 qubits and 138 native operations through a QFT adder
  against 11 qubits and 70 T gates through a ripple-carry adder - the two targets rank
  the same algorithm in opposite orders. Slanq picks the former, and the choice will
  become reversible through a `--target` switch.
- **The source says when each decision is made.** An unmarked construct is decided at
  compile time and never reaches the circuit; a `param` is bound at `assign_parameters`;
  an `rt` construct runs on the QPU's classical companion, within a single shot.
- **Slanq registers are big-endian, Qiskit's are not.** `a[0]` is the most significant bit;
  the compiler does the index transformation so the arithmetic primitives see what they
  expect.
- **Inside a `qif`, only what leaves a trace gets controlled.** A `c += d * e` there
  computes the product into an ancilla, adds it into `c`, then runs the multiplier
  backwards; only the write into `c` is controlled, because the compute/uncompute pair
  returns the ancilla to |0⟩ and leaves its inputs untouched whichever way the condition
  goes. Measured: **173 operations instead of 2,069**, verified against a superposed
  condition rather than a basis state, since a basis state would give the right answer
  even if the reasoning were wrong. The compiler also warns that controlled arithmetic
  costs far more than the same arithmetic outside a `qif`.

The language reference is being published alongside the paper; until then the examples
in [`examples/`](examples) and the grammar in
[`src/slanq/grammar/slanq.lark`](src/slanq/grammar/slanq.lark) are the best description
of the surface.

## Try it

Requires Python **3.12+**; Qiskit **2.x** is installed as a dependency.

```
pip install -e ".[dev]"
slanq compile examples/ghz.slanq -o circuit.py
python circuit.py
```

`compile` writes the generated Qiskit module to standard output unless `-o` is given.
The generated module is a runnable program on its own.

<details>
<summary>Full install and usage</summary>

Create a virtual environment and install the package in editable mode together with the
development extras (pytest, ruff, mypy). Editable mode means edits to the source tree are
picked up without reinstalling.

```
python -m venv .venv
.venv\Scripts\activate               # Windows
# source .venv/bin/activate          # Linux / macOS
pip install -e ".[dev]"
```

Two equivalent ways to invoke the compiler:

```
slanq compile examples/mvp_a.slanq                # via the console script
python -m slanq compile examples/mvp_a.slanq      # via the -m entry point
```

The `slanq` console script requires the virtual environment to be active, or the package
to be installed globally via `pipx`; `python -m slanq` works whenever the interpreter can
import the package.

`parse` prints the parse tree, which is useful when working on the grammar:

```
slanq parse examples/hello.slanq
```

Without arguments both print a usage message and exit with code 2; `slanq --help` lists
the subcommands.

Run the test suite with `pytest`, and lint with `ruff check src tests`
(`ruff check --fix src tests` to apply the fixable ones).

</details>

## License

See [LICENSE](LICENSE).
