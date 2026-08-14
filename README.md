# Slanq

Compiler for the **Slanq** high-level quantum programming language. Compiled programs
are emitted as Qiskit circuits, so they can run both on simulators and on real hardware.

The language specification lives in the accompanying laboratory documentation.

## Requirements

- Python **3.12+**
- Qiskit **2.x** (installed automatically as a dependency)

The project runs on Windows, Linux and macOS. The commands below use the Windows
`.venv\Scripts\` path; on Linux and macOS use `.venv/bin/` instead.

## Development install

Create a virtual environment and install the package in editable mode together
with the development extras (pytest, ruff, mypy):

```
python -m venv .venv
.venv\Scripts\activate               # Windows
# source .venv/bin/activate          # Linux / macOS
pip install -e ".[dev]"
```

Editable mode (`-e`) means edits to the source tree are picked up immediately,
without reinstalling.

## Usage

Two equivalent ways to invoke the compiler once installed:

```
slanq compile examples/mvp_a.slanq                # via the console script
python -m slanq compile examples/mvp_a.slanq      # via the -m entry point
```

`compile` writes the generated Qiskit module to standard output; pass `-o` to
write it to a file instead. The generated module is runnable on its own:

```
slanq compile examples/mvp_a.slanq -o circuit.py
python circuit.py
```

`parse` prints the parse tree, which is useful when working on the grammar:

```
slanq parse examples/hello.slanq
```

Without arguments both print a usage message and exit with code 2. `slanq --help`
lists the available subcommands.

The `slanq` console script requires the virtual environment to be active (or the
package to be installed globally via `pipx`); `python -m slanq` works whenever the
interpreter can import the package.

## Project layout

```
src/slanq/
├── __init__.py       package init, exposes __version__
├── __main__.py       entry point for `python -m slanq`
├── cli.py            command line interface (argparse)
├── compiler.py       compilation pipeline driver
├── diagnostics.py    Diagnostic / DiagnosticBag / SlanqError
├── parser.py         Lark-based syntactic analysis
├── transformer.py    parse tree -> AST
├── ast_nodes.py      AST classes
├── visitor.py        NodeVisitor over the AST
├── analysis.py       semantic analysis; quantum analyses (never mutate)
├── lowering.py       AST -> IR
├── ir.py             intermediate representation
├── passes.py         IR-mutating transformations
├── codegen.py        IR -> Qiskit Python source
└── grammar/
    ├── __init__.py   grammar loader (via importlib.resources)
    └── slanq.lark    the grammar definition
```

The pipeline runs in that order: `parser` → `transformer` → `analysis` →
`lowering` → (`analysis` / `passes` over the IR) → `codegen`.

**Architectural rule.** `analysis.py` only derives facts about the IR; all IR
mutation happens in `passes.py`. This split is enforced by convention, not by a
directory boundary, so that it can later be promoted to a Qiskit-style
`PassManager` without any code restructuring.

## Development

Run the test suite:

```
pytest
```

Lint the code:

```
ruff check src tests
```

Automatically fix any fixable lint issues:

```
ruff check --fix src tests
```
