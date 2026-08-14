"""Command line interface of the compiler."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from slanq import __version__
from slanq.compiler import CompilationResult, compile_source
from slanq.diagnostics import SlanqError

EXIT_OK = 0
EXIT_COMPILE_ERROR = 1
EXIT_USAGE_ERROR = 2


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="slanq",
        description="Compiler for the Slanq quantum programming language.",
    )
    parser.add_argument("--version", action="version", version=f"slanq {__version__}")

    sub = parser.add_subparsers(dest="command", required=True)

    parse_cmd = sub.add_parser("parse", help="Parse the source and print the parse tree.")
    parse_cmd.add_argument("source", type=Path, help="Path to the .slanq source file.")

    compile_cmd = sub.add_parser("compile", help="Compile the source to a Qiskit Python module.")
    compile_cmd.add_argument("source", type=Path, help="Path to the .slanq source file.")
    compile_cmd.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write the generated module here instead of standard output.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    source = _read_source(args.source)
    if source is None:
        return EXIT_USAGE_ERROR

    stop_after = "parse" if args.command == "parse" else None
    try:
        result = compile_source(source, source_name=args.source.name, stop_after=stop_after)
    except SlanqError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_COMPILE_ERROR

    _report(result)
    if result.diagnostics.has_errors:
        return EXIT_COMPILE_ERROR

    if args.command == "parse":
        print(result.parse_tree.pretty())
        return EXIT_OK

    return _emit(result, args.output)


def _read_source(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"Could not read file: {exc}", file=sys.stderr)
        return None


def _report(result: CompilationResult) -> None:
    for diagnostic in result.diagnostics:
        print(str(diagnostic), file=sys.stderr)


def _emit(result: CompilationResult, output: Path | None) -> int:
    if result.qiskit_source is None:
        print("Compilation produced no output.", file=sys.stderr)
        return EXIT_COMPILE_ERROR

    if output is None:
        print(result.qiskit_source, end="")
        return EXIT_OK

    try:
        output.write_text(result.qiskit_source, encoding="utf-8")
    except OSError as exc:
        print(f"Could not write file: {exc}", file=sys.stderr)
        return EXIT_USAGE_ERROR
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
