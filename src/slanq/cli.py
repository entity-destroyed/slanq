"""Command line interface of the compiler."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from slanq import __version__
from slanq.compiler import compile_source
from slanq.diagnostics import SlanqError


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="slanq",
        description="Compiler for the Slanq quantum programming language.",
    )
    parser.add_argument("--version", action="version", version=f"slanq {__version__}")

    sub = parser.add_subparsers(dest="command", required=True)

    parse_cmd = sub.add_parser("parse", help="Parse the source and print the parse tree.")
    parse_cmd.add_argument("source", type=Path, help="Path to the .slanq source file.")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.command == "parse":
        try:
            source = args.source.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"Could not read file: {exc}", file=sys.stderr)
            return 2

        try:
            result = compile_source(source, stop_after="parse")
        except SlanqError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        print(result.parse_tree.pretty())

        for diagnostic in result.diagnostics:
            print(str(diagnostic), file=sys.stderr)

        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
