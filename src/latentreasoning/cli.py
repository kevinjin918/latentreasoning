"""Minimal CLI: report available models and package version."""

from __future__ import annotations

import argparse

import latentreasoning

# Import the adapters package so real models self-register alongside the mock.
import latentreasoning.models  # noqa: F401,E402  (registration side effect)
from latentreasoning.core.model import available_models


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="latentreasoning")
    parser.add_argument("--version", action="store_true", help="print version and exit")
    args = parser.parse_args(argv)
    if args.version:
        print(latentreasoning.__version__)
        return 0
    print(f"latentreasoning {latentreasoning.__version__}")
    print("registered models:", ", ".join(available_models()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
