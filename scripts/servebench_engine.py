#!/usr/bin/env python3
"""`servebench engine` for lab extras, which call `python3 SCRIPT URL ARGS...`.

usage: scripts/servebench_engine.py URL [servebench engine options]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from servebench.__main__ import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["engine", *sys.argv[1:]]))
