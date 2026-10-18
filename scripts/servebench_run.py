#!/usr/bin/env python3
"""`servebench run` for lab extras, which call `python3 SCRIPT URL ARGS...`.

usage: scripts/servebench_run.py URL [servebench run options]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from servebench.__main__ import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["run", *sys.argv[1:]]))
