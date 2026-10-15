#!/usr/bin/env python3
"""Prepare real vision datasets from official downloads or acquired sources."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from matu.preparation import main

if __name__ == "__main__":
    main()
