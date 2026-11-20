"""Compatibility entry point for the Section Vd performance-metric report."""

import json
from pathlib import Path
import sys

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "secVd_control_gain_optimization/code")
)
from secvd_metrics import report

if __name__ == "__main__":
    print(json.dumps(report(), indent=2, allow_nan=False))
