from __future__ import annotations

DIMENSIONS = ("safety", "lively", "beautiful", "wealthy", "boring", "depressing")
CHOICES = ("left", "right", "equal")
VERDICTS = (
    "MODEL_CALIBRATION_FAILED",
    "SCALAR_SIGNAL_NOT_ESTABLISHED",
    "SCALAR_NOT_REJECTED",
    "SCALAR_REJECTED_CONTINUOUS",
    "SCALAR_REJECTED_MIXTURE",
    "BIMODAL_NON_CUSP",
    "CUSP_COMPATIBLE",
    "DATA_INSUFFICIENT",
)
CHOICE_TO_INDEX = {"left": 0, "right": 1, "equal": 2}

