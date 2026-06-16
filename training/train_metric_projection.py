"""Compatibility wrapper for the replaced few-shot metric trainer.

The few-shot training path now uses an objectness + type classifier head. This
module forwards old direct calls to training.train_classifier_head.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.train_classifier_head import main, parse_args, run  # noqa: E402


if __name__ == "__main__":
    main()
