"""
Training step for the content-validation pipeline.

This module wraps the existing training routine so the main pipeline can
call a single function without embedding training details.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import train_binary_classifier as tbc


def train_models(
    *,
    args,
    train_variants: Sequence[tbc.SampleVariant],
    val_variants: Sequence[tbc.SampleVariant],
    test_variants: Sequence[tbc.SampleVariant],
    label_texts: Dict[int, str],
    run_root: Path,
) -> List[tbc.SummaryRow]:
    """
    Train each requested model and return their metrics.
    """

    summary: List[tbc.SummaryRow] = []
    for model_name in args.models:
        model_dir = run_root / model_name
        result = tbc.train_one_model(
            model_name,
            args,
            train_variants,
            val_variants,
            test_variants,
            model_dir,
            label_texts,
        )
        summary.append(result)
    return summary
