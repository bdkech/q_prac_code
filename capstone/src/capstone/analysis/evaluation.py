from __future__ import annotations

import numpy as np
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray,
) -> dict[str, float]:
    """Compute standard binary classification metrics for csPCa prediction.

    Args:
        y_true: Ground-truth binary labels (0 = no csPCa, 1 = csPCa).
        y_pred: Predicted binary labels from the QSVM.
        y_scores: Decision function scores — higher values indicate the model
            is more confident in the positive class. Used for AUC computation.

    Returns:
        Dict with keys: ``auc``, ``accuracy``, ``f1``, ``precision``, ``recall``.
    """
    return {
        "auc": float(roc_auc_score(y_true, y_scores)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def print_report(metrics: dict[str, float], split_name: str) -> None:
    """Log a single-line formatted metrics summary for a data split.

    Args:
        metrics: Output of compute_metrics.
        split_name: Human-readable split label (e.g., ``"val"``, ``"test"``).
    """
    logger.info(
        f"[{split_name}] "
        f"AUC={metrics['auc']:.3f}  "
        f"Acc={metrics['accuracy']:.3f}  "
        f"F1={metrics['f1']:.3f}  "
        f"Prec={metrics['precision']:.3f}  "
        f"Rec={metrics['recall']:.3f}"
    )
