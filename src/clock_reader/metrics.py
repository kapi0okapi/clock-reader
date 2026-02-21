from __future__ import annotations

import numpy as np

FAILURE_ERROR = 360.0  # a failed reading counts as the worst possible error (6 hours)


def confusion_matrix(pred: np.ndarray, target: np.ndarray, num_classes: int) -> np.ndarray:
    idx = target.ravel().astype(np.int64) * num_classes + pred.ravel()
    return np.bincount(idx, minlength=num_classes**2).reshape(num_classes, num_classes)


def iou_per_class(cm: np.ndarray) -> np.ndarray:
    inter = np.diag(cm).astype(float)
    union = cm.sum(0) + cm.sum(1) - inter
    return np.divide(inter, union, out=np.full_like(inter, np.nan), where=union > 0)


def reading_metrics(errors_min: list[float], failures: int) -> dict[str, float]:
    """Errors are circular differences in minutes on the 12-hour dial; failures count as wrong."""
    errors = np.array(list(errors_min) + [FAILURE_ERROR] * failures, dtype=float)
    if errors.size == 0:
        return {}
    return {
        "mean_error_min": float(errors.mean()),
        "median_error_min": float(np.median(errors)),
        "acc_1min": float((errors <= 1).mean()),
        "acc_5min": float((errors <= 5).mean()),
        "acc_15min": float((errors <= 15).mean()),
        "fail_rate": failures / errors.size,
    }
