from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    e = y_true - y_pred
    return float(np.sqrt(np.mean(e * e)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, *, eps: float = 1e-6) -> float:
    denom = np.maximum(np.abs(y_true), float(eps))
    return float(np.mean(np.abs((y_true - y_pred) / denom)))


def smape(y_true: np.ndarray, y_pred: np.ndarray, *, eps: float = 1e-6) -> float:
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), float(eps))
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))


def compute_all(
    *,
    metrics: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for m in metrics:
        if m == 'rmse':
            out[m] = rmse(y_true, y_pred)
        elif m == 'mae':
            out[m] = mae(y_true, y_pred)
        elif m == 'mape':
            out[m] = mape(y_true, y_pred, eps=eps)
        elif m == 'smape':
            out[m] = smape(y_true, y_pred, eps=eps)
        else:
            raise ValueError(f'неизвестная метрика: {m}')
    return out
