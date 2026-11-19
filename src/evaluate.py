"""Ranking metrics for audit prioritization.

An auditor can review only a small share of prescribers, so what matters is how many true
positives land at the top of the ranked list, not overall accuracy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def precision_recall_at_k(y_true: np.ndarray, scores: np.ndarray, frac: float) -> tuple[float, float]:
    """Precision and recall when auditing the top `frac` of prescribers by score."""
    y_true = np.asarray(y_true)
    k = max(1, int(round(len(y_true) * frac)))
    top = np.argsort(-np.asarray(scores), kind="stable")[:k]
    hits = y_true[top].sum()
    return hits / k, hits / max(1, y_true.sum())


def ranking_report(y_true, scores, fracs=(0.001, 0.01, 0.05)) -> dict:
    """PR-AUC, ROC-AUC, and precision/recall/lift at several audit capacities."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    scores = np.where(np.isnan(scores), np.nanmin(scores) - 1, scores)
    base = y_true.mean()
    out = {
        "positives": int(y_true.sum()),
        "base_rate_per_10k": base * 1e4,
        "pr_auc": average_precision_score(y_true, scores),
        "pr_auc_lift": average_precision_score(y_true, scores) / base,
        "roc_auc": roc_auc_score(y_true, scores),
    }
    for f in fracs:
        p, r = precision_recall_at_k(y_true, scores, f)
        tag = f"{f * 100:g}%"
        out[f"precision@{tag}"] = p
        out[f"recall@{tag}"] = r
        out[f"lift@{tag}"] = p / base
    return out


def bootstrap_ci(y_true, scores, metric: str, n_boot: int = 200, seed: int = 0,
                 fracs=(0.001, 0.01, 0.05)) -> tuple[float, float]:
    """95% bootstrap interval for one metric of `ranking_report` (resampling prescribers)."""
    rng = np.random.default_rng(seed)
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    n, vals = len(y_true), []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y_true[idx].sum() == 0:
            continue
        vals.append(ranking_report(y_true[idx], scores[idx], fracs)[metric])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population stability index between two samples of one feature (drift check).

    Rule of thumb: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 large shift.
    """
    e, a = expected.dropna(), actual.dropna()
    if e.nunique() < 2 or a.empty:
        return np.nan
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(e, edges)[0] / len(e)
    pa = np.histogram(a, edges)[0] / len(a)
    pe, pa = np.clip(pe, 1e-6, None), np.clip(pa, 1e-6, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))
