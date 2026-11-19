"""Tests for the parts where a silent mistake would inflate results: labels and ranking metrics."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import build_labels  # noqa: E402
from src.evaluate import precision_recall_at_k, ranking_report, psi  # noqa: E402


@pytest.fixture
def toy():
    partd = pd.DataFrame({"Prscrbr_NPI": ["1", "2", "3", "4", "5"], "Tot_Clms": [10, 20, 30, 40, 50]})
    leie = pd.DataFrame({
        "NPI": ["1", "2", "3", "4"],
        "excl_date": pd.to_datetime(["2021-06-01", "2022-12-31", "2023-03-01", "2026-05-01"]),
        "excl_type": ["1128a1", "1128a1", "1128b4", "1128a1"],
        "is_fraud_type": [True, True, False, True],
        "GENERAL": "x", "SPECIALTY": "y",
    })
    return partd, leie


def test_prescribers_excluded_before_or_during_feature_year_are_dropped(toy):
    out = build_labels(*toy, year=2022)
    # NPI 1 (2021) and NPI 2 (Dec 31, 2022) were already excluded when the features were recorded.
    assert set(out["Prscrbr_NPI"]) == {"3", "4", "5"}


def test_positive_only_inside_horizon(toy):
    out = build_labels(*toy, year=2022, horizon_years=2).set_index("Prscrbr_NPI")
    assert out.loc["3", "label_any"] == 1      # excluded Mar 2023: inside 2023-2024
    assert out.loc["4", "label_any"] == 0      # excluded 2026: after the window
    assert out.loc["4", "excluded_later"] == 1
    assert out.loc["5", "label_any"] == 0      # never excluded


def test_fraud_label_is_subset_of_any(toy):
    out = build_labels(*toy, year=2022).set_index("Prscrbr_NPI")
    assert out.loc["3", "label_fraud"] == 0    # license revocation is not a fraud type
    assert (out["label_fraud"] <= out["label_any"]).all()


def test_precision_recall_at_k_perfect_and_random():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    p, r = precision_recall_at_k(y, np.arange(10)[::-1], 0.2)   # positives scored highest
    assert (p, r) == (1.0, 1.0)
    p, r = precision_recall_at_k(y, np.arange(10), 0.2)         # positives scored lowest
    assert (p, r) == (0.0, 0.0)


def test_ranking_report_handles_missing_scores():
    y = np.array([1, 0, 0, 0])
    rep = ranking_report(y, np.array([0.9, np.nan, 0.1, 0.2]), fracs=(0.25,))
    assert rep["precision@25%"] == 1.0


def test_psi_zero_for_identical_and_positive_for_shift():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(size=5000))
    assert psi(a, a) == pytest.approx(0, abs=1e-9)
    assert psi(a, a + 1) > 0.25
