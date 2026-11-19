"""Feature engineering for prescriber-level audit prioritization.

Three kinds of features:
  1. Rates and mixes: how a prescriber prescribes, independent of size (e.g. share of claims
     that are opioids, cost per claim).
  2. Volume: how much they prescribe (log-scaled counts).
  3. Peer comparisons: how unusual each rate is compared with prescribers of the same
     specialty in the same year (robust z-scores and percentile ranks).

All features use only the same calendar year as the row, so no information from later years
enters the features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    """Safe ratio: NaN when the denominator is missing or zero."""
    den = den.where(den > 0)
    return num / den


# Rate features that get a peer comparison. Kept short on purpose: each is interpretable.
PEER_FEATURES = [
    "claims_per_bene", "days_per_claim", "cost_per_claim", "cost_per_bene",
    "brand_claim_share", "brand_cost_share", "opioid_claim_share", "opioid_days_per_claim",
    "opioid_la_share", "opioid_bene_share", "antibiotic_claim_share", "antipsychotic_claim_share",
    "lis_claim_share", "dual_bene_share", "log_claims", "log_cost",
]


def add_rate_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["log_claims"] = np.log1p(df["Tot_Clms"])
    out["log_benes"] = np.log1p(df["Tot_Benes"])
    out["log_cost"] = np.log1p(df["Tot_Drug_Cst"])
    out["claims_per_bene"] = _ratio(df["Tot_Clms"], df["Tot_Benes"])
    out["days_per_claim"] = _ratio(df["Tot_Day_Suply"], df["Tot_Clms"])
    out["fills_per_claim"] = _ratio(df["Tot_30day_Fills"], df["Tot_Clms"])
    out["cost_per_claim"] = _ratio(df["Tot_Drug_Cst"], df["Tot_Clms"])
    out["cost_per_bene"] = _ratio(df["Tot_Drug_Cst"], df["Tot_Benes"])
    out["brand_claim_share"] = _ratio(df["Brnd_Tot_Clms"], df["Tot_Clms"])
    out["brand_cost_share"] = _ratio(df["Brnd_Tot_Drug_Cst"], df["Tot_Drug_Cst"])
    out["opioid_claim_share"] = _ratio(df["Opioid_Tot_Clms"], df["Tot_Clms"])
    out["opioid_days_per_claim"] = _ratio(df["Opioid_Tot_Suply"], df["Opioid_Tot_Clms"])
    out["opioid_la_share"] = _ratio(df["Opioid_LA_Tot_Clms"], df["Opioid_Tot_Clms"])
    out["opioid_bene_share"] = _ratio(df["Opioid_Tot_Benes"], df["Tot_Benes"])
    out["opioid_cost_per_claim"] = _ratio(df["Opioid_Tot_Drug_Cst"], df["Opioid_Tot_Clms"])
    out["antibiotic_claim_share"] = _ratio(df["Antbtc_Tot_Clms"], df["Tot_Clms"])
    out["antipsychotic_claim_share"] = _ratio(df["Antpsyct_GE65_Tot_Clms"], df["Tot_Clms"])
    out["lis_claim_share"] = _ratio(df["LIS_Tot_Clms"], df["Tot_Clms"])
    out["mapd_claim_share"] = _ratio(df["MAPD_Tot_Clms"], df["Tot_Clms"])
    out["dual_bene_share"] = _ratio(df["Bene_Dual_Cnt"], df["Tot_Benes"])
    out["female_bene_share"] = _ratio(df["Bene_Feml_Cnt"], df["Tot_Benes"])
    out["bene_avg_age"] = df["Bene_Avg_Age"]
    out["bene_risk_score"] = df["Bene_Avg_Risk_Scre"]
    # How much of the record CMS suppressed (small cells): itself a signal of small panels.
    suppressible = ["Brnd_Tot_Clms", "Opioid_Tot_Clms", "Antbtc_Tot_Clms", "LIS_Tot_Clms",
                    "Bene_Dual_Cnt", "Tot_Benes"]
    out["n_suppressed"] = df[suppressible].isna().sum(axis=1)
    out["is_organization"] = (df["Prscrbr_Ent_Cd"] == "O").astype(int)
    ruca = pd.to_numeric(df["Prscrbr_RUCA"], errors="coerce")
    out["is_rural"] = (ruca >= 4).astype(int)  # RUCA 4-10: micropolitan, small town, rural
    return out


def add_peer_features(feats: pd.DataFrame, specialty: pd.Series,
                      min_peers: int = 30) -> pd.DataFrame:
    """Compare each rate with prescribers of the same specialty in the same year.

    Uses robust z-scores (distance from the specialty median in units of the specialty IQR)
    and percentile ranks. Specialties with fewer than `min_peers` prescribers are pooled into
    an "other" group so that small groups do not produce extreme scores by chance.
    """
    counts = specialty.value_counts()
    group = specialty.where(specialty.map(counts) >= min_peers, "other").fillna("other")
    out = pd.DataFrame(index=feats.index)
    g = feats[PEER_FEATURES].groupby(group)
    med = g.transform("median")
    iqr = g.transform(lambda s: s.quantile(0.75) - s.quantile(0.25))
    z = (feats[PEER_FEATURES] - med) / iqr.where(iqr > 0)
    out[[f"{c}_peer_z" for c in PEER_FEATURES]] = z.clip(-50, 50).to_numpy()
    pct = g.rank(pct=True)
    out[[f"{c}_peer_pct" for c in PEER_FEATURES]] = pct.to_numpy()
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature matrix for one labeled year (rows aligned with `df`)."""
    rates = add_rate_features(df)
    peers = add_peer_features(rates, df["Prscrbr_Type"])
    feats = pd.concat([rates, peers], axis=1)
    feats["specialty"] = df["Prscrbr_Type"].fillna("Unknown").astype("category")
    feats["state"] = df["Prscrbr_State_Abrvtn"].fillna("Unknown").astype("category")
    return feats


def feature_columns(feats: pd.DataFrame, include_categorical: bool = True) -> list[str]:
    cols = [c for c in feats.columns if c not in ("specialty", "state")]
    return cols + (["specialty", "state"] if include_categorical else [])
