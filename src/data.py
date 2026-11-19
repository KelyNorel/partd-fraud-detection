"""Load the raw CMS Part D and OIG LEIE files and build leakage-safe labels.

Raw files are expected in data/raw/ (see README for download links):
  - Medicare_Part_D_Prescribers_by_Provider_<YEAR>.zip  (one per year)
  - LEIE_<YYYY-MM>.csv                                  (OIG exclusion list)
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

# Exclusion types (section of the Social Security Act) that we treat as fraud-related.
# 1128(a)(1): conviction of program-related crimes (Medicare/Medicaid fraud)
# 1128(a)(3): felony conviction for health care fraud
# 1128(a)(4): felony conviction relating to controlled substances
# 1128(b)(7): fraud, kickbacks and other prohibited activities
FRAUD_TYPES = {"1128a1", "1128a3", "1128a4", "1128b7"}

# Plain-language labels for the most common exclusion types, used in tables and plots.
EXCLUSION_TYPE_NAMES = {
    "1128a1": "Conviction: program-related crime",
    "1128a2": "Conviction: patient abuse or neglect",
    "1128a3": "Felony: health care fraud",
    "1128a4": "Felony: controlled substances",
    "1128b1": "Misdemeanor: health care fraud",
    "1128b4": "License revocation or suspension",
    "1128b7": "Fraud, kickbacks, prohibited acts",
    "1128b14": "Default on health education loan",
}

# Columns we keep from the Part D file. Identity columns (names, street address) are dropped:
# the model must not learn from who someone is, only from how they prescribe.
PARTD_COLUMNS = [
    "Prscrbr_NPI", "Prscrbr_Ent_Cd", "Prscrbr_State_Abrvtn", "Prscrbr_RUCA", "Prscrbr_Type",
    "Tot_Clms", "Tot_30day_Fills", "Tot_Drug_Cst", "Tot_Day_Suply", "Tot_Benes",
    "Brnd_Tot_Clms", "Brnd_Tot_Drug_Cst", "Gnrc_Tot_Clms", "Gnrc_Tot_Drug_Cst",
    "MAPD_Tot_Clms", "PDP_Tot_Clms", "LIS_Tot_Clms", "LIS_Drug_Cst", "NonLIS_Tot_Clms",
    "Opioid_Tot_Clms", "Opioid_Tot_Drug_Cst", "Opioid_Tot_Suply", "Opioid_Tot_Benes",
    "Opioid_Prscrbr_Rate", "Opioid_LA_Tot_Clms", "Opioid_LA_Tot_Suply",
    "Antbtc_Tot_Clms", "Antpsyct_GE65_Tot_Clms",
    "Bene_Avg_Age", "Bene_Feml_Cnt", "Bene_Dual_Cnt", "Bene_Ndual_Cnt", "Bene_Avg_Risk_Scre",
]
CATEGORICAL = {"Prscrbr_NPI", "Prscrbr_Ent_Cd", "Prscrbr_State_Abrvtn", "Prscrbr_RUCA", "Prscrbr_Type"}


def load_partd(year: int, raw_dir: Path = RAW) -> pd.DataFrame:
    """Read one year of Part D Prescribers by Provider, keeping only the modeling columns.

    Empty cells mean the value was suppressed by CMS (counts below 11) or not applicable;
    they stay as NaN so the models can treat "suppressed" differently from zero.
    """
    path = raw_dir / f"Medicare_Part_D_Prescribers_by_Provider_{year}.zip"
    with zipfile.ZipFile(path) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".csv"))
        with zf.open(name) as fh:
            df = pd.read_csv(fh, usecols=PARTD_COLUMNS, dtype=str, low_memory=False)
    for col in PARTD_COLUMNS:
        if col not in CATEGORICAL:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["year"] = year
    return df


def load_leie(raw_dir: Path = RAW) -> pd.DataFrame:
    """Read the OIG LEIE file and keep the records we can link to Part D (those with an NPI)."""
    path = sorted(raw_dir.glob("LEIE_*.csv"))[-1]
    leie = pd.read_csv(path, dtype=str)
    leie["NPI"] = leie["NPI"].fillna("0").str.strip()
    leie["excl_date"] = pd.to_datetime(leie["EXCLDATE"], format="%Y%m%d", errors="coerce")
    leie["is_fraud_type"] = leie["EXCLTYPE"].str.strip().isin(FRAUD_TYPES)
    with_npi = leie[leie["NPI"].ne("0") & leie["excl_date"].notna()]
    # A few NPIs appear more than once; keep the earliest exclusion.
    return (with_npi.sort_values("excl_date")
                    .drop_duplicates("NPI")[["NPI", "excl_date", "EXCLTYPE", "is_fraud_type",
                                              "GENERAL", "SPECIALTY"]]
                    .rename(columns={"EXCLTYPE": "excl_type"}))


def build_labels(partd: pd.DataFrame, leie: pd.DataFrame, year: int,
                 horizon_years: int = 2) -> pd.DataFrame:
    """Attach leakage-safe labels to one year of Part D data.

    Features describe prescribing in calendar year `year`. A prescriber is positive if the
    OIG excluded them in the next `horizon_years` years, i.e. strictly after the year the
    features describe. Prescribers excluded on or before the end of `year` are dropped:
    their exclusion was already known, so using them would leak the answer.
    """
    end_of_year = pd.Timestamp(year=year, month=12, day=31)
    horizon_end = pd.Timestamp(year=year + horizon_years, month=12, day=31)

    out = partd.merge(leie, how="left", left_on="Prscrbr_NPI", right_on="NPI")
    already_excluded = out["excl_date"].notna() & (out["excl_date"] <= end_of_year)
    out = out[~already_excluded].copy()

    in_window = out["excl_date"].gt(end_of_year) & out["excl_date"].le(horizon_end)
    out["label_any"] = in_window.astype(int)
    out["label_fraud"] = (in_window & out["is_fraud_type"].fillna(False)).astype(int)
    # Excluded after the horizon: not a positive for this window, but flag it for analysis.
    out["excluded_later"] = out["excl_date"].gt(horizon_end).astype(int)
    return out.drop(columns=["NPI"])


def summarize_labels(df: pd.DataFrame) -> pd.Series:
    """Counts and rates of positives for one labeled year."""
    n = len(df)
    return pd.Series({
        "prescribers": n,
        "positives_any": int(df["label_any"].sum()),
        "positives_fraud": int(df["label_fraud"].sum()),
        "rate_any_per_10k": round(df["label_any"].mean() * 1e4, 2),
        "rate_fraud_per_10k": round(df["label_fraud"].mean() * 1e4, 2),
    })


def save_processed(df: pd.DataFrame, name: str) -> Path:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    path = PROCESSED / f"{name}.parquet"
    df.to_parquet(path, index=False)
    return path


def load_processed(name: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / f"{name}.parquet")


if __name__ == "__main__":  # pragma: no cover
    leie = load_leie()
    for y in (2022, 2023, 2024):
        labeled = build_labels(load_partd(y), leie, y)
        save_processed(labeled, f"labeled_{y}")
        print(y, summarize_labels(labeled).to_dict())
