# Part D Fraud Detection: Prioritizing Prescribers for Audit

**Question:** if an audit team can review only a small share of Medicare Part D prescribers, which ones should it look at first?

This project ranks 1.4 million prescribers using public CMS prescribing data and evaluates the ranking against who the HHS Office of Inspector General (OIG) actually excluded from federal health programs in the following two years. It is a public-data proxy for the problem payers and payment-integrity vendors face with claims: rare positives, imperfect labels, limited audit capacity, and the need to explain every flag.

## Results (test year 2024, exclusions in 2025-2026, never seen in training)

| Audit capacity | Prescribers reviewed | Future exclusions found | Random would find | Lift (x random) |
|---|---|---|---|---|
| Top 0.1% | 1,416 | 14 | 0.3 | 52x |
| Top 0.5% | 7,084 | 33 | 1.3 | 25x |
| Top 1% | 14,168 | 46 (17%) | 2.7 | 17x |
| Top 2% | 28,336 | 60 (22%) | 5.4 | 11x |
| Top 5% | 70,841 | 91 (34%) | 13.4 | 7x |
| Top 10% | 141,682 | 125 (47%) | 26.8 | 5x |

Lift is highest at the top of the ranked list and falls as audit capacity grows. The 0.1% row rests on only 14 cases and is the noisiest estimate.

- **Lift at 1% audit capacity: 17x over random** (95% bootstrap CI 13.7x to 22.1x), any-exclusion label; ROC-AUC 0.81.
- **Fraud-only label** (113 positives): recall 16% at 1% capacity, 37% at 5%; lift at 1% of 15.9x (CI 10.6x to 23.5x).
- **Supervised beats unsupervised.** On the validation year, an Isolation Forest on peer-comparison features (no labels) reached a lift of 4.1x at 1% capacity, and a rule of thumb 2.1x; gradient-boosted trees reached 17-18x. Being unusual is not the same as being excluded: past outcomes teach the model which unusual patterns matter.
- Precision stays low in absolute terms (positives are about 2 per 10,000 prescribers), which is why the evaluation is framed as recall and lift at a fixed audit capacity.

## Data (public, not included in the repo)

| Dataset | Source | Use |
|---|---|---|
| Medicare Part D Prescribers by Provider, 2022-2024 | [data.cms.gov](https://data.cms.gov/provider-summary-by-type-of-service/medicare-part-d-prescribers/medicare-part-d-prescribers-by-provider) | Features: one row per prescriber per year (claims, cost, days supply, brand/generic, opioids, antibiotics, antipsychotics, patient mix) |
| LEIE (List of Excluded Individuals/Entities), Nov 2026 | [oig.hhs.gov](https://oig.hhs.gov/exclusions/exclusions_list.asp) | Labels: exclusion type and date, linked by NPI |

Download the three Part D zip files and the LEIE CSV into `data/raw/` (rename the LEIE file to `LEIE_<YYYY-MM>.csv`).

## Method

**Leakage-safe labels.** Features describe prescribing in year *Y*. A prescriber is positive if excluded in the two years after *Y*; anyone excluded on or before the end of *Y* is dropped, because their exclusion was already known. Prescribers excluded after the two-year window are left out of training and validation years: they are neither clean positives nor clean negatives for that window.

| Use | Features | Positive if excluded in | Positives (any / fraud) |
|---|---|---|---|
| Train | 2022 | 2023-2024 | 309 / 185 |
| Validation | 2023 | 2024-2025 | 290 / 146 |
| Test | 2024 | 2025-2026 | 268 / 113 |

**Two label definitions.** `label_any` counts every exclusion type. `label_fraud` keeps only fraud-related types: 1128(a)(1) program-related crimes, 1128(a)(3) health care fraud felony, 1128(a)(4) controlled substance felony, 1128(b)(7) fraud and kickbacks. The most common type, 1128(b)(4) license revocation, is excluded from the fraud label because it can follow many causes.

**Features.** Rates and mixes (claims per patient, cost per patient, brand share, opioid share and days per opioid claim, low-income and dual-eligible patient shares), log volumes, and **peer comparisons**: robust z-scores and percentiles of each rate within the prescriber's specialty and year. CMS-suppressed values (counts below 11) are kept as missing, not zero.

**Models.** Random baseline, a peer-percentile rule of thumb, Isolation Forest (unsupervised), and LightGBM trained on all positives plus a 5% sample of negatives. Final models are trained on 2022 and 2023 and tested on 2024.

**Explainability and stability.** SHAP values explain each flag. Population stability index (PSI) checks drift between 2022 and 2024: brand cost share (0.11) shows a moderate shift; the other top features are stable (< 0.1).

## Findings worth discussing

- **Patient mix and intensity per patient carry most of the signal**, not raw volume: later-excluded prescribers sit high within their specialty on dual-eligible and low-income patient shares and on claims per patient, while their total volume is typical.
- **Specialty is the single most important feature** (base rates differ a lot across the 214 specialties); a model without it performed about the same on validation, so the signal does not depend on specialty alone.
- **Highest exclusion rates by specialty** (2022-2024 pooled, specialties with 10,000+ prescriber-years): Physical Medicine and Rehabilitation and General Practice, each about 5 fraud-label exclusions per 10,000 prescriber-years, roughly 4.5 times the overall rate. In absolute numbers, Family Practice and Internal Medicine have the most exclusions because they are large. Counts per specialty are small, so rates are indicative.
- **Opioids matter in a specific way**: not a higher share of opioid claims, but longer days of supply per opioid claim.

## Limitations

- Labels capture only providers who were caught and are still excluded: the LEIE removes reinstated providers, and undetected fraud appears as a negative.
- Only about 10% of LEIE records have an NPI, so many exclusions cannot be linked.
- Not every exclusion is about prescribing, and Part D covers only prescription drugs.
- The 2026 label window is incomplete (data downloaded November 2026).
- Part D data is aggregated by prescriber and year and released about 18 months late. With claim-level data and audit outcomes, the same framework (leakage-safe labels, capacity-based metrics, peer comparison, explanations) would rank claims instead of prescribers.

## Repository

```
notebooks/
  01_data_and_labels.ipynb       datasets, label design, positives by exclusion type
  02_features.ipynb              features and peer comparisons, univariate signal
  03_models_and_evaluation.ipynb models, out-of-time test, bootstrap CIs, SHAP, drift
src/
  data.py       loading and leakage-safe labels
  features.py   rate, volume and peer-comparison features
  evaluate.py   PR-AUC, precision/recall/lift at k, bootstrap CIs, PSI
tests/          label logic (no leakage) and metric tests
```

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
jupyter lab   # run notebooks 01, 02, 03 in order
```
