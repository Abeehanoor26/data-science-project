# Project 1: Advanced EDA & Feature Engineering
**DecodeLabs Data Science Industrial Training Kit — Batch 2026**

## What's in this folder

| File | Purpose |
|---|---|
| `generate_dataset.py` | Builds a realistic, deliberately messy retail-customer dataset (missing values at 3 different bands, injected outliers, one near-duplicate column for the multicollinearity demo). |
| `raw_customer_data.csv` | The raw output of the generator — 1,200 rows, 11 columns. |
| `Project1_Advanced_EDA_Feature_Engineering.ipynb` | The full solution notebook, already executed with outputs. |
| `cleaned_customer_data.csv` | The final, model-ready dataset produced by the notebook. |
| `boxplots_before_cleaning.png` / `boxplots_after_cleaning.png` | Visual proof outliers were neutralized. |

## What the notebook does (mapped to the brief's key requirements)

1. **EDA** — dtypes, missingness per column, descriptive stats, boxplots.
2. **Missing data** — applies the missing-data decision matrix from the brief:
   - `< 5%` missing → row deletion (`Age`)
   - `5–20%` missing → median/mode imputation (`Rating`, `Category`)
   - `> 20%` missing → KNN imputation (`Annual_Income`)
3. **Outliers** — IQR-based bounds (`Q1 - 1.5*IQR`, `Q3 + 1.5*IQR`), neutralized via `clip()`
   (winsorization) instead of row deletion, to preserve data volume.
4. **Encoding** — one-hot encoding for `City` and `Category` (avoids the false ordinal
   distance that label encoding would introduce).
5. **Feature engineering** — 4 new engineered features, all vectorized (no Python loops):
   - `Income_to_Purchase_Ratio`
   - `Purchase_per_Membership_Year`
   - `Recency_Score`
   - `Age_Group` (binned + one-hot encoded)
6. **Multicollinearity** — builds the correlation matrix, flags pairs with `|corr| > 0.80`,
   and for each pair drops whichever variable correlates *less* with the target
   (`Customer_LTV`), rather than an arbitrary first-found column.
7. **Output contract** — validates the final dataset against a Pandera schema
   (`lazy=True`), which actually catches a real edge case: a handful of ages statistically
   clipped to ~96 (IQR bound) that violate the business rule of 18–70 — showing why a
   structural contract layer matters even after statistical cleaning.

## How to run it yourself

```bash
pip install pandas numpy scikit-learn matplotlib pandera jupyter --break-system-packages
python3 generate_dataset.py
jupyter nbconvert --to notebook --execute --inplace Project1_Advanced_EDA_Feature_Engineering.ipynb
```

## Ideas to extend (per the brief's conclusion)

- Compare **Mean vs. KNN imputation** for `Annual_Income` and check which preserves the
  original distribution's shape (plot both histograms overlaid).
- Try **Z-score** outlier detection (`|z| > 3`) instead of IQR and compare how many points
  each method flags.
- Swap the Pandera schema's `Age` bound handling to re-clip values that fail the business
  rule, and re-validate.
