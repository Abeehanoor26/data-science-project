"""
Generates a deliberately messy, realistic retail-customer dataset for
Project 1: Advanced EDA & Feature Engineering.

The dataset intentionally contains:
- Missing values at different proportions (for the missing-data decision matrix)
- Extreme outliers (hardware-glitch style + human transcription errors)
- A near-duplicate / highly-collinear column (Income_Score vs Annual_Income)
- Categorical columns (City, Category) that need encoding
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 1200

# --- Base signal -----------------------------------------------------------
age = rng.integers(18, 70, N).astype(float)
annual_income = np.round(rng.normal(65000, 22000, N).clip(15000, 250000), 2)
membership_years = rng.integers(0, 15, N).astype(float)
category = rng.choice(["Electronics", "Groceries", "Apparel", "Home", "Beauty"], N,
                       p=[0.25, 0.3, 0.2, 0.15, 0.1])
city = rng.choice(["Lahore", "Karachi", "Islamabad", "Faisalabad"], N,
                   p=[0.4, 0.3, 0.2, 0.1])
rating = np.round(rng.normal(3.7, 0.9, N).clip(1, 5), 1)
days_since_last_purchase = rng.integers(0, 400, N).astype(float)

# Purchase amount driven by income + membership loyalty + noise
purchase_amount = (
    0.02 * annual_income
    + 40 * membership_years
    + rng.normal(0, 400, N)
).clip(50, None)
purchase_amount = np.round(purchase_amount, 2)

# Target: Customer Lifetime Value - correlated with income & purchase behaviour
customer_ltv = np.round(
    2.5 * purchase_amount + 150 * membership_years + 0.01 * annual_income
    + rng.normal(0, 800, N), 2
)

# Near-duplicate / highly collinear feature (classic multicollinearity trap)
income_score = np.round(annual_income * 0.001 + rng.normal(0, 2, N), 2)

df = pd.DataFrame({
    "CustomerID": np.arange(1001, 1001 + N),
    "Age": age,
    "Annual_Income": annual_income,
    "Income_Score": income_score,          # engineered to correlate ~0.95+ with Annual_Income
    "Membership_Years": membership_years,
    "Category": category,
    "City": city,
    "Rating": rating,
    "Days_Since_Last_Purchase": days_since_last_purchase,
    "Purchase_Amount": purchase_amount,
    "Customer_LTV": customer_ltv,
})

# --- Inject missingness (per the decision-matrix bands from the slides) ----
def inject_missing(series, frac, rng):
    s = series.copy()
    idx = rng.choice(s.index, size=int(len(s) * frac), replace=False)
    s.loc[idx] = np.nan
    return s

df["Age"] = inject_missing(df["Age"], 0.03, rng)                     # <5%  -> row drop candidate
df["Rating"] = inject_missing(df["Rating"], 0.12, rng)                # 5-20% -> statistical imputation
df["Annual_Income"] = inject_missing(df["Annual_Income"], 0.25, rng)  # >20% -> KNN imputation
df["Category"] = df["Category"].astype(object)
mask = rng.choice(df.index, size=int(0.08 * N), replace=False)
df.loc[mask, "Category"] = np.nan                                     # categorical missingness

# --- Inject outliers (hardware-glitch / transcription-error style) --------
outlier_idx = rng.choice(df.index, size=15, replace=False)
df.loc[outlier_idx, "Purchase_Amount"] = df.loc[outlier_idx, "Purchase_Amount"] * rng.uniform(6, 12, 15)

age_typo_idx = rng.choice(df.dropna(subset=["Age"]).index, size=5, replace=False)
df.loc[age_typo_idx, "Age"] = df.loc[age_typo_idx, "Age"] * 10  # e.g. 45 -> 450 (transcription error)

df.to_csv("/home/claude/project/raw_customer_data.csv", index=False)
print("Saved raw_customer_data.csv with shape", df.shape)
print(df.isna().mean().round(3))
