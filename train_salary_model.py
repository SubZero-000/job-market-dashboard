"""
train_salary_model.py 
----------------------------------------------------------
Run this ONCE locally to regenerate the model:

    python train_salary_model.py

Outputs:
    model/salary_model.pkl
    model/salary_model_meta.json
"""

import pandas as pd
import numpy as np
import re, json, os, joblib, warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_PATH = "data/cleaned_jobs2.csv"
MODEL_DIR = "model"
SAL_MIN   = 20_000
SAL_MAX   = 500_000

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"  Raw rows: {len(df):,}")

# ── Feature engineering ────────────────────────────────────────────────────────
print("Engineering features...")

def extract_state(loc):
    m = re.search(r",\s*([A-Z]{2})$", str(loc))
    return m.group(1) if m else "Unknown"

df["state"]            = df["location"].apply(extract_state)
df["experience_level"] = df["formatted_experience_level"].fillna("Unknown")

# ── Filter valid salaries ──────────────────────────────────────────────────────
clean = df[(df["normalized_salary"] >= SAL_MIN) & (df["normalized_salary"] <= SAL_MAX)].copy()
print(f"  Usable rows: {len(clean):,}")

# ── Features ───────────────────────────────────────────────────────────────────
# Low cardinality  → OrdinalEncoder (safe, no leakage risk)
# High cardinality → TargetEncoder  (encodes as mean log-salary per category)
LOW_CARD  = ["job_category", "remote_status", "work_type", "experience_level", "state"]
HIGH_CARD = ["title_clean", "company_name"]
ALL_FEATURES = LOW_CARD + HIGH_CARD

X = clean[ALL_FEATURES].copy()
for col in ALL_FEATURES:
    X[col] = X[col].fillna("Unknown").astype(str)

y = np.log1p(clean["normalized_salary"])

print(f"  Features: {len(ALL_FEATURES)}")
print(f"    OrdinalEncoder : {LOW_CARD}")
print(f"    TargetEncoder  : {HIGH_CARD}")

# ── Split ──────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── Pipeline ───────────────────────────────────────────────────────────────────
preprocessor = ColumnTransformer([
    ("ord", OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    ), LOW_CARD),
    ("tgt", TargetEncoder(
        target_type="continuous",
        smooth="auto",       # smooths toward global mean for rare categories
    ), HIGH_CARD),
])

model = Pipeline([
    ("pre", preprocessor),
    ("reg", GradientBoostingRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        random_state=42,
    ))
])

# ── Train ──────────────────────────────────────────────────────────────────────
print("\nTraining model (2-3 mins)...")
model.fit(X_train, y_train)

# ── Evaluate ───────────────────────────────────────────────────────────────────
preds_raw = np.expm1(model.predict(X_test))
y_raw     = np.expm1(y_test)

mae  = mean_absolute_error(y_raw, preds_raw)
rmse = np.sqrt(mean_squared_error(y_raw, preds_raw))
r2   = r2_score(y_raw, preds_raw)

print(f"\nTest Results:")
print(f"  MAE:  ${mae:>10,.0f}  (v2 was ~$28,800  |  v1 was ~$30,700)")
print(f"  RMSE: ${rmse:>10,.0f}  (v2 was ~$45,500  |  v1 was ~$46,200)")
print(f"  R2:   {r2:.4f}         (v2 was ~0.35     |  v1 was ~0.33)")

print("\nRunning 5-fold cross-validation...")
cv_scores = cross_val_score(
    model, X, y,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    scoring="r2",
    n_jobs=-1,
)
print(f"  CV R2: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"  Fold scores: {[round(float(s), 4) for s in cv_scores]}")

# Feature importance
reg = model.named_steps["reg"]
importances = pd.Series(
    reg.feature_importances_, index=ALL_FEATURES
).sort_values(ascending=False)
print("\nFeature importances:")
for feat, imp in importances.items():
    bar = "█" * int(imp * 200)
    print(f"  {feat:<25} {imp:.4f}  {bar}")

# ── Save ───────────────────────────────────────────────────────────────────────
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(model, os.path.join(MODEL_DIR, "salary_model.pkl"))

meta = {
    "features":          ALL_FEATURES,
    "low_card_features": LOW_CARD,
    "high_card_features":HIGH_CARD,
    "job_categories":    sorted(clean["job_category"].dropna().unique().tolist()),
    "remote_statuses":   sorted(clean["remote_status"].dropna().unique().tolist()),
    "work_types":        sorted(clean["work_type"].dropna().unique().tolist()),
    "experience_levels": ["Unknown", "Internship", "Entry level", "Associate",
                          "Mid-Senior level", "Director", "Executive"],
    "states":            sorted([s for s in clean["state"].unique() if s != "Unknown"]) + ["Unknown"],
    "title_options":     sorted(clean["title_clean"].dropna().unique().tolist()),
    "company_options":   sorted(clean["company_name"].dropna().unique().tolist()),
    "model_name":        "Gradient Boosting Regressor v3",
    "encoder":           "TargetEncoder (title_clean, company_name)",
    "test_mae":          mae,
    "test_rmse":         rmse,
    "test_r2":           r2,
    "cv_r2_mean":        float(cv_scores.mean()),
    "cv_r2_std":         float(cv_scores.std()),
}

with open(os.path.join(MODEL_DIR, "salary_model_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print(f"\n Saved:")
print(f"   {MODEL_DIR}/salary_model.pkl")
print(f"   {MODEL_DIR}/salary_model_meta.json")
print(f"\nRun: streamlit run ap.py")
