"""
train_salary_model.py  (v2 — enhanced features)
------------------------------------------------
Run this ONCE locally to regenerate the model:

    python train_salary_model.py

Outputs:
    model/salary_model.pkl
    model/salary_model_meta.json

Changes from v1:
    - Added title_clean as a feature (biggest accuracy gain)
    - Added skill_count (number of skills mentioned)
    - Added binary flags for top-paying skills
"""

import pandas as pd
import numpy as np
import re, json, os, ast, joblib, warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_PATH = "data/cleaned_jobs.csv"
MODEL_DIR = "model"
SAL_MIN   = 20_000
SAL_MAX   = 500_000

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"  Raw rows: {len(df):,}")

# ── Feature Engineering ────────────────────────────────────────────────────────
print("Engineering features...")

def extract_state(loc):
    m = re.search(r",\s*([A-Z]{2})$", str(loc))
    return m.group(1) if m else "Unknown"

df["state"] = df["location"].apply(extract_state)
df["experience_level"] = df["formatted_experience_level"].fillna("Unknown")

def parse_list_col(val):
    if isinstance(val, list): return val
    if pd.isna(val): return []
    try: return ast.literal_eval(str(val))
    except: return []

df["extracted_skills"] = df["extracted_skills"].apply(parse_list_col)
df["skill_count"] = df["extracted_skills"].apply(len)

HIGH_VALUE_SKILLS = ["python", "sql", "aws", "machine learning", "azure",
                     "java", "kubernetes", "tableau", "snowflake", "scala"]
for skill in HIGH_VALUE_SKILLS:
    col = "has_" + skill.replace(" ", "_")
    df[col] = df["extracted_skills"].apply(lambda s: 1 if skill in s else 0)

skill_flag_cols = ["has_" + s.replace(" ", "_") for s in HIGH_VALUE_SKILLS]

# ── Filter ─────────────────────────────────────────────────────────────────────
clean = df[(df["normalized_salary"] >= SAL_MIN) & (df["normalized_salary"] <= SAL_MAX)].copy()
print(f"  Usable rows: {len(clean):,}")

CAT_FEATURES = ["title_clean", "job_category", "remote_status",
                "work_type", "experience_level", "state"]
NUM_FEATURES = ["skill_count"] + skill_flag_cols
ALL_FEATURES = CAT_FEATURES + NUM_FEATURES

X = clean[ALL_FEATURES].copy()
for col in CAT_FEATURES:
    X[col] = X[col].fillna("Unknown").astype(str)
for col in NUM_FEATURES:
    X[col] = X[col].fillna(0)

y = np.log1p(clean["normalized_salary"])

print(f"  Features: {len(ALL_FEATURES)}  ({len(CAT_FEATURES)} categorical, {len(NUM_FEATURES)} numeric)")

# ── Split ──────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── Pipeline ───────────────────────────────────────────────────────────────────
preprocessor = ColumnTransformer([
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_FEATURES),
    ("num", "passthrough", NUM_FEATURES),
])

model = Pipeline([
    ("pre", preprocessor),
    ("reg", GradientBoostingRegressor(
        n_estimators=300, max_depth=6,
        learning_rate=0.08, subsample=0.8, random_state=42,
    ))
])

# ── Train ──────────────────────────────────────────────────────────────────────
print("\nTraining model (this takes 2-3 mins)...")
model.fit(X_train, y_train)

preds_raw = np.expm1(model.predict(X_test))
y_raw     = np.expm1(y_test)
mae  = mean_absolute_error(y_raw, preds_raw)
rmse = np.sqrt(mean_squared_error(y_raw, preds_raw))
r2   = r2_score(y_raw, preds_raw)

print(f"\nTest Results:")
print(f"  MAE:  ${mae:>10,.0f}  (v1 was ~$30,700)")
print(f"  RMSE: ${rmse:>10,.0f}  (v1 was ~$46,200)")
print(f"  R2:   {r2:.4f}         (v1 was ~0.33)")

print("\nRunning 5-fold cross-validation...")
cv_scores = cross_val_score(
    model, X, y,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    scoring="r2", n_jobs=-1,
)
print(f"  CV R2: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

# Feature importance
reg = model.named_steps["reg"]
importances = pd.Series(reg.feature_importances_, index=ALL_FEATURES).sort_values(ascending=False)
print("\nTop 10 feature importances:")
for feat, imp in importances.head(10).items():
    print(f"  {feat:<35} {imp:.4f}")

# ── Save ───────────────────────────────────────────────────────────────────────
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(model, os.path.join(MODEL_DIR, "salary_model.pkl"))

meta = {
    "features":          ALL_FEATURES,
    "cat_features":      CAT_FEATURES,
    "num_features":      NUM_FEATURES,
    "skill_flag_skills": HIGH_VALUE_SKILLS,
    "job_categories":    sorted(clean["job_category"].dropna().unique().tolist()),
    "remote_statuses":   sorted(clean["remote_status"].dropna().unique().tolist()),
    "work_types":        sorted(clean["work_type"].dropna().unique().tolist()),
    "experience_levels": ["Unknown","Internship","Entry level","Associate",
                          "Mid-Senior level","Director","Executive"],
    "states":            sorted([s for s in clean["state"].unique() if s != "Unknown"]) + ["Unknown"],
    "title_options":     sorted(clean["title_clean"].dropna().unique().tolist()),
    "model_name":        "Gradient Boosting Regressor v2",
    "test_mae":          mae,
    "test_rmse":         rmse,
    "test_r2":           r2,
    "cv_r2_mean":        float(cv_scores.mean()),
    "cv_r2_std":         float(cv_scores.std()),
}

with open(os.path.join(MODEL_DIR, "salary_model_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nSaved: {MODEL_DIR}/salary_model.pkl")
print(f"Saved: {MODEL_DIR}/salary_model_meta.json")
print("Now run: streamlit run ap.py")
