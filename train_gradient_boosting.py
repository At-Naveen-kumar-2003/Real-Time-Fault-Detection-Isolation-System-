# =============================================================================
#  train_gradient_boosting.py — Train and save the Gradient Boosting model
#
#  Run: python3 train_gradient_boosting.py
#  Output:
#    saved_models/gradient_boosting_model.json  (XGBoost) or .pkl (sklearn)
#    saved_models/gradient_boosting_scaler.pkl
#    saved_models/model_accuracy.json
#    saved_models/model_metrics.json
# =============================================================================

import os
import sys
import json
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import FEATURE_NAMES, FAULT_LABELS, N_CLASSES, MODEL_DIR
from gradient_boosting import GradientBoostingModel

# ── 1. Generate or load dataset ───────────────────────────────────────────────
CSV = os.path.join(MODEL_DIR, "training_data.csv")

if not os.path.exists(CSV):
    print("Dataset not found — generating now...")
    from generate_dataset import generate_dataset
    generate_dataset(n_per_class=1500)

import pandas as pd
df = pd.read_csv(CSV)
X  = df[FEATURE_NAMES].values.astype(np.float32)
y  = df["label"].values.astype(np.int64)

print(f"Dataset loaded: {len(df):,} samples  |  {N_CLASSES} classes")

# ── 2. Train / test split ─────────────────────────────────────────────────────
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

print(f"Train: {len(X_tr):,}   Test: {len(X_te):,}")

# ── 3. Train ──────────────────────────────────────────────────────────────────
print("\nTraining Gradient Boosting model...")
model = GradientBoostingModel()
model.fit(X_tr, y_tr, X_val=X_te, y_val=y_te)

# ── 4. Evaluate ───────────────────────────────────────────────────────────────
metrics = model.evaluate(X_te, y_te, verbose=True)

# ── 5. Feature importance ─────────────────────────────────────────────────────
print("\nTop 10 Feature Importances:")
print(f"{'Feature':<25} {'Importance':>10}")
print("-" * 38)
for item in model.feature_importance()[:10]:
    bar = "█" * int(item["importance"] * 300)
    print(f"  {item['feature']:<23} {item['importance']:>10.4f}  {bar}")

# ── 6. Save model ─────────────────────────────────────────────────────────────
model.save(MODEL_DIR)

# ── 7. Save accuracy + metrics JSON (read by app.py dashboard) ───────────────
acc_path = os.path.join(MODEL_DIR, "model_accuracy.json")
met_path = os.path.join(MODEL_DIR, "model_metrics.json")

# Load existing files if present (to preserve other models' entries)
acc_data = {}
met_data = {}
if os.path.exists(acc_path):
    with open(acc_path) as f:
        try: acc_data = json.load(f)
        except: pass
if os.path.exists(met_path):
    with open(met_path) as f:
        try: met_data = json.load(f)
        except: pass

acc_data["gradient_boosting"] = metrics["accuracy"]
met_data["gradient_boosting"] = metrics

with open(acc_path, "w") as f:
    json.dump(acc_data, f, indent=2)
with open(met_path, "w") as f:
    json.dump(met_data, f, indent=2)

print(f"\nAccuracy JSON saved → {acc_path}")
print(f"Metrics JSON saved  → {met_path}")

print("\n" + "="*65)
print(f"  GRADIENT BOOSTING TRAINING COMPLETE")
print(f"  Accuracy  : {metrics['accuracy']*100:.2f}%")
print(f"  Precision : {metrics['precision']*100:.2f}%")
print(f"  Recall    : {metrics['recall']*100:.2f}%")
print(f"  F1 Score  : {metrics['f1']*100:.2f}%")
print(f"  Model dir : {MODEL_DIR}/")
print("="*65)