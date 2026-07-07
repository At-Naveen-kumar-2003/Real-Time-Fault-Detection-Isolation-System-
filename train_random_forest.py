# =============================================================================
#  train_random_forest.py — Train and save the Random Forest model
#
#  Run: python3 train_random_forest.py
#  Output:
#    saved_models/random_forest_model.pkl
#    saved_models/random_forest_scaler.pkl
#    saved_models/model_accuracy.json   (updated — preserves GB entry)
#    saved_models/model_metrics.json    (updated — preserves GB entry)
# =============================================================================

import os
import sys
import json
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import FEATURE_NAMES, FAULT_LABELS, N_CLASSES, MODEL_DIR
from random_forest import RandomForestModel

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
print("\nTraining Random Forest model (200 trees)...")
model = RandomForestModel(n_estimators=200, max_depth=12)
model.fit(X_tr, y_tr)

# ── 4. Evaluate ───────────────────────────────────────────────────────────────
metrics = model.evaluate(X_te, y_te, verbose=True)

# ── 5. Save model ─────────────────────────────────────────────────────────────
model.save(MODEL_DIR)

# ── 6. Save accuracy + metrics JSON (read by app.py dashboard) ───────────────
acc_path = os.path.join(MODEL_DIR, "model_accuracy.json")
met_path = os.path.join(MODEL_DIR, "model_metrics.json")

# Load existing files if present (to preserve GB entries)
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

acc_data["random_forest"] = metrics["accuracy"]
met_data["random_forest"] = metrics

with open(acc_path, "w") as f:
    json.dump(acc_data, f, indent=2)
with open(met_path, "w") as f:
    json.dump(met_data, f, indent=2)

print(f"\nAccuracy JSON saved → {acc_path}")
print(f"Metrics JSON saved  → {met_path}")

print("\n" + "="*65)
print(f"  RANDOM FOREST TRAINING COMPLETE")
print(f"  Accuracy  : {metrics['accuracy']*100:.2f}%")
print(f"  Precision : {metrics['precision']*100:.2f}%")
print(f"  Recall    : {metrics['recall']*100:.2f}%")
print(f"  F1 Score  : {metrics['f1']*100:.2f}%")
print(f"  OOB Score : {metrics.get('oob_score', 0)*100:.2f}%")
print(f"  Model dir : {MODEL_DIR}/")
print("="*65)