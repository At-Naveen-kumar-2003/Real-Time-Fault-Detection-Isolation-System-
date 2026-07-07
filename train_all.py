# =============================================================================
#  train_all.py — Train all models in one shot
#
#  Run: python3 train_all.py
#
#  Order:
#    1. Generate dataset (if not exists)
#    2. Train Gradient Boosting
#    3. Train Random Forest
#    4. Print summary
# =============================================================================

import os
import sys
import time

print("=" * 65)
print("  Edge AI — Training All Models")
print("=" * 65)

# ── Step 1: Generate dataset ──────────────────────────────────────────────────
CSV = os.path.join("saved_models", "training_data.csv")
if not os.path.exists(CSV):
    print("\n[1/3] Generating dataset...")
    from generate_dataset import generate_dataset
    generate_dataset(n_per_class=1500)
else:
    print(f"\n[1/3] Dataset exists: {CSV}")

# ── Step 2: Train Gradient Boosting ──────────────────────────────────────────
print("\n[2/3] Training Gradient Boosting...")
t0 = time.time()
import subprocess
result = subprocess.run(
    [sys.executable, "train_gradient_boosting.py"],
    capture_output=False
)
if result.returncode != 0:
    print("  GB training failed — check errors above")
else:
    print(f"  GB done in {time.time()-t0:.1f}s")

# ── Step 3: Train Random Forest ───────────────────────────────────────────────
print("\n[3/3] Training Random Forest...")
t0 = time.time()
result = subprocess.run(
    [sys.executable, "train_random_forest.py"],
    capture_output=False
)
if result.returncode != 0:
    print("  RF training failed — check errors above")
else:
    print(f"  RF done in {time.time()-t0:.1f}s")

# ── Summary ───────────────────────────────────────────────────────────────────
import json
acc_path = os.path.join("saved_models", "model_accuracy.json")
if os.path.exists(acc_path):
    with open(acc_path) as f:
        acc = json.load(f)
    print("\n" + "=" * 65)
    print("  TRAINING COMPLETE")
    print("=" * 65)
    gb_acc = acc.get("gradient_boosting", 0)
    rf_acc = acc.get("random_forest", 0)
    print(f"  Gradient Boosting : {gb_acc*100:.2f}%")
    print(f"  Random Forest     : {rf_acc*100:.2f}%")
    print(f"\n  Run: python3 app.py")
    print(f"  Open: http://localhost:5000")
    print("=" * 65)
