# =============================================================================
#  compare_models.py — Compare all 3 ML algorithms with graphs
#  Run: python3 compare_models.py
#  Outputs: saved_models/comparison_report.html
# =============================================================================
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score, precision_score,
                              recall_score, roc_auc_score)
from sklearn.preprocessing import MinMaxScaler, label_binarize
from sklearn.ensemble import (GradientBoostingClassifier,
                               RandomForestClassifier, IsolationForest)
import joblib, time, warnings
warnings.filterwarnings('ignore')

from config import FEATURE_NAMES, ATTACK_LABELS, MODEL_DIR, N_CLASSES

os.makedirs(MODEL_DIR, exist_ok=True)
GRAPH_DIR = f"{MODEL_DIR}/graphs"
os.makedirs(GRAPH_DIR, exist_ok=True)

COLORS = ['#22c55e','#ef4444','#f97316','#eab308','#a855f7',
          '#78716c','#dc2626','#b45309','#0284c7','#db2777','#7c3aed']

CLASS_NAMES = [ATTACK_LABELS[i] for i in range(N_CLASSES)]

print("="*65)
print("  ML Algorithm Comparison — Edge AI Fault Detection")
print("="*65)

# ── 1. Load data ──────────────────────────────────────────────────────────────
CSV = f"{MODEL_DIR}/training_data.csv"
df  = pd.read_csv(CSV)
X   = df[FEATURE_NAMES].values.astype(np.float32)
y   = df["label"].values.astype(np.int64)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

scaler  = MinMaxScaler()
X_tr_n  = scaler.fit_transform(X_tr)
X_te_n  = scaler.transform(X_te)
print(f"Train: {len(X_tr):,}   Test: {len(X_te):,}")

# ── 2. Train all 3 models ─────────────────────────────────────────────────────
models  = {}
times   = {}
results = {}

# Model 1: Gradient Boosting
print("\n[1/3] Training Gradient Boosting...")
t0 = time.time()
gb = GradientBoostingClassifier(n_estimators=100, max_depth=5,
                                 learning_rate=0.1, random_state=42)
gb.fit(X_tr_n, y_tr)
times['Gradient Boosting'] = time.time() - t0
models['Gradient Boosting'] = gb
print(f"  Done in {times['Gradient Boosting']:.1f}s")

# Model 2: Random Forest
print("[2/3] Training Random Forest...")
t0 = time.time()
rf = RandomForestClassifier(n_estimators=200, max_depth=15,
                              n_jobs=-1, random_state=42)
rf.fit(X_tr_n, y_tr)
times['Random Forest'] = time.time() - t0
models['Random Forest'] = rf
print(f"  Done in {times['Random Forest']:.1f}s")

# Model 3: Isolation Forest (binary)
print("[3/3] Training Isolation Forest...")
t0 = time.time()
X_norm = X_tr_n[y_tr == 0]
iso = IsolationForest(n_estimators=100, contamination=0.1,
                       random_state=42, n_jobs=-1)
iso.fit(X_norm)
times['Isolation Forest'] = time.time() - t0
models['Isolation Forest'] = iso
print(f"  Done in {times['Isolation Forest']:.1f}s")

# ── 3. Evaluate supervised models ────────────────────────────────────────────
for name in ['Gradient Boosting', 'Random Forest']:
    m     = models[name]
    preds = m.predict(X_te_n)
    proba = m.predict_proba(X_te_n)
    results[name] = {
        'preds'      : preds,
        'proba'      : proba,
        'acc'        : accuracy_score(y_te, preds) * 100,
        'f1_macro'   : f1_score(y_te, preds, average='macro') * 100,
        'f1_weighted': f1_score(y_te, preds, average='weighted') * 100,
        'precision'  : precision_score(y_te, preds, average='weighted') * 100,
        'recall'     : recall_score(y_te, preds, average='weighted') * 100,
        'cm'         : confusion_matrix(y_te, preds),
        'report'     : classification_report(y_te, preds,
                         target_names=CLASS_NAMES, digits=3, output_dict=True),
        'train_time' : times[name],
        'importances': m.feature_importances_,
    }
    print(f"\n{name}: Accuracy={results[name]['acc']:.2f}%  "
          f"F1={results[name]['f1_macro']:.2f}%")

# Isolation Forest binary evaluation
iso_scores = -iso.score_samples(X_te_n)
threshold  = float(np.percentile(-iso.score_samples(X_tr_n[y_tr==0]), 95))
iso_preds  = (iso_scores > threshold).astype(int)
y_binary   = (y_te != 0).astype(int)
results['Isolation Forest'] = {
    'preds'      : iso_preds,
    'scores'     : iso_scores,
    'threshold'  : threshold,
    'acc'        : accuracy_score(y_binary, iso_preds) * 100,
    'f1_macro'   : f1_score(y_binary, iso_preds, average='macro') * 100,
    'f1_weighted': f1_score(y_binary, iso_preds, average='weighted') * 100,
    'precision'  : precision_score(y_binary, iso_preds, average='weighted') * 100,
    'recall'     : recall_score(y_binary, iso_preds, average='weighted') * 100,
    'tpr'        : recall_score(y_binary, iso_preds, pos_label=1) * 100,
    'tnr'        : recall_score(y_binary, iso_preds, pos_label=0) * 100,
    'cm'         : confusion_matrix(y_binary, iso_preds),
    'train_time' : times['Isolation Forest'],
}
print(f"\nIsolation Forest: Accuracy={results['Isolation Forest']['acc']:.2f}%  "
      f"TPR={results['Isolation Forest']['tpr']:.1f}%  "
      f"TNR={results['Isolation Forest']['tnr']:.1f}%")

# ── 4. Generate all graphs ────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
PALETTE = ['#0284c7', '#22c55e', '#f97316']

# ── Graph 1: Accuracy comparison bar chart ────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Algorithm Performance Comparison', fontsize=14, fontweight='bold')

metrics     = ['acc', 'f1_macro', 'f1_weighted', 'precision', 'recall']
met_names   = ['Accuracy', 'F1 Macro', 'F1 Weighted', 'Precision', 'Recall']
model_names = ['Gradient Boosting', 'Random Forest', 'Isolation Forest']

ax = axes[0]
for i, (name, color) in enumerate(zip(model_names, PALETTE)):
    val = results[name]['acc']
    ax.bar(name.replace(' ', '\n'), val, color=color, alpha=0.85, width=0.5)
    ax.text(i, val + 0.3, f'{val:.1f}%', ha='center', va='bottom',
            fontsize=10, fontweight='bold')
ax.set_ylim(0, 115)
ax.set_ylabel('Accuracy (%)')
ax.set_title('Overall Accuracy')
ax.tick_params(axis='x', labelsize=8)

# ── Graph 2: F1 Score comparison ──────────────────────────────────────────────
ax = axes[1]
x  = np.arange(len(met_names))
w  = 0.25
for i, (name, color) in enumerate(zip(model_names, PALETTE)):
    vals = [results[name].get(m, 0) for m in metrics]
    ax.bar(x + i*w, vals, w, label=name, color=color, alpha=0.85)
ax.set_xticks(x + w)
ax.set_xticklabels(met_names, rotation=25, ha='right', fontsize=8)
ax.set_ylim(0, 115)
ax.set_ylabel('Score (%)')
ax.set_title('Metrics Comparison')
ax.legend(fontsize=7)

# ── Graph 3: Training time ────────────────────────────────────────────────────
ax = axes[2]
t_vals = [times[n] for n in model_names]
bars   = ax.bar([n.replace(' ', '\n') for n in model_names],
                t_vals, color=PALETTE, alpha=0.85, width=0.5)
for bar, val in zip(bars, t_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'{val:.1f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylabel('Training Time (seconds)')
ax.set_title('Training Time')
ax.tick_params(axis='x', labelsize=8)

plt.tight_layout()
plt.savefig(f'{GRAPH_DIR}/1_performance_comparison.png', dpi=120, bbox_inches='tight')
plt.close()
print("\nGraph 1 saved: performance_comparison.png")

# ── Graph 2: Confusion matrices ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Confusion Matrices', fontsize=14, fontweight='bold')

for idx, name in enumerate(model_names):
    ax = axes[idx]
    cm = results[name]['cm']
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(name, fontsize=10)

    if name == 'Isolation Forest':
        labels = ['Normal', 'Anomaly']
    else:
        labels = [l[:8] for l in CLASS_NAMES]

    ticks = np.arange(len(labels))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_ylabel('True Label', fontsize=8)
    ax.set_xlabel('Predicted Label', fontsize=8)

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center', fontsize=6,
                    color='white' if cm[i, j] > thresh else 'black')

plt.tight_layout()
plt.savefig(f'{GRAPH_DIR}/2_confusion_matrices.png', dpi=120, bbox_inches='tight')
plt.close()
print("Graph 2 saved: confusion_matrices.png")

# ── Graph 3: Per-class F1 scores ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Per-Class F1 Scores', fontsize=14, fontweight='bold')

for idx, name in enumerate(['Gradient Boosting', 'Random Forest']):
    ax     = axes[idx]
    report = results[name]['report']
    f1s    = [report[c]['f1-score'] * 100 for c in CLASS_NAMES]
    bars   = ax.barh(CLASS_NAMES, f1s,
                     color=[COLORS[i] for i in range(N_CLASSES)], alpha=0.85)
    for bar, val in zip(bars, f1s):
        ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=8)
    ax.set_xlim(0, 115)
    ax.set_xlabel('F1 Score (%)')
    ax.set_title(name)
    ax.tick_params(axis='y', labelsize=8)

plt.tight_layout()
plt.savefig(f'{GRAPH_DIR}/3_per_class_f1.png', dpi=120, bbox_inches='tight')
plt.close()
print("Graph 3 saved: per_class_f1.png")

# ── Graph 4: Feature importance comparison ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Top 10 Feature Importances', fontsize=14, fontweight='bold')

for idx, name in enumerate(['Gradient Boosting', 'Random Forest']):
    ax        = axes[idx]
    imp       = results[name]['importances']
    top       = np.argsort(imp)[::-1][:10]
    top_names = [FEATURE_NAMES[i] for i in top]
    top_vals  = [imp[i] * 100 for i in top]
    bars      = ax.barh(top_names[::-1], top_vals[::-1], color='#0284c7', alpha=0.8)
    for bar, val in zip(bars, top_vals[::-1]):
        ax.text(val + 0.1, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}%', va='center', fontsize=8)
    ax.set_xlabel('Importance (%)')
    ax.set_title(name)

plt.tight_layout()
plt.savefig(f'{GRAPH_DIR}/4_feature_importance.png', dpi=120, bbox_inches='tight')
plt.close()
print("Graph 4 saved: feature_importance.png")

# ── Graph 5: Class distribution ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
counts  = [int((y == i).sum()) for i in range(N_CLASSES)]
bars    = ax.bar(CLASS_NAMES, counts, color=COLORS, alpha=0.85)
for bar, val in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
            str(val), ha='center', va='bottom', fontsize=9)
ax.set_xlabel('Class')
ax.set_ylabel('Sample Count')
ax.set_title('Dataset Class Distribution', fontsize=12, fontweight='bold')
ax.tick_params(axis='x', rotation=35, labelsize=8)
plt.tight_layout()
plt.savefig(f'{GRAPH_DIR}/5_class_distribution.png', dpi=120, bbox_inches='tight')
plt.close()
print("Graph 5 saved: class_distribution.png")

# ── Graph 6: Isolation Forest anomaly score distribution ─────────────────────
fig, ax       = plt.subplots(figsize=(10, 5))
scores_normal = results['Isolation Forest']['scores'][y_te == 0]
scores_fault  = results['Isolation Forest']['scores'][y_te != 0]
ax.hist(scores_normal, bins=50, alpha=0.6, color='#22c55e', label='Normal')
ax.hist(scores_fault,  bins=50, alpha=0.6, color='#ef4444', label='Fault/Attack')
ax.axvline(results['Isolation Forest']['threshold'],
           color='black', linestyle='--', linewidth=2,
           label=f"Threshold={results['Isolation Forest']['threshold']:.4f}")
ax.set_xlabel('Anomaly Score')
ax.set_ylabel('Count')
ax.set_title('Isolation Forest — Anomaly Score Distribution',
             fontsize=12, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(f'{GRAPH_DIR}/6_isolation_forest_scores.png', dpi=120, bbox_inches='tight')
plt.close()
print("Graph 6 saved: isolation_forest_scores.png")

# ── Graph 7: Radar chart of metrics ──────────────────────────────────────────
fig = plt.figure(figsize=(8, 8))
ax  = fig.add_subplot(111, polar=True)

categories = ['Accuracy', 'F1 Macro', 'F1 Weighted', 'Precision', 'Recall']
N          = len(categories)
angles     = [n / float(N) * 2 * np.pi for n in range(N)]
angles    += angles[:1]

for name, color in zip(['Gradient Boosting', 'Random Forest'], ['#0284c7', '#22c55e']):
    vals  = [results[name]['acc'],
             results[name]['f1_macro'],
             results[name]['f1_weighted'],
             results[name]['precision'],
             results[name]['recall']]
    vals += vals[:1]
    ax.plot(angles, vals, 'o-', linewidth=2, color=color, label=name)
    ax.fill(angles, vals, alpha=0.15, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10)
ax.set_ylim(0, 100)
ax.set_title('Algorithm Comparison — Radar Chart',
             fontsize=12, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig(f'{GRAPH_DIR}/7_radar_chart.png', dpi=120, bbox_inches='tight')
plt.close()
print("Graph 7 saved: radar_chart.png")

# ── 5. Generate HTML report ───────────────────────────────────────────────────
def img_tag(fname):
    return f'<img src="graphs/{fname}" style="width:100%;border-radius:8px;margin:8px 0">'

html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>ML Algorithm Comparison Report</title>
<style>
body{{font-family:monospace;background:#0f172a;color:#e2e8f0;padding:24px;margin:0}}
h1{{color:#38bdf8;text-align:center;border-bottom:1px solid #1e3a5f;padding-bottom:12px}}
h2{{color:#7dd3fc;margin-top:32px}}
h3{{color:#94a3b8}}
.grid{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin:16px 0}}
.card{{background:#1e293b;border-radius:8px;padding:16px;border:1px solid #334155}}
.card h3{{margin:0 0 8px;font-size:0.9em;color:#94a3b8}}
.big{{font-size:2em;font-weight:bold;color:#38bdf8}}
.green{{color:#22c55e}}.red{{color:#ef4444}}.orange{{color:#f97316}}
table{{width:100%;border-collapse:collapse;font-size:0.85em;margin:12px 0}}
th{{background:#1e3a5f;padding:8px;text-align:left;color:#7dd3fc}}
td{{padding:7px 8px;border-bottom:1px solid #1e293b}}
tr:hover td{{background:#1e293b}}
.badge{{display:inline-block;padding:3px 10px;border-radius:10px;font-size:0.75em}}
.best{{background:#14532d;color:#4ade80}}
.good{{background:#1e3a5f;color:#7dd3fc}}
.img-full{{margin:16px 0}}
</style>
</head>
<body>
<h1>ML Algorithm Comparison Report — Edge AI Fault Detection</h1>

<h2>Summary Metrics</h2>
<div class="grid">
  <div class="card">
    <h3>Gradient Boosting</h3>
    <div class="big green">{results['Gradient Boosting']['acc']:.2f}%</div>
    <p>Accuracy</p>
    <p>F1 Macro: {results['Gradient Boosting']['f1_macro']:.2f}%</p>
    <p>Train time: {times['Gradient Boosting']:.1f}s</p>
    <span class="badge best">Best Accuracy</span>
  </div>
  <div class="card">
    <h3>Random Forest</h3>
    <div class="big orange">{results['Random Forest']['acc']:.2f}%</div>
    <p>Accuracy</p>
    <p>F1 Macro: {results['Random Forest']['f1_macro']:.2f}%</p>
    <p>Train time: {times['Random Forest']:.1f}s</p>
    <span class="badge good">Multi-class</span>
  </div>
  <div class="card">
    <h3>Isolation Forest</h3>
    <div class="big" style="color:#a855f7">{results['Isolation Forest']['acc']:.2f}%</div>
    <p>Binary Accuracy</p>
    <p>TPR: {results['Isolation Forest']['tpr']:.1f}%  TNR: {results['Isolation Forest']['tnr']:.1f}%</p>
    <p>Train time: {times['Isolation Forest']:.1f}s</p>
    <span class="badge good">Unsupervised</span>
  </div>
</div>

<h2>Comparison Table</h2>
<table>
<tr><th>Metric</th><th>Gradient Boosting</th><th>Random Forest</th><th>Isolation Forest</th></tr>
<tr><td>Accuracy</td><td class="green">{results['Gradient Boosting']['acc']:.2f}%</td>
    <td>{results['Random Forest']['acc']:.2f}%</td>
    <td>{results['Isolation Forest']['acc']:.2f}% (binary)</td></tr>
<tr><td>F1 Macro</td><td class="green">{results['Gradient Boosting']['f1_macro']:.2f}%</td>
    <td>{results['Random Forest']['f1_macro']:.2f}%</td>
    <td>{results['Isolation Forest']['f1_macro']:.2f}% (binary)</td></tr>
<tr><td>F1 Weighted</td><td>{results['Gradient Boosting']['f1_weighted']:.2f}%</td>
    <td>{results['Random Forest']['f1_weighted']:.2f}%</td>
    <td>{results['Isolation Forest']['f1_weighted']:.2f}%</td></tr>
<tr><td>Precision</td><td>{results['Gradient Boosting']['precision']:.2f}%</td>
    <td>{results['Random Forest']['precision']:.2f}%</td>
    <td>{results['Isolation Forest']['precision']:.2f}%</td></tr>
<tr><td>Recall</td><td>{results['Gradient Boosting']['recall']:.2f}%</td>
    <td>{results['Random Forest']['recall']:.2f}%</td>
    <td>{results['Isolation Forest']['recall']:.2f}%</td></tr>
<tr><td>Training Time</td><td>{times['Gradient Boosting']:.1f}s</td>
    <td>{times['Random Forest']:.1f}s</td>
    <td>{times['Isolation Forest']:.1f}s</td></tr>
<tr><td>Type</td><td>Supervised</td><td>Supervised</td><td>Unsupervised</td></tr>
<tr><td>Classes</td><td>11</td><td>11</td><td>2 (binary)</td></tr>
</table>

<h2>Performance Graphs</h2>
<div class="img-full">{img_tag('1_performance_comparison.png')}</div>
<div class="img-full">{img_tag('7_radar_chart.png')}</div>
<div class="img-full">{img_tag('2_confusion_matrices.png')}</div>
<div class="img-full">{img_tag('3_per_class_f1.png')}</div>
<div class="img-full">{img_tag('4_feature_importance.png')}</div>
<div class="img-full">{img_tag('5_class_distribution.png')}</div>
<div class="img-full">{img_tag('6_isolation_forest_scores.png')}</div>

<h2>Per-Class Report — Gradient Boosting</h2>
<table>
<tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr>
"""
for cls in CLASS_NAMES:
    r = results['Gradient Boosting']['report'][cls]
    html += f"<tr><td>{cls}</td><td>{r['precision']*100:.1f}%</td>"
    html += f"<td>{r['recall']*100:.1f}%</td><td>{r['f1-score']*100:.1f}%</td>"
    html += f"<td>{int(r['support'])}</td></tr>\n"

html += f"""</table>

<h2>Per-Class Report — Random Forest</h2>
<table>
<tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr>
"""
for cls in CLASS_NAMES:
    r = results['Random Forest']['report'][cls]
    html += f"<tr><td>{cls}</td><td>{r['precision']*100:.1f}%</td>"
    html += f"<td>{r['recall']*100:.1f}%</td><td>{r['f1-score']*100:.1f}%</td>"
    html += f"<td>{int(r['support'])}</td></tr>\n"

html += """</table>
<p style="color:#475569;text-align:center;margin-top:32px;font-size:0.8em">
  Edge AI Fault Detection — Algorithm Comparison Report
</p>
</body></html>"""

report_path = f"{MODEL_DIR}/comparison_report.html"
with open(report_path, 'w') as f:
    f.write(html)

print(f"\nHTML report saved: {report_path}")
print("\n" + "="*65)
print("  COMPARISON COMPLETE")
print(f"  Open: saved_models/comparison_report.html")
print(f"  Graphs in: saved_models/graphs/")
print("="*65)