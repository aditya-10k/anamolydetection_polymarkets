#!/usr/bin/env python3
"""
Contamination Rate Validation for Isolation Forest.

Tests multiple contamination rates and evaluates using:
  1. Silhouette Score — cluster separation quality
  2. Calinski-Harabasz Index — ratio of between-cluster to within-cluster variance
  3. Davies-Bouldin Index — average similarity between clusters (lower = better)
  4. Score Elbow Analysis — natural breakpoint in sorted anomaly scores
  5. Stability Analysis — overlap of top anomalies across thresholds
  6. Feature Separation — how distinct anomalous samples are from normal ones
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

output_dir = Path('./results/contamination_validation')
output_dir.mkdir(parents=True, exist_ok=True)

CONTAMINATION_RATES = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05]
EVAL_SAMPLE_SIZE = 50_000

print("=" * 80)
print(" CONTAMINATION RATE VALIDATION")
print("=" * 80)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Rates to test: {CONTAMINATION_RATES}")
print("=" * 80)

print("\n[1/7] LOADING & PREPARING DATA")
print("-" * 72)
df = pd.read_parquet('data/ml_features_1m_v2.parquet')
print(f"Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

feature_cols = [
    'mean_spread', 'bar_volatility', 'order_flow_imbalance',
    'bid_depth', 'ask_depth', 'depth_imbalance', 'trade_count',
]

X = df[feature_cols].copy()
X = X.fillna(X.median())
X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
n = len(X_scaled_df)
print(f"Prepared {n:,} samples with {len(feature_cols)} features")

print("\n[2/7] TRAINING MODEL & SCORING")
print("-" * 72)
train_size = min(500_000, n)
sample_idx = np.random.RandomState(42).choice(n, train_size, replace=False)
sample_idx.sort()
X_train = X_scaled_df.iloc[sample_idx]
print(f"Training on {len(X_train):,} subsampled rows")

model = IsolationForest(
    contamination='auto',
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train)
scores = model.score_samples(X_scaled_df)
print(f"Scored all {n:,} samples")
print(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")

print("\n[3/7] COMPUTING VALIDATION METRICS")
print("-" * 72)

np.random.seed(42)
eval_idx = np.random.choice(n, EVAL_SAMPLE_SIZE, replace=False)
X_eval = X_scaled_df.iloc[eval_idx].values
scores_eval = scores[eval_idx]

results = []

for c in CONTAMINATION_RATES:
    threshold = np.percentile(scores, c * 100)
    labels_full = (scores < threshold).astype(int)
    labels_eval = (scores_eval < threshold).astype(int)

    n_anomalies = int(labels_full.sum())
    n_normal = n - n_anomalies

    n_anom_eval = int(labels_eval.sum())
    n_norm_eval = EVAL_SAMPLE_SIZE - n_anom_eval

    if n_anom_eval < 2 or n_norm_eval < 2:
        print(f"  c={c:.3f}: Skipped (insufficient samples in one class)")
        continue

    sil = silhouette_score(X_eval, labels_eval, sample_size=min(10_000, EVAL_SAMPLE_SIZE))
    ch = calinski_harabasz_score(X_eval, labels_eval)
    db = davies_bouldin_score(X_eval, labels_eval)

    anom_mask_full = labels_full == 1
    normal_mask_full = labels_full == 0
    anom_features = X_scaled_df.iloc[np.where(anom_mask_full)[0]]
    normal_features = X_scaled_df.iloc[np.where(normal_mask_full)[0]]
    feature_separation = np.mean(np.abs(anom_features.mean() - normal_features.mean()))

    top_100_idx = set(np.argsort(scores)[:100])
    flagged_idx = set(np.where(anom_mask_full)[0])
    top100_recall = len(top_100_idx & flagged_idx) / 100.0

    result = {
        'contamination': c,
        'threshold': float(threshold),
        'n_anomalies': n_anomalies,
        'anomaly_pct': round(n_anomalies / n * 100, 3),
        'silhouette': round(sil, 5),
        'calinski_harabasz': round(ch, 2),
        'davies_bouldin': round(db, 5),
        'feature_separation': round(feature_separation, 4),
        'top100_recall': round(top100_recall, 3),
    }
    results.append(result)

    print(f"  c={c:.3f} | threshold={threshold:.4f} | anomalies={n_anomalies:>7,} "
          f"| silhouette={sil:.4f} | CH={ch:>10.1f} | DB={db:.4f} | sep={feature_separation:.3f}")

print("\n[4/7] STABILITY ANALYSIS")
print("-" * 72)

top_sets = {}
for c in CONTAMINATION_RATES:
    threshold = np.percentile(scores, c * 100)
    flagged = set(np.where(scores < threshold)[0])
    top_sets[c] = flagged

stability_matrix = np.zeros((len(CONTAMINATION_RATES), len(CONTAMINATION_RATES)))
for i, c1 in enumerate(CONTAMINATION_RATES):
    for j, c2 in enumerate(CONTAMINATION_RATES):
        smaller = min(len(top_sets[c1]), len(top_sets[c2]))
        if smaller > 0:
            overlap = len(top_sets[c1] & top_sets[c2])
            stability_matrix[i, j] = overlap / smaller
        else:
            stability_matrix[i, j] = 0.0

print("Jaccard-style overlap (intersection / smaller set):")
header = "         " + "  ".join([f"c={c:.3f}" for c in CONTAMINATION_RATES])
print(header)
for i, c1 in enumerate(CONTAMINATION_RATES):
    row_str = f"c={c1:.3f}  " + "  ".join([f"{stability_matrix[i,j]:.3f}  " for j in range(len(CONTAMINATION_RATES))])
    print(row_str)

print("\n[5/7] ELBOW ANALYSIS")
print("-" * 72)

sorted_scores = np.sort(scores)
percentiles = np.arange(0.1, 10.1, 0.1)
percentile_thresholds = [np.percentile(scores, p) for p in percentiles]

diffs = np.diff(percentile_thresholds)
second_diffs = np.diff(diffs)

elbow_idx = np.argmax(np.abs(second_diffs)) + 1
elbow_pct = percentiles[elbow_idx]
elbow_threshold = percentile_thresholds[elbow_idx]
print(f"Detected elbow at: {elbow_pct:.1f}% (threshold={elbow_threshold:.4f})")
print(f"This suggests a natural contamination rate around {elbow_pct/100:.3f}")

print("\n[6/7] SELECTING OPTIMAL CONTAMINATION")
print("-" * 72)

if results:
    for r in results:
        sil_norm = (r['silhouette'] - min(x['silhouette'] for x in results)) / (max(x['silhouette'] for x in results) - min(x['silhouette'] for x in results) + 1e-10)
        ch_norm = (r['calinski_harabasz'] - min(x['calinski_harabasz'] for x in results)) / (max(x['calinski_harabasz'] for x in results) - min(x['calinski_harabasz'] for x in results) + 1e-10)
        db_norm = 1.0 - (r['davies_bouldin'] - min(x['davies_bouldin'] for x in results)) / (max(x['davies_bouldin'] for x in results) - min(x['davies_bouldin'] for x in results) + 1e-10)
        sep_norm = (r['feature_separation'] - min(x['feature_separation'] for x in results)) / (max(x['feature_separation'] for x in results) - min(x['feature_separation'] for x in results) + 1e-10)

        r['composite_score'] = round(0.3 * sil_norm + 0.25 * ch_norm + 0.25 * db_norm + 0.2 * sep_norm, 4)

    best = max(results, key=lambda x: x['composite_score'])
    print(f"\nComposite scoring (Silhouette 30%, CH 25%, DB 25%, Feature Sep 20%):")
    print(f"{'Rate':>8} {'Silhouette':>12} {'CH Index':>12} {'DB Index':>10} {'Feat Sep':>10} {'Composite':>10}")
    print("-" * 65)
    for r in results:
        marker = " <<<" if r['contamination'] == best['contamination'] else ""
        print(f"{r['contamination']:>8.3f} {r['silhouette']:>12.5f} {r['calinski_harabasz']:>12.1f} "
              f"{r['davies_bouldin']:>10.5f} {r['feature_separation']:>10.4f} {r['composite_score']:>10.4f}{marker}")

    optimal_c = best['contamination']
    print(f"\n>>> OPTIMAL CONTAMINATION RATE: {optimal_c}")
    print(f"    Threshold: {best['threshold']:.4f}")
    print(f"    Anomalies: {best['n_anomalies']:,} ({best['anomaly_pct']}%)")

print("\n[7/7] GENERATING VISUALIZATIONS")
print("-" * 72)

fig = plt.figure(figsize=(22, 18))
gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
fig.suptitle('Contamination Rate Validation — Isolation Forest', fontsize=18, fontweight='bold', y=0.98)

ax1 = fig.add_subplot(gs[0, 0])
sil_vals = [r['silhouette'] for r in results]
c_vals = [r['contamination'] for r in results]
bars = ax1.bar([f"{c:.1%}" for c in c_vals], sil_vals, color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5)
best_idx = c_vals.index(optimal_c)
bars[best_idx].set_color('#e74c3c')
bars[best_idx].set_edgecolor('darkred')
ax1.set_xlabel('Contamination Rate', fontsize=11)
ax1.set_ylabel('Silhouette Score', fontsize=11)
ax1.set_title('Silhouette Score\n(higher = better separation)', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(sil_vals):
    ax1.text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax2 = fig.add_subplot(gs[0, 1])
ch_vals = [r['calinski_harabasz'] for r in results]
bars = ax2.bar([f"{c:.1%}" for c in c_vals], ch_vals, color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=0.5)
bars[best_idx].set_color('#e74c3c')
bars[best_idx].set_edgecolor('darkred')
ax2.set_xlabel('Contamination Rate', fontsize=11)
ax2.set_ylabel('Calinski-Harabasz Index', fontsize=11)
ax2.set_title('Calinski-Harabasz Index\n(higher = better defined clusters)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(ch_vals):
    ax2.text(i, v + max(ch_vals)*0.01, f'{v:.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax3 = fig.add_subplot(gs[0, 2])
db_vals = [r['davies_bouldin'] for r in results]
bars = ax3.bar([f"{c:.1%}" for c in c_vals], db_vals, color='#e67e22', alpha=0.8, edgecolor='black', linewidth=0.5)
bars[best_idx].set_color('#e74c3c')
bars[best_idx].set_edgecolor('darkred')
ax3.set_xlabel('Contamination Rate', fontsize=11)
ax3.set_ylabel('Davies-Bouldin Index', fontsize=11)
ax3.set_title('Davies-Bouldin Index\n(lower = better separation)', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(db_vals):
    ax3.text(i, v + max(db_vals)*0.01, f'{v:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax4 = fig.add_subplot(gs[1, 0])
composite_vals = [r['composite_score'] for r in results]
bars = ax4.bar([f"{c:.1%}" for c in c_vals], composite_vals, color='#9b59b6', alpha=0.8, edgecolor='black', linewidth=0.5)
bars[best_idx].set_color('#e74c3c')
bars[best_idx].set_edgecolor('darkred')
ax4.set_xlabel('Contamination Rate', fontsize=11)
ax4.set_ylabel('Composite Score', fontsize=11)
ax4.set_title('Composite Validation Score\n(higher = better overall)', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(composite_vals):
    ax4.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax5 = fig.add_subplot(gs[1, 1])
n_points = min(200_000, n)
step = max(1, n // n_points)
sorted_sub = np.sort(scores[::step])
ax5.plot(np.linspace(0, 100, len(sorted_sub)), sorted_sub, color='steelblue', linewidth=1.5)
colors_map = ['#1abc9c', '#e74c3c', '#3498db', '#e67e22', '#9b59b6', '#f39c12']
for i, c in enumerate(CONTAMINATION_RATES):
    thr = np.percentile(scores, c * 100)
    color = colors_map[i % len(colors_map)]
    ax5.axvline(x=c*100, color=color, linestyle='--', alpha=0.8, linewidth=1.5)
    ax5.axhline(y=thr, color=color, linestyle=':', alpha=0.4, linewidth=1)
    ax5.annotate(f'c={c:.1%}\n(thr={thr:.3f})', xy=(c*100, thr),
                fontsize=7, color=color, fontweight='bold',
                xytext=(c*100 + 2, thr + 0.01))
ax5.set_xlabel('Percentile', fontsize=11)
ax5.set_ylabel('Anomaly Score', fontsize=11)
ax5.set_title('Score Distribution with Thresholds\n(elbow = natural cutoff)', fontsize=12, fontweight='bold')
ax5.set_xlim(0, 15)
ax5.grid(True, alpha=0.3)

ax6 = fig.add_subplot(gs[1, 2])
sep_vals = [r['feature_separation'] for r in results]
bars = ax6.bar([f"{c:.1%}" for c in c_vals], sep_vals, color='#1abc9c', alpha=0.8, edgecolor='black', linewidth=0.5)
bars[best_idx].set_color('#e74c3c')
bars[best_idx].set_edgecolor('darkred')
ax6.set_xlabel('Contamination Rate', fontsize=11)
ax6.set_ylabel('Mean Feature Separation', fontsize=11)
ax6.set_title('Feature Separation\n(higher = anomalies more distinct)', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(sep_vals):
    ax6.text(i, v + max(sep_vals)*0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax7 = fig.add_subplot(gs[2, 0])
im = ax7.imshow(stability_matrix, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
ax7.set_xticks(range(len(CONTAMINATION_RATES)))
ax7.set_yticks(range(len(CONTAMINATION_RATES)))
ax7.set_xticklabels([f"{c:.1%}" for c in CONTAMINATION_RATES], fontsize=9)
ax7.set_yticklabels([f"{c:.1%}" for c in CONTAMINATION_RATES], fontsize=9)
for i in range(len(CONTAMINATION_RATES)):
    for j in range(len(CONTAMINATION_RATES)):
        ax7.text(j, i, f'{stability_matrix[i,j]:.2f}', ha='center', va='center', fontsize=8,
                color='white' if stability_matrix[i,j] > 0.5 else 'black')
ax7.set_title('Anomaly Stability Matrix\n(overlap between thresholds)', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax7, shrink=0.8)

ax8 = fig.add_subplot(gs[2, 1:])
feature_data = {}
for c in [CONTAMINATION_RATES[0], optimal_c, CONTAMINATION_RATES[-1]]:
    thr = np.percentile(scores, c * 100)
    anom_mask = scores < thr
    anom_means = X_scaled_df.iloc[np.where(anom_mask)[0]].mean()
    norm_means = X_scaled_df.iloc[np.where(~anom_mask)[0]].mean()
    feature_data[c] = (anom_means - norm_means).abs()

x_pos = np.arange(len(feature_cols))
width = 0.25
labels_used = []
for i, (c, diff) in enumerate(feature_data.items()):
    label = f'c={c:.1%}' + (' (optimal)' if c == optimal_c else '')
    labels_used.append(label)
    color = '#e74c3c' if c == optimal_c else (['#3498db', '#2ecc71', '#e67e22'][i % 3])
    ax8.bar(x_pos + i * width, diff.values, width, label=label, color=color, alpha=0.8, edgecolor='black', linewidth=0.3)

ax8.set_xlabel('Feature', fontsize=11)
ax8.set_ylabel('|Mean Anomalous - Mean Normal| (scaled)', fontsize=11)
ax8.set_title('Feature Separation by Contamination Rate\n(taller bars = anomalies more distinct on that feature)', fontsize=12, fontweight='bold')
ax8.set_xticks(x_pos + width)
ax8.set_xticklabels(feature_cols, rotation=30, ha='right', fontsize=9)
ax8.legend(fontsize=10)
ax8.grid(True, alpha=0.3, axis='y')

plt.savefig(output_dir / 'contamination_validation.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {output_dir / 'contamination_validation.png'}")

fig2, axes2 = plt.subplots(2, 3, figsize=(20, 10))
fig2.suptitle('Score Distributions at Each Contamination Rate', fontsize=16, fontweight='bold')

for i, c in enumerate(CONTAMINATION_RATES):
    ax = axes2[i // 3, i % 3]
    thr = np.percentile(scores, c * 100)
    ax.hist(scores, bins=100, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.3)
    ax.axvline(x=thr, color='red', linestyle='--', linewidth=2, label=f'Threshold ({thr:.4f})')

    n_anom = int((scores < thr).sum())
    ax.set_title(f'c = {c:.1%} → {n_anom:,} anomalies', fontsize=12, fontweight='bold')
    ax.set_xlabel('Anomaly Score')
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    if c == optimal_c:
        ax.set_facecolor('#fff5f5')
        ax.set_title(f'c = {c:.1%} → {n_anom:,} anomalies ★ OPTIMAL', fontsize=12, fontweight='bold', color='red')

plt.tight_layout()
plt.savefig(output_dir / 'score_distributions.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {output_dir / 'score_distributions.png'}")

validation_report = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'contamination_rates_tested': CONTAMINATION_RATES,
    'optimal_contamination': optimal_c,
    'elbow_suggested_rate': round(elbow_pct / 100, 3),
    'total_samples': int(n),
    'results': results,
}

with open(output_dir / 'validation_report.json', 'w') as f:
    json.dump(validation_report, f, indent=2)
print(f"  Saved: {output_dir / 'validation_report.json'}")

print("\n" + "=" * 80)
print(" VALIDATION COMPLETE")
print("=" * 80)
print(f"\n  Optimal contamination rate: {optimal_c}")
print(f"  Elbow-suggested rate:       {elbow_pct/100:.3f}")
print(f"  Results saved to:           {output_dir}")
print(f"\n  Next step: update run_pipeline.py with contamination={optimal_c}")
