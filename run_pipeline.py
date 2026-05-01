#!/usr/bin/env python3
"""
Run the Polymarket Anomaly Detection Pipeline with correct feature columns.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

output_dir = Path('./results')
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print(" POLYMARKET ANOMALY DETECTION PIPELINE")
print("=" * 80)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# ── Step 1: Load data ──────────────────────────────────────────
print("\n[1/6] LOADING DATA")
print("-" * 72)
df = pd.read_parquet('data/ml_features_1m_v2.parquet')
print(f"Loaded features: {df.shape[0]:,} rows, {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")

# ── Step 2: Prepare features ──────────────────────────────────
print("\n[2/6] PREPARING FEATURES")
print("-" * 72)
feature_cols = [
    'mean_spread',
    'bar_volatility',
    'order_flow_imbalance',
    'bid_depth',
    'ask_depth',
    'depth_imbalance',
    'trade_count',
]
print(f"Selected {len(feature_cols)} features: {feature_cols}")

X = df[feature_cols].copy()
X = X.fillna(X.median())
X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
print(f"Normalized {len(feature_cols)} features via RobustScaler")
print(f"Shape: {X_scaled_df.shape}")

# ── Step 3: Train model ──────────────────────────────────────
print("\n[3/6] TRAINING MODEL")
print("-" * 72)
n = len(X_scaled_df)

# Subsample for speed if dataset is very large
if n > 500_000:
    sample_idx = np.random.RandomState(42).choice(n, 500_000, replace=False)
    sample_idx.sort()
    X_train = X_scaled_df.iloc[sample_idx]
    print(f"Subsampled to {len(X_train):,} for training (full data: {n:,})")
else:
    X_train = X_scaled_df
    print(f"Training on {len(X_train):,} samples")

model = IsolationForest(
    contamination=0.01,
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train)
print("Model trained!")

# ── Step 4: Detect anomalies on FULL data ─────────────────────
print("\n[4/6] DETECTING ANOMALIES")
print("-" * 72)
scores = model.score_samples(X_scaled_df)
predictions = model.predict(X_scaled_df)

n_anomalies = int((predictions == -1).sum())
summary = {
    'total_samples': int(n),
    'n_anomalies': n_anomalies,
    'anomaly_percentage': round(n_anomalies / n * 100, 4),
    'anomaly_score_min': round(float(scores.min()), 6),
    'anomaly_score_max': round(float(scores.max()), 6),
    'anomaly_score_mean': round(float(scores.mean()), 6),
    'anomaly_score_std': round(float(scores.std()), 6),
    'threshold': round(float(np.percentile(scores, 1)), 6),
}
print(f"Total samples: {summary['total_samples']:,}")
print(f"Anomalies detected: {summary['n_anomalies']:,} ({summary['anomaly_percentage']:.2f}%)")
print(f"Score range: [{summary['anomaly_score_min']:.4f}, {summary['anomaly_score_max']:.4f}]")
print(f"Detection threshold: {summary['threshold']:.4f}")

# ── Step 5: Analyze ──────────────────────────────────────────
print("\n[5/6] ANALYZING RESULTS")
print("-" * 72)

top_idx = np.argsort(scores)[:10]
print("Top 10 Most Anomalous Events:")
print("=" * 80)
for rank, idx in enumerate(top_idx, 1):
    row = df.iloc[idx]
    print(f"\n{rank}. Index {idx} | Score: {scores[idx]:.4f}")
    print(f"   market_id: {row['market_id'][:20]}...")
    print(f"   timestamp: {row['minute_bar']}")
    print(f"   mean_spread={row['mean_spread']:.4f}, volatility={row['bar_volatility']:.4f}")
    print(f"   ofi={row['order_flow_imbalance']:.4f}, depth_imb={row['depth_imbalance']:.4f}")
    print(f"   bid_depth={row['bid_depth']:.2f}, ask_depth={row['ask_depth']:.2f}")

# ── Step 6: Visualizations ────────────────────────────────────
print("\n[6/6] CREATING VISUALIZATIONS")
print("-" * 72)

# 1. Anomaly timeline
fig, ax = plt.subplots(figsize=(14, 5))
# Subsample for plotting
plot_step = max(1, len(scores) // 50000)
plot_scores = scores[::plot_step]
ax.plot(plot_scores, color='steelblue', alpha=0.6, linewidth=0.5)
ax.fill_between(range(len(plot_scores)), plot_scores, alpha=0.2, color='steelblue')

threshold = np.percentile(scores, 1)
anomaly_mask = plot_scores < threshold
anomaly_plot_idx = np.where(anomaly_mask)[0]
if len(anomaly_plot_idx) > 0:
    ax.scatter(anomaly_plot_idx, plot_scores[anomaly_plot_idx],
               color='red', s=30, marker='*', zorder=5, label='Anomalies')

ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.5, label=f'Threshold (1%)')
ax.set_xlabel('Time (samples)', fontsize=12)
ax.set_ylabel('Anomaly Score (lower = more anomalous)', fontsize=12)
ax.set_title('Polymarket: High-Frequency Anomaly Detection Timeline', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig(output_dir / '01_anomaly_timeline.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir / '01_anomaly_timeline.png'}")

# 2. Score distribution
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(scores, bins=100, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.3)
ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2,
           label=f'1% Threshold ({threshold:.4f})')
ax.set_xlabel('Anomaly Score', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Anomaly Scores', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.savefig(output_dir / '02_score_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir / '02_score_distribution.png'}")

# 3. Feature importance (which features are most anomalous)
fig, ax = plt.subplots(figsize=(10, 5))
anomalous_rows = X_scaled_df.iloc[np.where(predictions == -1)[0]]
normal_rows = X_scaled_df.iloc[np.where(predictions == 1)[0]]
feature_diff = anomalous_rows.mean() - normal_rows.mean()
feature_diff_sorted = feature_diff.abs().sort_values(ascending=True)
colors = ['#e74c3c' if v > 0 else '#3498db' for v in feature_diff[feature_diff_sorted.index]]
ax.barh(feature_diff_sorted.index, feature_diff[feature_diff_sorted.index], color=colors, alpha=0.8)
ax.set_xlabel('Mean Difference (Anomalous - Normal)', fontsize=12)
ax.set_title('Feature Importance: What Makes an Anomaly?', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
plt.savefig(output_dir / '03_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir / '03_feature_importance.png'}")

# ── Save results ─────────────────────────────────────────────
print("\nSAVING RESULTS")
print("-" * 72)

# Save summary
with open(output_dir / 'summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"  Saved: {output_dir / 'summary.json'}")

# Save top anomalies
df_copy = df.copy()
df_copy['anomaly_score'] = scores
df_copy['is_anomaly'] = predictions == -1
top100 = df_copy.nsmallest(100, 'anomaly_score')
top100.to_csv(output_dir / 'top_anomalies.csv', index=False)
print(f"  Saved: {output_dir / 'top_anomalies.csv'}")

# Save model info
model_info = {
    'model_type': 'IsolationForest',
    'contamination': 0.01,
    'n_estimators': 100,
    'features_used': feature_cols,
    'training_samples': int(len(X_train)),
    'total_samples_scored': int(n),
    'anomalies_detected': n_anomalies,
}
with open(output_dir / 'model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)
print(f"  Saved: {output_dir / 'model_info.json'}")

print("\n" + "=" * 80)
print(" PIPELINE COMPLETE!")
print("=" * 80)
print(f"\nResults saved to: {output_dir}")
print(f"  - summary.json          : Detection summary and stats")
print(f"  - top_anomalies.csv     : Top 100 anomalous events with features")
print(f"  - model_info.json       : Model configuration")
print(f"  - 01_anomaly_timeline.png  : Timeline visualization")
print(f"  - 02_score_distribution.png: Score distribution")
print(f"  - 03_feature_importance.png: Feature importance chart")
