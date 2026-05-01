#!/usr/bin/env python3
"""
POLYMARKET ANOMALY DETECTION - COMPLETE PIPELINE
===============================================

This single script runs the entire anomaly detection pipeline:
1. Load features
2. Prepare features
3. Train Isolation Forest
4. Detect anomalies
5. Generate visualizations
6. Create summary report

USAGE:
    python polymarket_complete_pipeline.py --data-dir ./data --output-dir ./results

REQUIREMENTS:
    pip install pandas polars pyarrow scikit-learn numpy scipy plotly matplotlib
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
from datetime import datetime
import argparse

# ============================================================================
# PART 1: DATA LOADING
# ============================================================================

class DataLoader:
    """Load Polymarket feature and label data"""
    
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
    
    def load_features(self, filename="features/ml_features_1m_v2.parquet"):
        """Load pre-computed features"""
        path = self.data_dir / filename
        try:
            # Try polars first (faster)
            try:
                import polars as pl
                df = pl.read_parquet(path).to_pandas()
            except:
                # Fall back to pandas
                df = pd.read_parquet(path)
            
            print(f"✓ Loaded features: {df.shape[0]:,} rows, {df.shape[1]} columns")
            return df
        except Exception as e:
            print(f"✗ Error loading {path}: {e}")
            return None
    
    def load_trades(self, filename="labels/trades.parquet"):
        """Load price/trade data"""
        path = self.data_dir / filename
        try:
            try:
                import polars as pl
                df = pl.read_parquet(path).to_pandas()
            except:
                df = pd.read_parquet(path)
            
            print(f"✓ Loaded trades: {df.shape[0]:,} rows")
            return df
        except Exception as e:
            print(f"⚠ Could not load trades ({e})")
            return None

# ============================================================================
# PART 2: FEATURE ENGINEERING
# ============================================================================

class FeaturePreprocessor:
    """Prepare features for anomaly detection"""
    
    @staticmethod
    def find_stress_features(df):
        """Auto-detect market stress indicators"""
        stress_keywords = {
            'spread': ['spread', 'ba', 'bid_ask'],
            'depth': ['depth', 'l2', 'liquidity'],
            'ofi': ['ofi', 'order_flow', 'imbalance'],
            'volatility': ['vol', 'volatility', 'sigma'],
            'momentum': ['momentum', 'price_move', 'delta']
        }
        
        found_features = []
        print("\n📊 Auto-detecting features:")
        
        for category, keywords in stress_keywords.items():
            for col in df.columns:
                col_lower = col.lower()
                if any(kw in col_lower for kw in keywords):
                    found_features.append(col)
                    print(f"  ✓ {category:12} → {col}")
                    break
        
        return found_features if found_features else list(df.columns[1:6])
    
    @staticmethod
    def preprocess(df, feature_cols=None):
        """Normalize features for anomaly detection"""
        if feature_cols is None:
            feature_cols = FeaturePreprocessor.find_stress_features(df)
        
        print(f"\n🔧 Preprocessing {len(feature_cols)} features")
        
        # Extract features
        X = df[feature_cols].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
        
        # Normalize using RobustScaler (better for outliers)
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
        
        print(f"  ✓ Normalized {len(feature_cols)} features")
        print(f"    Mean: {X_scaled_df.mean().round(2).to_dict()}")
        
        return X_scaled_df, feature_cols

# ============================================================================
# PART 3: ANOMALY DETECTION
# ============================================================================

class IsolationForestDetector:
    """Detect anomalies using Isolation Forest"""
    
    def __init__(self, contamination=0.01):
        from sklearn.ensemble import IsolationForest
        self.contamination = contamination
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        self.is_trained = False
    
    def fit(self, X):
        """Train the detector"""
        print(f"\n🤖 Training Isolation Forest")
        print(f"   Samples: {X.shape[0]:,}")
        print(f"   Features: {X.shape[1]}")
        print(f"   Anomaly contamination: {self.contamination*100:.1f}%")
        
        self.model.fit(X)
        self.is_trained = True
        
        print(f"  ✓ Model trained")
        return self
    
    def predict(self, X):
        """Predict: -1=anomaly, 1=normal"""
        return self.model.predict(X)
    
    def score_samples(self, X):
        """Get anomaly scores (negative = more anomalous)"""
        return self.model.score_samples(X)
    
    def get_summary(self, X):
        """Get summary statistics"""
        scores = self.score_samples(X)
        predictions = self.predict(X)
        
        n_anomalies = (predictions == -1).sum()
        
        summary = {
            'total_samples': int(X.shape[0]),
            'n_anomalies': int(n_anomalies),
            'anomaly_percentage': float(n_anomalies / X.shape[0] * 100),
            'anomaly_score_min': float(scores.min()),
            'anomaly_score_max': float(scores.max()),
            'anomaly_score_mean': float(scores.mean()),
            'anomaly_score_std': float(scores.std()),
            'threshold': float(np.percentile(scores, 1))
        }
        
        print(f"\n📈 Anomaly Detection Summary")
        print(f"   Total samples: {summary['total_samples']:,}")
        print(f"   Anomalies detected: {summary['n_anomalies']} ({summary['anomaly_percentage']:.2f}%)")
        print(f"   Anomaly score range: [{summary['anomaly_score_min']:.4f}, {summary['anomaly_score_max']:.4f}]")
        print(f"   Detection threshold: {summary['threshold']:.4f}")
        
        return summary, scores

# ============================================================================
# PART 4: ANALYSIS & INTERPRETATION
# ============================================================================

class AnomalyAnalyzer:
    """Analyze detected anomalies"""
    
    def __init__(self, df, scores, feature_cols):
        self.df = df.copy()
        self.scores = scores
        self.feature_cols = feature_cols
        
        self.df['anomaly_score'] = scores
        self.df['is_anomaly'] = (scores < np.percentile(scores, 1))
    
    def get_top_anomalies(self, n=10):
        """Get most anomalous samples"""
        top_idx = np.argsort(self.scores)[:n]
        return self.df.iloc[top_idx].copy()
    
    def print_report(self, n=10):
        """Print top anomalies"""
        print(f"\n🔍 Top {n} Most Anomalous Events")
        print("=" * 80)
        
        top = self.get_top_anomalies(n)
        
        for idx, (i, row) in enumerate(top.iterrows(), 1):
            print(f"\n{idx}. Index {i} | Anomaly Score: {row['anomaly_score']:.4f}")
            print(f"   Feature values:")
            for col in self.feature_cols[:3]:  # Show top 3 features
                if col in row.index:
                    print(f"     {col:20} = {row[col]:8.3f}")

# ============================================================================
# PART 5: VISUALIZATION
# ============================================================================

class Visualizer:
    """Create publication-quality visualizations"""
    
    @staticmethod
    def plot_timeline(scores, output_path=None):
        """Plot anomaly timeline"""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(14, 5))
        
        # Plot scores
        ax.plot(scores, color='steelblue', alpha=0.6, linewidth=1)
        ax.fill_between(range(len(scores)), scores, alpha=0.3, color='steelblue')
        
        # Highlight anomalies
        threshold = np.percentile(scores, 1)
        anomaly_mask = scores < threshold
        anomaly_idx = np.where(anomaly_mask)[0]
        
        ax.scatter(anomaly_idx, scores.iloc[anomaly_idx], 
                  color='red', s=100, marker='*', zorder=5, label='Anomalies')
        
        # Add threshold line
        ax.axhline(y=threshold, color='red', linestyle='--', 
                  alpha=0.5, label=f'Threshold (1%)')
        
        # Styling
        ax.set_xlabel('Time (samples)', fontsize=12)
        ax.set_ylabel('Anomaly Score (lower = more anomalous)', fontsize=12)
        ax.set_title('Polymarket: High-Frequency Anomaly Detection Timeline', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"  ✓ Saved: {output_path}")
        
        return fig
    
    @staticmethod
    def plot_distribution(scores, output_path=None):
        """Plot score distribution"""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        ax.hist(scores, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        
        threshold = np.percentile(scores, 1)
        ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2,
                  label=f'1% Threshold ({threshold:.4f})')
        
        # Styling
        ax.set_xlabel('Anomaly Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Anomaly Scores', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"  ✓ Saved: {output_path}")
        
        return fig

# ============================================================================
# PART 6: MAIN PIPELINE
# ============================================================================

def run_pipeline(data_dir, output_dir):
    """Execute complete anomaly detection pipeline"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print(" POLYMARKET ANOMALY DETECTION PIPELINE")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    # Step 1: Load data
    print("\n[1/6] LOADING DATA")
    print("-" * 80)
    loader = DataLoader(data_dir)
    features = loader.load_features()
    
    if features is None:
        print("✗ Failed to load features. Exiting.")
        return
    
    # Step 2: Prepare features
    print("\n[2/6] PREPARING FEATURES")
    print("-" * 80)
    X_scaled, feature_cols = FeaturePreprocessor.preprocess(features)
    
    # Step 3: Train detector
    print("\n[3/6] TRAINING MODEL")
    print("-" * 80)
    detector = IsolationForestDetector(contamination=0.01)
    detector.fit(X_scaled)
    
    # Step 4: Detect anomalies
    print("\n[4/6] DETECTING ANOMALIES")
    print("-" * 80)
    summary, scores = detector.get_summary(X_scaled)
    
    # Step 5: Analyze results
    print("\n[5/6] ANALYZING RESULTS")
    print("-" * 80)
    analyzer = AnomalyAnalyzer(features, scores, feature_cols)
    analyzer.print_report(n=10)
    
    # Step 6: Visualize
    print("\n[6/6] CREATING VISUALIZATIONS")
    print("-" * 80)
    Visualizer.plot_timeline(scores, output_dir / "01_anomaly_timeline.png")
    Visualizer.plot_distribution(scores, output_dir / "02_score_distribution.png")
    
    # Save results
    print(f"\n💾 SAVING RESULTS")
    print("-" * 80)
    
    # Save summary
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ Saved: {output_dir}/summary.json")
    
    # Save anomalies
    anomalies = analyzer.get_top_anomalies(100)
    anomalies.to_csv(output_dir / "top_anomalies.csv", index=False)
    print(f"  ✓ Saved: {output_dir}/top_anomalies.csv")
    
    # Save model info
    model_info = {
        'model_type': 'IsolationForest',
        'contamination': 0.01,
        'n_estimators': 100,
        'features_used': feature_cols,
        'training_samples': X_scaled.shape[0],
        'anomalies_detected': int((scores < np.percentile(scores, 1)).sum())
    }
    
    with open(output_dir / "model_info.json", 'w') as f:
        json.dump(model_info, f, indent=2)
    print(f"  ✓ Saved: {output_dir}/model_info.json")
    
    print("\n" + "="*80)
    print(" ✓ PIPELINE COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - summary.json: Model summary and statistics")
    print(f"  - top_anomalies.csv: Top 100 anomalous events")
    print(f"  - model_info.json: Model configuration")
    print(f"  - 01_anomaly_timeline.png: Timeline visualization")
    print(f"  - 02_score_distribution.png: Score distribution")
    
    return summary, scores

# ============================================================================
# PART 7: MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Polymarket Anomaly Detection Pipeline"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Path to data directory (default: ./data)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Path to output directory (default: ./results)"
    )
    
    args = parser.parse_args()
    
    # Check if data directory exists
    if not Path(args.data_dir).exists():
        print(f"✗ Data directory not found: {args.data_dir}")
        print(f"  Please place your parquet files in: {args.data_dir}")
        sys.exit(1)
    
    # Run pipeline
    try:
        run_pipeline(args.data_dir, args.output_dir)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
