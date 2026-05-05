import pandas as pd
import numpy as np
import json

df = pd.read_parquet('data/ml_features_1m_v2.parquet')
print('=== DATASET OVERVIEW ===')
print(f'Total rows: {df.shape[0]:,}')
print(f'Total columns: {df.shape[1]}')
print(f'Unique markets: {df["market_id"].nunique()}')
print(f'Date range: {df["minute_bar"].min()} to {df["minute_bar"].max()}')
print(f'Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB')
print()

print('=== COLUMN DTYPES ===')
for col in df.columns:
    print(f'  {col:30s} {str(df[col].dtype):20s}')
print()

print('=== COLUMN STATISTICS ===')
num_cols = ['close_mid','mean_spread','close_spread','bar_volatility','total_volume',
            'buy_volume','sell_volume','trade_count','order_flow_imbalance',
            'return_1m','bid_depth','ask_depth','depth_imbalance']
stats = df[num_cols].describe().round(4)
print(stats.to_string())
print()

print('=== MISSING VALUES ===')
print(df.isnull().sum().to_string())
print()

print('=== TARGET DISTRIBUTION ===')
print(df['target'].value_counts().to_string())
print(f'Positive rate: {df["target"].mean():.4f}')
print()

anom = pd.read_csv('results/top_anomalies.csv')
print('=== TOP 100 ANOMALIES: MARKET DISTRIBUTION ===')
print(anom['market_id'].value_counts().head(10).to_string())
print()

print('=== TOP 100 ANOMALIES: TIME DISTRIBUTION ===')
anom['minute_bar'] = pd.to_datetime(anom['minute_bar'])
print(anom['minute_bar'].dt.date.value_counts().sort_index().to_string())
print()

print('=== FEATURE CORRELATIONS (full dataset) ===')
feat_cols = ['mean_spread','bar_volatility','order_flow_imbalance',
             'bid_depth','ask_depth','depth_imbalance','trade_count']
corr = df[feat_cols].corr().round(3)
print(corr.to_string())
print()

print('=== ANOMALOUS vs NORMAL: MEAN COMPARISON ===')
all_anom = pd.read_csv('results/top_anomalies.csv')
for col in feat_cols:
    anom_mean = all_anom[col].mean()
    norm_mean = df[col].mean()
    print(f'{col:30s}  Normal={norm_mean:12.4f}  Anomalous={anom_mean:12.4f}  Ratio={anom_mean/(norm_mean+1e-10):.2f}x')
