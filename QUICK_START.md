# QUICK START GUIDE - Polymarket Anomaly Detection
## Get from 0 → Portfolio Project in 4 Hours

---

## 📋 PREREQUISITES (5 minutes)

### What You Need
- Python 3.8+ installed
- Your Polymarket data files
- ~2GB disk space
- Terminal/Command prompt access

### Check Python Installation
```bash
python --version  # Should be 3.8 or higher
pip --version     # Should be installed
```

---

## 🚀 STEP 1: Environment Setup (10 minutes)

### Option A: Using Virtual Environment (Recommended)
```bash
# Navigate to your project folder
cd /path/to/polymarket-project

# Create virtual environment
python -m venv polymarket_env

# Activate it
# On macOS/Linux:
source polymarket_env/bin/activate

# On Windows:
polymarket_env\Scripts\activate
```

### Option B: Using Conda (if you prefer)
```bash
conda create -n polymarket python=3.10
conda activate polymarket
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

**Expected output:**
```
Successfully installed pandas-2.0.3 numpy-1.24.3 scikit-learn-1.3.0 ...
```

---

## 📁 STEP 2: Organize Your Data (5 minutes)

### Create Directory Structure
```bash
mkdir -p polymarket-project/{data/{features,labels},results,notebooks,src}
```

### Expected Directory Tree
```
polymarket-project/
├── data/
│   ├── features/
│   │   └── ml_features_1m_v2.parquet        ← Your features file
│   └── labels/
│       └── trades.parquet                    ← (Optional) Price labels
├── results/                                  ← Where output will be saved
├── src/                                      ← Source code
├── polymarket_complete_pipeline.py           ← Main script
├── requirements.txt
└── QUICK_START.md
```

### Copy Your Data Files
```bash
# Copy your parquet files to the data folder
cp /path/to/ml_features_1m_v2.parquet data/features/
cp /path/to/trades.parquet data/labels/
```

**Verify files are there:**
```bash
ls -lh data/features/
ls -lh data/labels/
```

---

## 🎯 STEP 3: Run the Complete Pipeline (30 minutes)

### The Easy Way (Recommended)
Run the all-in-one script:

```bash
python polymarket_complete_pipeline.py \
    --data-dir ./data \
    --output-dir ./results
```

**What it does:**
1. ✅ Loads your feature data
2. ✅ Auto-detects market stress indicators (spread, depth, OFI, volatility)
3. ✅ Normalizes features using RobustScaler
4. ✅ Trains Isolation Forest model
5. ✅ Detects anomalies (top 1% most unusual)
6. ✅ Generates visualizations
7. ✅ Saves results

**Expected output:**
```
================================================================================
 POLYMARKET ANOMALY DETECTION PIPELINE
================================================================================
Timestamp: 2024-XX-XX XX:XX:XX
Output directory: ./results
================================================================================

[1/6] LOADING DATA
------------------------------------------------------------------------
✓ Loaded features: 100,000 rows, 15 columns

[2/6] PREPARING FEATURES
------------------------------------------------------------------------
📊 Auto-detecting features:
  ✓ spread         → bid_ask_spread
  ✓ depth          → total_depth
  ✓ ofi            → order_flow_imbalance
  ✓ volatility     → price_volatility
  ✓ momentum       → price_momentum

🔧 Preprocessing 5 features
  ✓ Normalized 5 features
    Mean: {...}

[3/6] TRAINING MODEL
------------------------------------------------------------------------
🤖 Training Isolation Forest
   Samples: 100,000
   Features: 5
   Anomaly contamination: 1.0%
  ✓ Model trained

[4/6] DETECTING ANOMALIES
------------------------------------------------------------------------
📈 Anomaly Detection Summary
   Total samples: 100,000
   Anomalies detected: 1,000 (1.00%)
   Anomaly score range: [-0.5234, -0.0001]
   Detection threshold: -0.0456

[5/6] ANALYZING RESULTS
------------------------------------------------------------------------
🔍 Top 10 Most Anomalous Events
================================================================================

1. Index 45234 | Anomaly Score: -0.5234
   Feature values:
     bid_ask_spread           =   3.124
     total_depth              =  -2.456
     order_flow_imbalance     =   4.789

...

[6/6] CREATING VISUALIZATIONS
------------------------------------------------------------------------
✓ Saved: ./results/01_anomaly_timeline.png
✓ Saved: ./results/02_score_distribution.png

💾 SAVING RESULTS
------------------------------------------------------------------------
✓ Saved: ./results/summary.json
✓ Saved: ./results/top_anomalies.csv
✓ Saved: ./results/model_info.json

================================================================================
 ✓ PIPELINE COMPLETE
================================================================================
```

### What Gets Created
After running, check your `results/` folder:

```
results/
├── summary.json                    # Key metrics in JSON
├── top_anomalies.csv              # Top 100 anomalous events with features
├── model_info.json                # Model configuration
├── 01_anomaly_timeline.png        # Visualization 1
└── 02_score_distribution.png      # Visualization 2
```

---

## 🔍 STEP 4: Explore Results (10 minutes)

### View Summary Statistics
```bash
# On macOS/Linux:
cat results/summary.json

# On Windows:
type results\summary.json

# Or in Python:
import json
with open('results/summary.json') as f:
    print(json.dumps(json.load(f), indent=2))
```

**You'll see something like:**
```json
{
  "total_samples": 100000,
  "n_anomalies": 1000,
  "anomaly_percentage": 1.0,
  "anomaly_score_min": -0.5234,
  "anomaly_score_max": -0.0001,
  "anomaly_score_mean": -0.0234,
  "anomaly_score_std": 0.0456,
  "threshold": -0.0456
}
```

### View Top Anomalies
```bash
# View in terminal (first 20 lines)
head -20 results/top_anomalies.csv

# Or open in Excel/Google Sheets
# The CSV has columns: timestamp, feature1, feature2, anomaly_score, etc.
```

### View Visualizations
```bash
# Open images with your default viewer
# macOS:
open results/01_anomaly_timeline.png
open results/02_score_distribution.png

# Windows:
start results/01_anomaly_timeline.png
start results/02_score_distribution.png

# Linux:
xdg-open results/01_anomaly_timeline.png
```

---

## 📊 STEP 5: Build Interactive Dashboard (Optional, 20 minutes)

Create `app.py`:

```python
import dash
from dash import dcc, html
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json

# Load results
with open('results/summary.json') as f:
    summary = json.load(f)

anomalies_df = pd.read_csv('results/top_anomalies.csv')

# Create app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.H1("🎯 Polymarket Anomaly Detection Dashboard"),
        html.P("Real-time detection of unusual orderbook activity")
    ], style={'padding': '20px'}),
    
    # Summary cards
    html.Div([
        html.Div([
            html.H3(f"{summary['total_samples']:,}"),
            html.P("Total Samples")
        ], style={'flex': '1', 'padding': '20px', 'background': '#f0f0f0'}),
        
        html.Div([
            html.H3(f"{summary['n_anomalies']}"),
            html.P("Anomalies Detected")
        ], style={'flex': '1', 'padding': '20px', 'background': '#fff0f0'}),
        
        html.Div([
            html.H3(f"{summary['anomaly_percentage']:.2f}%"),
            html.P("Detection Rate")
        ], style={'flex': '1', 'padding': '20px', 'background': '#f0fff0'}),
    ], style={'display': 'flex', 'gap': '20px', 'padding': '20px'}),
    
    # Charts
    html.Div([
        dcc.Markdown(f"""
        ## Results Summary
        
        - **Total Samples**: {summary['total_samples']:,}
        - **Anomalies**: {summary['n_anomalies']} samples ({summary['anomaly_percentage']:.2f}%)
        - **Score Range**: [{summary['anomaly_score_min']:.4f}, {summary['anomaly_score_max']:.4f}]
        - **Threshold**: {summary['threshold']:.4f}
        
        ## Top 10 Anomalies
        """)
    ], style={'padding': '20px'}),
    
    # Anomalies table
    html.Div([
        dcc.Markdown(anomalies_df.head(10).to_markdown())
    ], style={'padding': '20px'})
])

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
```

Run it:
```bash
python app.py
```

Visit: **http://localhost:8050** in your browser

---

## 🎓 STEP 6: Understanding the Results

### What's an Anomaly?
The model identifies samples where the combination of features is statistically unusual:

**Normal trading:** Tight spread, good liquidity, balanced order flow
**Anomaly:** Wide spread + low depth + extreme order imbalance = **Market Stress**

### Anomaly Score Interpretation
- **Higher (closer to 0)**: More normal
- **Lower (more negative)**: More anomalous

Example:
```
Anomaly Score: -0.45  ← Very unusual (red flag 🚩)
Anomaly Score: -0.02  ← Slightly unusual
Anomaly Score: +0.01  ← Normal
```

### Why Use Isolation Forest?
✅ Unsupervised (no labeled data needed)
✅ Fast (works with 100K+ samples)
✅ Robust to high-dimensional data
✅ Interpretable (can trace back which features triggered it)

---

## 🐛 TROUBLESHOOTING

### Problem: "Cannot find required column"
```
Error: KeyError: 'bid_ask_spread'
```
**Solution:** Check your actual column names
```python
import pandas as pd
df = pd.read_parquet('data/features/ml_features_1m_v2.parquet')
print(df.columns)  # See what columns you have
```

### Problem: "Out of Memory"
```
Error: MemoryError
```
**Solution:** Use a subset of your data
```bash
# Edit polymarket_complete_pipeline.py, after loading:
features = features.iloc[:50000]  # Use first 50K rows
```

### Problem: "Import Error: No module named 'polars'"
```
ModuleNotFoundError: No module named 'polars'
```
**Solution:** Install missing package
```bash
pip install polars pyarrow
```

### Problem: Dashboard won't start on port 8050
```bash
# Try a different port
python app.py --port 8080
# Or change in app.py:
app.run_server(port=8080)
```

---

## 📈 NEXT STEPS (Make it More Impressive)

### Tier 1: Quick Wins (30 min)
- [ ] Add SHAP feature importance
- [ ] Calculate price movement correlation
- [ ] Create PDF report

```python
# Simple correlation check
from scipy.stats import spearmanr

price_change = df['price'].diff()
corr, p_value = spearmanr(anomaly_scores, price_change)
print(f"Anomaly-Price Correlation: {corr:.3f} (p={p_value:.2e})")
```

### Tier 2: Professional (1-2 hours)
- [ ] Add real-time streaming simulation
- [ ] Create interactive feature heatmap
- [ ] Build confidence intervals
- [ ] Add anomaly clustering

### Tier 3: Production (4-8 hours)
- [ ] REST API with Flask/FastAPI
- [ ] Model serving with TensorFlow Serving
- [ ] Real-time streaming (Kafka/WebSocket)
- [ ] Monitoring dashboard
- [ ] Unit tests

---

## 📝 INTERVIEW TALKING POINTS

When you present this project, be ready to explain:

1. **Why Isolation Forest?**
   - "I chose Isolation Forest because it's unsupervised—I didn't need to manually label anomalies. It's also fast and handles high-dimensional data well."

2. **Feature Selection**
   - "I focused on market microstructure features: bid-ask spread (liquidity), order book depth, order flow imbalance, and volatility. These capture market 'stress'."

3. **Validation**
   - "I used a 1% contamination rate based on domain knowledge (rare events). Anomalies correlate with actual price movements."

4. **Practical Application**
   - "This could detect manipulation attempts, flash crashes, and liquidity crises—useful for risk management."

5. **Technical Depth**
   - "I used RobustScaler instead of StandardScaler to handle outliers better. I also validated on time-series data without look-ahead bias."

---

## 📚 USEFUL REFERENCES

### Understanding the Model
- Isolation Forest: https://en.wikipedia.org/wiki/Isolation_forest
- Scikit-learn Docs: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html

### Market Microstructure
- What is Bid-Ask Spread? https://www.investopedia.com/terms/b/bid-askspread.asp
- Order Flow Imbalance: https://en.wikipedia.org/wiki/Order_imbalance

### Tools
- Pandas: https://pandas.pydata.org/docs/
- Plotly: https://plotly.com/python/
- Dash: https://dash.plotly.com/

---

## 🎉 SUCCESS CHECKLIST

- [ ] Python environment set up
- [ ] Data files placed in `data/` folder
- [ ] `polymarket_complete_pipeline.py` runs without errors
- [ ] Results appear in `results/` folder
- [ ] Visualizations look reasonable
- [ ] Can explain why Isolation Forest
- [ ] Can explain what anomalies mean
- [ ] Optional: Dashboard running on localhost:8050

**If all boxes are checked, you have a portfolio-ready project! 🚀**

---

## ❓ FREQUENTLY ASKED QUESTIONS

**Q: How long should this take?**
A: 3-4 hours from scratch. The complete pipeline runs in ~5-10 minutes depending on data size.

**Q: What if my features are different?**
A: The script auto-detects features. It looks for common names (spread, depth, ofi, volatility, momentum). If yours have different names, edit the `feature_cols` list.

**Q: Can I use this on new data?**
A: Yes! Just dump new parquet files in `data/features/` and run the pipeline again.

**Q: What's the anomaly detection accuracy?**
A: Accuracy isn't the right metric for unsupervised learning. Check if anomalies correlate with real price movements instead.

**Q: How do I improve the results?**
A: Try different contamination rates (0.005, 0.02), add more features, or manually label some anomalies to train a supervised classifier.

---

## 💬 GETTING HELP

If something doesn't work:

1. Check error message carefully
2. Google the error (usually helpful!)
3. Check troubleshooting section above
4. Print intermediate results:
   ```python
   print(features.shape)
   print(features.columns)
   print(X_scaled.describe())
   ```

---

## 📞 NEXT: GitHub Portfolio

Once working, create a GitHub repo:

```bash
git init
git add .
git commit -m "Initial commit: Polymarket anomaly detection"
git remote add origin https://github.com/YOUR_USERNAME/polymarket-anomaly-detection
git push -u origin main
```

**Add to README:**
- Problem statement
- Data source
- Model architecture
- Results
- How to run
- Future improvements

**Boom! Portfolio project done.** 🎉

---

**Good luck! You've got this! 💪**
