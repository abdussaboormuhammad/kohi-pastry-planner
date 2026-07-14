#!/usr/bin/env python3
"""
goal1_main.py
Kohi Coffee Shop — Goal 1: Pastry Demand Forecasting
MABA Practicum II
"""

import os, re, pickle, warnings, base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                              r2_score, median_absolute_error)
from xgboost import XGBRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor, Pool

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, 'data')
MODELS_DIR  = os.path.join(BASE_DIR, 'goal1_trained_models')
RESULTS_DIR = os.path.join(BASE_DIR, 'goal1_results')
TABLES_DIR  = os.path.join(RESULTS_DIR, 'tables')
IMP_DIR     = os.path.join(RESULTS_DIR, 'importance')
PRED_DIR    = os.path.join(RESULTS_DIR, 'predictions')

for d in [MODELS_DIR, RESULTS_DIR, TABLES_DIR, IMP_DIR, PRED_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
CATEGORIES = [
    'Butter Croissant', 'Chocolate Croissant', 'Cookie/Brownie',
    'Morning Bun', 'Muffin', 'Overnight Oats', 'Savory', 'Sweet Treat',
    'Yogurt Parfait',
]
NUMERIC_FEATURES = [
    'temp_f', 'precip_in', 'humidity_pct', 'wind_mph',
    'Total Occ %', 'Group Occ %', 'Transient Occ %',
]
CAT_FEATURES = ['weather_condition', 'Day of Week', 'Weekday or Weekend', 'Event', 'Month']
TARGET = 'daily_count'


# ── Helpers ───────────────────────────────────────────────────────────────────
def slugify(s):
    return re.sub(r'[^a-zA-Z0-9]', '_', s)

def safe_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true, float), np.array(y_pred, float)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def compute_metrics(y_true, y_pred):
    return {
        'RMSE':     np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE':      mean_absolute_error(y_true, y_pred),
        'R2':       r2_score(y_true, y_pred),
        'MAPE':     safe_mape(y_true, y_pred),
        'MedianAE': median_absolute_error(y_true, y_pred),
    }

def img_to_b64(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode()

def img_tag(path, alt=''):
    if os.path.exists(path):
        return (f'<img src="data:image/png;base64,{img_to_b64(path)}" '
                f'alt="{alt}" style="max-width:100%;border-radius:6px;">')
    return f'<p style="color:gray">[missing: {path}]</p>'


# ── Step 2: Load & dummy-encode ───────────────────────────────────────────────
print('Loading data...')
df = pd.read_csv(os.path.join(DATA_DIR, 'pastry_kohi.csv'))
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
print(f'  {len(df)} rows | {df["Pastry Category"].nunique()} categories | '
      f'{df["Date"].min().date()} → {df["Date"].max().date()}')

print('\nStep 2: Dummy encoding...')
keep = ['Date', 'Pastry Category', TARGET] + NUMERIC_FEATURES + CAT_FEATURES
df_feat  = df[keep].copy()
df_dummy = pd.get_dummies(df_feat, columns=CAT_FEATURES, drop_first=True)
df_dummy.to_csv(os.path.join(DATA_DIR, 'pastry_kohi_dummy.csv'), index=False)

DUMMY_COLS = [c for c in df_dummy.columns if c not in ['Date', 'Pastry Category', TARGET]]
RAW_COLS   = NUMERIC_FEATURES + CAT_FEATURES
print(f'  {len(DUMMY_COLS)} dummy features | saved data/pastry_kohi_dummy.csv')


# ── Model training ────────────────────────────────────────────────────────────
def train_models(X_tr_d, y_tr, X_va_d, y_va, X_tr_r, X_va_r):
    m = {}

    # 1. Linear Regression
    m['LinearRegression'] = LinearRegression().fit(X_tr_d, y_tr)

    # 2. Decision Tree
    m['DecisionTree'] = DecisionTreeRegressor(
        max_depth=5, max_leaf_nodes=32, random_state=42
    ).fit(X_tr_d, y_tr)

    # 3. Random Forest — tune n_estimators on val
    best_rf, best_v = None, np.inf
    for n in [100, 200, 300]:
        rf = RandomForestRegressor(n_estimators=n, random_state=42, n_jobs=-1).fit(X_tr_d, y_tr)
        v  = np.sqrt(mean_squared_error(y_va, rf.predict(X_va_d)))
        if v < best_v:
            best_v, best_rf = v, rf
    m['RandomForest'] = best_rf

    # 4. XGBoost — val early stopping (early_stopping_rounds in constructor, XGBoost 3.x)
    xm = XGBRegressor(
        n_estimators=1000, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8,
        early_stopping_rounds=50,
        random_state=42, n_jobs=-1, verbosity=0,
    )
    xm.fit(X_tr_d, y_tr, eval_set=[(X_va_d, y_va)], verbose=False)
    m['XGBoost'] = xm

    # 5. LightGBM — val early stopping
    lm = LGBMRegressor(
        n_estimators=1000, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    lm.fit(X_tr_d, y_tr, eval_set=[(X_va_d, y_va)],
           callbacks=[lgb.early_stopping(50, first_metric_only=True),
                      lgb.log_evaluation(-1)])
    m['LightGBM'] = lm

    # 6. CatBoost — raw categoricals, val early stopping
    pool_tr = Pool(X_tr_r, y_tr, cat_features=CAT_FEATURES)
    pool_va = Pool(X_va_r, y_va, cat_features=CAT_FEATURES)
    cm = CatBoostRegressor(
        iterations=1000, learning_rate=0.05, depth=5,
        random_seed=42, verbose=0,
    )
    cm.fit(pool_tr, eval_set=pool_va, early_stopping_rounds=50)
    m['CatBoost'] = cm

    return m


def predict(model, name, X_d, X_r):
    if name == 'CatBoost':
        return model.predict(Pool(X_r, cat_features=CAT_FEATURES))
    return model.predict(X_d)


def get_importance(model, name):
    if name == 'LinearRegression':
        return None, None
    if name == 'CatBoost':
        return model.get_feature_importance(), RAW_COLS
    return model.feature_importances_, list(DUMMY_COLS)


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_importance(imp, names, cat, model_name, r2, path):
    n   = min(10, len(names))
    idx = np.argsort(imp)[-n:]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(n), imp[idx], color='steelblue')
    ax.set_yticks(range(n))
    ax.set_yticklabels([names[i] for i in idx], fontsize=8)
    ax.set_xlabel('Importance')
    ax.set_title(f'Feature Importance — {cat} {model_name} (R²={r2:.2f})', fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_pred(y_true, y_pred, wday, cat, model_name, rank, r2, rmse, path):
    colors = ['#2196F3' if w == 'Weekday' else '#FF7043' for w in wday]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, c=colors, alpha=0.65, edgecolors='none', s=40)
    lo = min(min(y_true), min(y_pred)) - 1
    hi = max(max(y_true), max(y_pred)) + 1
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
    ax.set_xlabel('Actual units')
    ax.set_ylabel('Predicted units')
    ax.set_title(f'{cat} — {model_name} (Rank {rank})', fontsize=10)
    ax.annotate(f'R²={r2:.3f}  RMSE={rmse:.2f}',
                xy=(0.05, 0.90), xycoords='axes fraction', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
    patches = [mpatches.Patch(color='#2196F3', label='Weekday'),
               mpatches.Patch(color='#FF7043', label='Weekend')]
    ax.legend(handles=patches, fontsize=8, loc='lower right')
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


# ── Performance table ─────────────────────────────────────────────────────────
def write_table(cat, ranked, path):
    lines = [f'# PASTRY CATEGORY: {cat}\n',
             '| Rank | Model | RMSE | MAE | R² | MAPE | Median AE |',
             '|------|-------|------|-----|----|------|-----------|']
    for i, (name, _, mets, _, _) in enumerate(ranked, 1):
        mape = f"{mets['MAPE']:.1f}%" if not np.isnan(mets['MAPE']) else 'N/A'
        lines.append(
            f"| {i} | {name} | {mets['RMSE']:.2f} | {mets['MAE']:.2f} | "
            f"{mets['R2']:.3f} | {mape} | {mets['MedianAE']:.2f} |"
        )
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


# ── Business insights ─────────────────────────────────────────────────────────
def compute_insights(cat_df, best_model, best_name):
    counts  = cat_df[TARGET]
    cv      = counts.std() / counts.mean() if counts.mean() > 0 else 0
    pattern = 'stable' if cv < 0.3 else ('moderate variability' if cv < 0.6 else 'volatile')

    dow_med  = cat_df.groupby('Day of Week')[TARGET].median()
    peak_day = dow_med.idxmax()

    month_med  = cat_df.groupby('Month')[TARGET].median()
    peak_month = month_med.idxmax()

    ev = cat_df.groupby('Event')[TARGET].mean()
    event_impact = ((ev.get('Yes', np.nan) - ev.get('No', np.nan)) / ev.get('No', np.nan) * 100
                    if 'Yes' in ev.index and 'No' in ev.index else None)

    rain_mask = cat_df['weather_condition'].str.lower().str.contains('rain', na=False)
    if rain_mask.sum() > 0 and (~rain_mask).sum() > 0:
        rain_impact = ((cat_df[rain_mask][TARGET].mean() - cat_df[~rain_mask][TARGET].mean())
                       / cat_df[~rain_mask][TARGET].mean() * 100)
    else:
        rain_impact = None

    imp, feat_names = get_importance(best_model, best_name)
    top3_drivers = []
    if imp is not None:
        idx = np.argsort(imp)[-3:][::-1]
        top3_drivers = [(feat_names[i], float(imp[i])) for i in idx]

    return {
        'baseline':     counts.median(),
        'pattern':      pattern,
        'peak_day':     peak_day,
        'peak_month':   peak_month,
        'event_impact': event_impact,
        'rain_impact':  rain_impact,
        'top3_drivers': top3_drivers,
    }


# ── Main training loop ────────────────────────────────────────────────────────
print('\n' + '='*60)
print('TRAINING MODELS')
print('='*60)

all_ranked = {}

for cat in CATEGORIES:
    print(f'\n{"─"*50}')
    print(f'Category: {cat}')

    cat_raw = (df_feat[df_feat['Pastry Category'] == cat]
               .sort_values('Date').reset_index(drop=True))
    cat_dum = (df_dummy[df_dummy['Pastry Category'] == cat]
               .sort_values('Date').reset_index(drop=True))

    n  = len(cat_raw)
    iv = int(0.70 * n)
    it = int(0.85 * n)
    print(f'  n={n}  train={iv}  val={it-iv}  test={n-it}')

    X_tr_d = cat_dum.iloc[:iv][DUMMY_COLS]
    X_va_d = cat_dum.iloc[iv:it][DUMMY_COLS]
    X_te_d = cat_dum.iloc[it:][DUMMY_COLS]

    X_tr_r = cat_raw.iloc[:iv][RAW_COLS]
    X_va_r = cat_raw.iloc[iv:it][RAW_COLS]
    X_te_r = cat_raw.iloc[it:][RAW_COLS]

    y_tr   = cat_raw.iloc[:iv][TARGET].values
    y_va   = cat_raw.iloc[iv:it][TARGET].values
    y_te   = cat_raw.iloc[it:][TARGET].values
    wday_te = cat_raw.iloc[it:]['Weekday or Weekend'].values

    print('  Training...')
    models = train_models(X_tr_d, y_tr, X_va_d, y_va, X_tr_r, X_va_r)

    results = []
    for mname, model in models.items():
        y_pred = predict(model, mname, X_te_d, X_te_r)
        mets   = compute_metrics(y_te, y_pred)
        results.append((mname, model, mets, y_pred, wday_te))
        mape_s = f"{mets['MAPE']:.1f}%" if not np.isnan(mets['MAPE']) else 'N/A'
        print(f'    {mname:<20} RMSE={mets["RMSE"]:.2f}  MAE={mets["MAE"]:.2f}  '
              f'R²={mets["R2"]:.3f}  MAPE={mape_s}')

    ranked = sorted(results, key=lambda x: (x[2]['RMSE'], x[2]['MAE'], -x[2]['R2']))
    all_ranked[cat] = ranked

    cat_slug = slugify(cat)
    for rank, (mname, model, mets, y_pred, wday) in enumerate(ranked[:3], 1):
        ms = slugify(mname)

        with open(os.path.join(MODELS_DIR, f'{cat_slug}_{ms}_{rank}.pkl'), 'wb') as f:
            pickle.dump(model, f)

        plot_pred(y_te, y_pred, wday, cat, mname, rank, mets['R2'], mets['RMSE'],
                  os.path.join(PRED_DIR, f'{cat_slug}_{ms}_{rank}_pred.png'))

        imp, feat_names = get_importance(model, mname)
        if imp is not None:
            plot_importance(imp, feat_names, cat, mname, mets['R2'],
                            os.path.join(IMP_DIR, f'{cat_slug}_{ms}_{rank}_imp.png'))

    write_table(cat, ranked, os.path.join(TABLES_DIR, f'{cat_slug}_model_table.md'))
    print(f'  Top 3: {[ranked[i][0] for i in range(min(3, len(ranked)))]}')

print('\n' + '='*60)
print('TRAINING COMPLETE — generating report artifacts')
print('='*60)


# ── HTML Report ───────────────────────────────────────────────────────────────
CSS = """
body{font-family:'Segoe UI',Arial,sans-serif;max-width:1200px;margin:0 auto;
     padding:24px;background:#f5f5f5;color:#222;}
h1{color:#3c3c3c;border-bottom:3px solid #c0392b;padding-bottom:10px;}
h2{color:#c0392b;margin-top:40px;border-bottom:1px solid #ddd;padding-bottom:6px;}
h3{color:#555;margin-top:20px;}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:13px;}
th{background:#c0392b;color:white;padding:8px 10px;text-align:left;}
td{border:1px solid #ddd;padding:7px 10px;}
tr:nth-child(even){background:#fafafa;}
tr.top1{background:#fff3cd;font-weight:bold;}
tr.top2{background:#e8f5e9;}
tr.top3{background:#e3f2fd;}
.box{background:white;border-left:4px solid #c0392b;padding:14px 18px;
     margin:14px 0;border-radius:4px;box-shadow:0 1px 4px rgba(0,0,0,.08);}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));
      gap:16px;margin:16px 0;}
.card{background:white;border-radius:8px;padding:12px;
      box-shadow:0 1px 4px rgba(0,0,0,.08);}
.rb{display:inline-block;width:24px;height:24px;border-radius:50%;
    text-align:center;line-height:24px;font-weight:bold;font-size:13px;color:white;}
.r1{background:#c0392b;}.r2{background:#e67e22;}.r3{background:#27ae60;}
footer{margin-top:40px;color:#888;font-size:12px;border-top:1px solid #ddd;padding-top:10px;}
"""

secs = []
secs.append(f"""
<h1>Kohi Coffee Shop — Goal 1: Pastry Demand Forecasting</h1>
<p>MABA Practicum II &nbsp;|&nbsp; Generated {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>
<p>9 pastry categories &nbsp;·&nbsp; 6 models per category &nbsp;·&nbsp;
   Top 3 selected by RMSE (lower better)</p>
<p>Trained on outlier-cleaned data: 38 rows (0.8%) removed via per-category
   MAD modified z-score (|M| &gt; 3.5) on daily_count —
   see <code>goal1_results/outlier_removal_log.md</code></p>
""")

# Summary table
secs.append('<h2>Summary: Best Model Per Category</h2>')
rows = ''
for cat in CATEGORIES:
    bname, _, bm, _, _ = all_ranked[cat][0]
    mape = f"{bm['MAPE']:.1f}%" if not np.isnan(bm['MAPE']) else 'N/A'
    rows += (f"<tr><td>{cat}</td><td><strong>{bname}</strong></td>"
             f"<td>{bm['RMSE']:.2f}</td><td>{bm['MAE']:.2f}</td>"
             f"<td>{bm['R2']:.3f}</td><td>{mape}</td><td>{bm['MedianAE']:.2f}</td></tr>")
secs.append(f"""<table>
<tr><th>Category</th><th>Best Model</th><th>RMSE</th><th>MAE</th>
<th>R²</th><th>MAPE</th><th>Median AE</th></tr>
{rows}</table>""")

# Per-category sections
for cat in CATEGORIES:
    ranked   = all_ranked[cat]
    cat_slug = slugify(cat)
    cat_df   = df_feat[df_feat['Pastry Category'] == cat]
    best_name, best_model = ranked[0][0], ranked[0][1]
    ins = compute_insights(cat_df, best_model, best_name)

    secs.append(f'<h2>{cat}</h2>')

    # Rankings table
    trows = ''
    for i, (mname, _, mets, _, _) in enumerate(ranked, 1):
        mape = f"{mets['MAPE']:.1f}%" if not np.isnan(mets['MAPE']) else 'N/A'
        rc   = f'top{i}' if i <= 3 else ''
        badge = f'<span class="rb r{i}">{i}</span>' if i <= 3 else str(i)
        trows += (f'<tr class="{rc}"><td>{badge}</td><td>{mname}</td>'
                  f'<td>{mets["RMSE"]:.2f}</td><td>{mets["MAE"]:.2f}</td>'
                  f'<td>{mets["R2"]:.3f}</td><td>{mape}</td>'
                  f'<td>{mets["MedianAE"]:.2f}</td></tr>')
    secs.append(f"""<h3>Model Rankings (all 6)</h3>
<table><tr><th>Rank</th><th>Model</th><th>RMSE</th><th>MAE</th>
<th>R²</th><th>MAPE</th><th>Median AE</th></tr>{trows}</table>""")

    # Business insights
    ev_str   = f'{ins["event_impact"]:+.1f}%' if ins['event_impact'] is not None else 'N/A'
    rain_str = f'{ins["rain_impact"]:+.1f}%'  if ins['rain_impact']  is not None else 'N/A'
    drivers_html = ''
    if ins['top3_drivers']:
        items = ''.join(f'<li>{n} (importance: {v:.3f})</li>'
                        for n, v in ins['top3_drivers'])
        drivers_html = (f'<li><strong>Top 3 demand drivers ({best_name}):</strong>'
                        f'<ol>{items}</ol></li>')

    rec_parts = [f"Increase production on {ins['peak_day']}s"]
    if ins['event_impact'] and ins['event_impact'] > 5:
        rec_parts.append(f"and on event days ({ev_str})")
    if ins['rain_impact'] and ins['rain_impact'] < -5:
        rec_parts.append(f"reduce on rainy days ({rain_str})")
    recommendation = ' '.join(rec_parts) + '.'

    secs.append(f"""<h3>Business Insights</h3>
<div class="box"><ul>
<li><strong>Baseline:</strong> ~{ins['baseline']:.0f} units/day (median)</li>
<li><strong>Demand pattern:</strong> {ins['pattern']}</li>
<li><strong>Peak day of week:</strong> {ins['peak_day']}</li>
<li><strong>Peak month:</strong> {ins['peak_month']}</li>
<li><strong>Event impact:</strong> {ev_str} vs non-event days</li>
<li><strong>Rain impact:</strong> {rain_str} vs clear days</li>
{drivers_html}
<li><strong>Recommendation:</strong> {recommendation}</li>
</ul></div>""")

    # Prediction plots
    secs.append('<h3>Prediction vs Actual — Top 3 Models</h3><div class="grid">')
    for rank in range(1, 4):
        mname = ranked[rank-1][0]
        ms    = slugify(mname)
        p     = os.path.join(PRED_DIR, f'{cat_slug}_{ms}_{rank}_pred.png')
        secs.append(f'<div class="card">{img_tag(p, f"{cat} {mname}")}</div>')
    secs.append('</div>')

    # Importance plots
    imp_cards = []
    for rank in range(1, 4):
        mname = ranked[rank-1][0]
        ms    = slugify(mname)
        p     = os.path.join(IMP_DIR, f'{cat_slug}_{ms}_{rank}_imp.png')
        if os.path.exists(p):
            imp_cards.append(f'<div class="card">{img_tag(p, f"{cat} {mname} importance")}</div>')
    if imp_cards:
        secs.append('<h3>Feature Importance — Top 3 Models</h3><div class="grid">')
        secs.extend(imp_cards)
        secs.append('</div>')

html_out = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Kohi — Goal 1 Results</title>
<style>{CSS}</style>
</head>
<body>
{''.join(secs)}
<footer>MABA Practicum II — Goal 1: Pastry Demand Forecasting | Kohi Coffee Shop</footer>
</body>
</html>"""

with open(os.path.join(BASE_DIR, 'goal1_results_report.html'), 'w') as f:
    f.write(html_out)
print('Saved goal1_results_report.html')


# ── model_explain.md ──────────────────────────────────────────────────────────
MODEL_EXPLAIN = """\
# Model Selection & Rationale — Goal 1: Kohi Pastry Demand Forecasting

## Architecture: 9 Individual Models

Each pastry category gets its own trained model set. A Morning Bun responds
differently to hotel occupancy than an Overnight Oat; combining them into one
multi-output model loses that nuance.

## Outlier Removal (applied before this training run)

`data/pastry_kohi.csv` was cleaned with a per-category **MAD modified
z-score** filter on `daily_count` (Iglewicz & Hoaglin 1993): rows with
|M| = |0.6745 · (x − median) / MAD| > 3.5 were removed. 38 of 4,483 rows
(0.8%) were dropped — one-off demand spikes (e.g. Butter Croissant days of
16 and 25 units against a median of 5, Savory days of 9–14 against a median
of 3) most plausibly caused by catering orders or data-entry anomalies that a
weather + occupancy model cannot and should not learn.

MAD was chosen over the more common 1.5×IQR fence because several categories
have discrete counts with IQR = 1 (e.g. Morning Bun, median 2), where Tukey
fences misclassify ordinary counts one unit above Q3 as outliers. Numeric
predictors (weather, occupancy) were left untouched — extreme weather days
are real conditions the models must learn from. The exact per-category
bounds and removed values are logged in
`goal1_results/outlier_removal_log.md`; the pre-cleaning dataset is
preserved as `data/pastry_kohi_raw.csv`.

## Data Split

Time-ordered 70/15/15 train / validation / test split per category:
- **Training (70%)** — model fitting
- **Validation (15%)** — hyperparameter tuning and early stopping only
- **Test (15%)** — final unbiased evaluation; never touched during training

No shuffling — preserves temporal order, prevents future-predicts-past leakage.

## Models

### 1. Multiple Linear Regression
- Uses dummy-encoded features (`pastry_kohi_dummy.csv`, `drop_first=True`)
- Interpretable baseline; assumes linear additive relationships
- Cannot capture weather × weekend interactions

### 2. Decision Tree
- `max_depth=5`, `max_leaf_nodes=32`
- Non-linear rules; depth-constrained to reduce variance on small training sets

### 3. Random Forest
- Ensemble of decision trees; n_estimators ∈ {100, 200, 300} selected by val RMSE
- Reduces DT variance through bagging; strong on small tabular datasets

### 4. XGBoost
- `learning_rate=0.05`, `max_depth=5`, early stopping at 50 rounds (val set)
- Handles feature interactions and non-linear weather-demand relationships

### 5. LightGBM
- Leaf-wise gradient boosting; same hyperparameter profile as XGBoost
- Faster training; early stopping on val set

### 6. CatBoost
- Receives the **original (non-encoded)** dataset with raw categorical columns
- Ordered boosting reduces within-fold leakage
- Robust to high-cardinality categoricals (`Month`, `weather_condition`)

## Why No Neural Networks?

~415 training rows per category is too few for deep learning without heavy
regularization. Tree ensembles are the standard choice for this data regime.
If the dataset grows in future phases, LSTM or Temporal Fusion Transformer
should be reconsidered.
"""

with open(os.path.join(BASE_DIR, 'model_explain.md'), 'w') as f:
    f.write(MODEL_EXPLAIN)
print('Saved model_explain.md')


# ── metrics_explain.md ────────────────────────────────────────────────────────
METRICS_EXPLAIN = """\
# Metrics Explanation — Goal 1: Kohi Pastry Demand Forecasting

## RMSE — Root Mean Squared Error
**Formula**: √mean((predicted − actual)²)

Primary ranking metric. Errors are squared before averaging, so large misses
(e.g., under-baking by 15 units on a busy event day) are penalised far more
than small ones. This aligns with the business cost of stock-outs.

**Guide**: An RMSE of 3.5 means predictions are off by roughly 3.5 units,
with larger errors weighted more heavily. Lower is better.

---

## MAE — Mean Absolute Error
**Formula**: mean(|predicted − actual|)

Every error is weighted equally — no squaring. An MAE of 2.0 means the model
is typically within 2 units of the actual count. Easier to communicate to the
pastry chef than RMSE. Used as first tiebreaker when RMSE values are equal.

---

## R² — Coefficient of Determination
**Formula**: 1 − SS_res / SS_tot

Proportion of variance in daily_count explained by the model.

| Range | Interpretation |
|-------|----------------|
| ≥ 0.70 | Strong — recommended for production use |
| 0.30–0.69 | Acceptable — useful for directional guidance |
| < 0.30 | Weak — little better than predicting the daily average |
| < 0 | Model's test error exceeds that of predicting the test-set mean |

**Note on negative values**: out-of-sample R² is *not* bounded at 0. The
metric compares the model's squared error against a baseline that predicts
the *test set's own mean* — a baseline the model never sees. On small
(~60–90 row), noisy test sets, a model can easily do worse than that
baseline, producing a negative R² even when the formula and pipeline are
correct.

---

## MAPE — Mean Absolute Percentage Error
**Formula**: mean(|predicted − actual| / actual) × 100

Volume-independent: a 3-unit miss on an item selling 10/day (30%) is correctly
flagged as worse than the same miss on an item selling 50/day (6%).
Enables fair cross-category comparison.

**Caveat**: Undefined when actual = 0. Days with zero sales are excluded;
result shown as N/A if no non-zero test days exist.

---

## Median AE — Median Absolute Error
**Formula**: median(|predicted − actual|)

Half of predictions fall within this distance of the actual count.
Robust to outlier event-day spikes that inflate MAE and RMSE.

---

## Ranking Priority
1. **RMSE ↑** — minimise large errors (primary)
2. **MAE ↑** — minimise typical error magnitude (tiebreaker)
3. **R² ↓** — prefer higher explanatory power (secondary tiebreaker)
"""

with open(os.path.join(BASE_DIR, 'metrics_explain.md'), 'w') as f:
    f.write(METRICS_EXPLAIN)
print('Saved metrics_explain.md')


# ── Done ──────────────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('GOAL 1 COMPLETE')
print('='*60)
print('  goal1_trained_models/   — 27 .pkl files')
print('  goal1_results/tables/   — 9 performance tables (.md)')
print('  goal1_results/importance/ — feature importance plots (.png)')
print('  goal1_results/predictions/ — prediction scatter plots (.png)')
print('  goal1_results_report.html — comprehensive HTML report')
print('  data/pastry_kohi_dummy.csv — dummy-encoded features')
print('  model_explain.md')
print('  metrics_explain.md')
