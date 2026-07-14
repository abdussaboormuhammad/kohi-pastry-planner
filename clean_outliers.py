#!/usr/bin/env python3
"""
clean_outliers.py
Kohi Coffee Shop — Goal 1 data cleaning: remove statistical outliers from
data/pastry_kohi.csv and rebuild data/pastry_kohi_dummy.csv.

Method: per pastry category, remove rows whose daily_count has a
MAD-based modified z-score above 3.5 (Iglewicz & Hoaglin, 1993):

    M_i = 0.6745 * (x_i - median(x)) / MAD(x)

Chosen over Tukey 1.5*IQR because these are small discrete count
distributions (several categories have IQR = 1), where 1.5*IQR fences
misclassify ordinary counts one unit above Q3 as outliers. Numeric
predictors (weather, occupancy) are left untouched — extreme weather is a
real condition the models must learn, and those columns are shared across
all 9 categories.

The original file is preserved as data/pastry_kohi_raw.csv.
Full removal log is printed and saved to goal1_results/outlier_removal_log.md.
"""

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'goal1_results')
os.makedirs(RESULTS_DIR, exist_ok=True)

CAT_FEATURES = ['weather_condition', 'Day of Week', 'Weekday or Weekend', 'Event', 'Month']
NUMERIC_FEATURES = [
    'temp_f', 'precip_in', 'humidity_pct', 'wind_mph',
    'Total Occ %', 'Group Occ %', 'Transient Occ %',
]
TARGET = 'daily_count'
THRESHOLD = 3.5

src = os.path.join(DATA_DIR, 'pastry_kohi.csv')
raw_backup = os.path.join(DATA_DIR, 'pastry_kohi_raw.csv')

# Idempotent: always clean from the raw backup if one exists
if os.path.exists(raw_backup):
    df = pd.read_csv(raw_backup)
else:
    df = pd.read_csv(src)
    df.to_csv(raw_backup, index=False)
    print(f'Backed up original → data/pastry_kohi_raw.csv ({len(df)} rows)')

keep_mask = pd.Series(True, index=df.index)
log_rows = []
for cat, g in df.groupby('Pastry Category'):
    x = g[TARGET].astype(float)
    med = x.median()
    mad = (x - med).abs().median()
    if mad == 0:
        continue
    mz = 0.6745 * (x - med) / mad
    out_idx = g.index[mz.abs() > THRESHOLD]
    keep_mask.loc[out_idx] = False
    removed = df.loc[out_idx]
    log_rows.append({
        'Category': cat,
        'Rows before': len(g),
        'Removed': len(out_idx),
        'Removed %': f'{100 * len(out_idx) / len(g):.1f}%',
        'Median': med,
        'MAD': mad,
        'Removed values': ', '.join(str(v) for v in sorted(removed[TARGET].unique())) or '—',
    })

cleaned = df[keep_mask].reset_index(drop=True)
cleaned.to_csv(src, index=False)
print(f'pastry_kohi.csv: {len(df)} → {len(cleaned)} rows ({len(df) - len(cleaned)} removed)')

# Rebuild the dummy-encoded counterpart (same 29-feature schema as goal1_main.py)
keep_cols = ['Date', 'Pastry Category', TARGET] + NUMERIC_FEATURES + CAT_FEATURES
df_feat = cleaned[keep_cols].copy()
df_dummy = pd.get_dummies(df_feat, columns=CAT_FEATURES, drop_first=True)
df_dummy.to_csv(os.path.join(DATA_DIR, 'pastry_kohi_dummy.csv'), index=False)
dummy_cols = [c for c in df_dummy.columns if c not in ['Date', 'Pastry Category', TARGET]]
print(f'pastry_kohi_dummy.csv rebuilt: {len(dummy_cols)} feature columns')
assert len(dummy_cols) == 29, f'Expected 29 dummy features, got {len(dummy_cols)}'

# Paper trail
log = pd.DataFrame(log_rows)
header = '| ' + ' | '.join(log.columns) + ' |'
sep = '|' + '|'.join('---' for _ in log.columns) + '|'
body = ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in log.itertuples(index=False)]
log_md = [
    '# Outlier Removal Log — Goal 1 Rebuild',
    '',
    f'Method: per-category MAD modified z-score on `{TARGET}`, threshold |M| > {THRESHOLD}',
    '(Iglewicz & Hoaglin 1993). Numeric predictors untouched.',
    '',
    header, sep, *body,
    '',
    f'**Total: {len(df) - len(cleaned)} of {len(df)} rows removed '
    f'({100 * (len(df) - len(cleaned)) / len(df):.1f}%). '
    f'Original preserved as `data/pastry_kohi_raw.csv`.**',
]
with open(os.path.join(RESULTS_DIR, 'outlier_removal_log.md'), 'w') as f:
    f.write('\n'.join(log_md) + '\n')
print('\n' + log.to_string(index=False))
print('\nSaved goal1_results/outlier_removal_log.md')
