# Goal

Build a Streamlit web app for the Kohi Coffee Shop head pastry chef: the chef
enters next-day hotel occupancy and an event flag (Yes/No), the app auto-pulls
tomorrow's weather for Bentonville AR, then outputs predicted unit counts for
all 9 pastry categories using the trained ML models from Goal 1.

---

# Current State

Goal 1 is **fully complete**. All 9 categories trained, evaluated, and
exported. The Streamlit app (`streamlit_app.py`) does not exist yet — that is
the entire remaining scope.

The two scripts that produced Goal 1 artifacts are:
- `goal1_main.py` — trained the original 8 categories (runs clean)
- `add_sweet_treat.py` — added Sweet Treat retroactively (runs clean, one-time)

Do not re-run either script unless you need to regenerate artifacts from scratch.

---

# Files in Flight

| File | Status |
|------|--------|
| `goal1_main.py` | Complete, production-ready |
| `add_sweet_treat.py` | Complete, one-time use only |
| `goal1_results_report.html` | Complete, self-contained (base64 images embedded) |
| `goal1_trained_models/` | 27 .pkl files, complete |
| `data/pastry_kohi.csv` | 4,483 rows, 9 categories, complete |
| `data/pastry_kohi_dummy.csv` | 29 dummy features, rebuilt, complete |
| `goal1_results/tables/` | 9 .md performance tables, complete |
| `goal1_results/predictions/` | 27 scatter plots, complete |
| `goal1_results/importance/` | 23 importance plots (LR has none by design), complete |
| `model_explain.md` | Complete |
| `metrics_explain.md` | Complete |
| `prompt.md` | Updated to 9 categories throughout |
| `streamlit_app.py` | **Does not exist — this is the next deliverable** |

---

# What Changed This Session

1. Rewrote `prompt.md`: removed stale meta-instruction, fixed typo ("6 mdoels"),
   extracted dummy encoding into its own Step 2, added explicit validation-set
   usage rules for early stopping models.
2. Wrote and ran `goal1_main.py` end-to-end: 6 models × 8 categories = 48
   trained, top 3 per category saved as .pkl, all plots and tables generated.
3. Fixed XGBoost 3.x breaking change (see Failed Attempts).
4. Discovered Sweet Treat has 0 rows in `pastry_kohi.csv` — it was excluded
   during Phase 1 cleaning. Recovered it from `hour_kohi.csv`.
5. Wrote and ran `add_sweet_treat.py`: aggregated 504 Sweet Treat daily rows
   from hourly data, appended to `pastry_kohi.csv`, rebuilt dummy CSV, trained
   6 models for Sweet Treat, appended Sweet Treat section to HTML report.
6. Updated `prompt.md` counts throughout (8→9 categories, 48→54 models trained,
   24→27 final models, 3979→4483 rows).

---

# Failed Attempts

**XGBoost 3.x: `early_stopping_rounds` removed from `fit()`**
- Symptom: `TypeError: XGBModel.fit() got an unexpected keyword argument 'early_stopping_rounds'`
- Fix: pass `early_stopping_rounds=50` to the **constructor** (`XGBRegressor(..., early_stopping_rounds=50)`), not to `.fit()`.

**LightGBM `early_stopping(verbose=False)` invalid**
- `verbose` is not a parameter of `lgb.early_stopping()`.
- Fix: use `lgb.early_stopping(50, first_metric_only=True)` and suppress output separately with `lgb.log_evaluation(-1)`.

---

# Rank-1 Model Per Category (load these in the Streamlit app)

Retrained 2026-07-13 on outlier-cleaned data (see "Outlier Removal" below).
Rank-1 winners changed for 6 of 9 categories vs the original run.

| Category | Rank-1 Model | PKL filename | RMSE | R² |
|----------|-------------|--------------|------|----|
| Butter Croissant | LightGBM | `Butter_Croissant_LightGBM_1.pkl` | 2.33 | 0.224 |
| Chocolate Croissant | LinearRegression | `Chocolate_Croissant_LinearRegression_1.pkl` | 1.04 | 0.225 |
| Cookie/Brownie | CatBoost | `Cookie_Brownie_CatBoost_1.pkl` | 1.37 | -0.073 |
| Morning Bun | LightGBM | `Morning_Bun_LightGBM_1.pkl` | 1.42 | 0.096 |
| Muffin | DecisionTree | `Muffin_DecisionTree_1.pkl` | 1.39 | 0.065 |
| Overnight Oats | LinearRegression | `Overnight_Oats_LinearRegression_1.pkl` | 2.99 | 0.466 |
| Savory | LightGBM | `Savory_LightGBM_1.pkl` | 1.65 | -0.081 |
| Sweet Treat | CatBoost | `Sweet_Treat_CatBoost_1.pkl` | 0.97 | -1.879 |
| Yogurt Parfait | DecisionTree | `Yogurt_Parfait_DecisionTree_1.pkl` | 2.00 | 0.422 |

⚠️ **Sweet Treat's rank-1 model is now CatBoost** — it must be fed the raw
12-column feature format (like Cookie/Brownie), not the dummy format.

Sweet Treat still has negative R² across all 6 models — low signal in the
available features; predictions are directional at best. (The in-app warning
banner was removed by request 2026-07-13.) Negative R² here is expected
behavior, not a metric bug: out-of-sample R² goes negative whenever model
error exceeds that of predicting the test-set mean, which is common on these
small (~60–90 row) test sets. The metric formulas were verified line-by-line
against `metrics_explain.md` on 2026-07-13 — all correct.

# Outlier Removal (2026-07-13)

`data/pastry_kohi.csv` was cleaned with a per-category MAD modified z-score
filter on `daily_count` (|M| > 3.5, Iglewicz & Hoaglin 1993): 38 of 4,483
rows (0.8%) removed — one-off demand spikes (e.g. Butter Croissant 16/25
units vs median 5). MAD was preferred over 1.5×IQR because several categories
have discrete counts with IQR = 1, where Tukey fences flag ordinary values.
Full log: `goal1_results/outlier_removal_log.md` (script: `clean_outliers.py`).
Pre-cleaning data preserved as `data/pastry_kohi_raw.csv`;
`data/pastry_kohi_dummy.csv` rebuilt to match (same 29-column schema).

---

# Inference Pipeline (critical for Streamlit)

Two distinct feature formats are required depending on model type:

**Non-CatBoost models** (LR, DT, RF, XGBoost, LightGBM) expect a single-row
DataFrame with exactly these 29 columns in this order:

```python
DUMMY_COLS = [
    'temp_f', 'precip_in', 'humidity_pct', 'wind_mph',
    'Total Occ %', 'Group Occ %', 'Transient Occ %',
    'weather_condition_cloudy', 'weather_condition_rainy', 'weather_condition_snowy',
    'Day of Week_MONDAY', 'Day of Week_SATURDAY', 'Day of Week_SUNDAY',
    'Day of Week_THURSDAY', 'Day of Week_TUESDAY', 'Day of Week_WEDNESDAY',
    'Weekday or Weekend_Weekend',
    'Event_Yes',
    'Month_August', 'Month_December', 'Month_February', 'Month_January',
    'Month_July', 'Month_June', 'Month_March', 'Month_May',
    'Month_November', 'Month_October', 'Month_September',
]
# Reference (dropped) levels: weather_condition=clear, Day of Week=FRIDAY,
# Weekday or Weekend=Weekday, Event=No, Month=April
```

**CatBoost models** (Cookie/Brownie and Sweet Treat rank-1) expect a raw
DataFrame with these 12 columns and string categoricals — no dummy encoding:

```python
RAW_COLS = [
    'temp_f', 'precip_in', 'humidity_pct', 'wind_mph',
    'Total Occ %', 'Group Occ %', 'Transient Occ %',
    'weather_condition', 'Day of Week', 'Weekday or Weekend', 'Event', 'Month',
]
CAT_FEATURES = ['weather_condition', 'Day of Week', 'Weekday or Weekend', 'Event', 'Month']

# Predict:
from catboost import Pool
pred = model.predict(Pool(X_raw, cat_features=CAT_FEATURES))
```

Valid categorical values from training data:
- `weather_condition`: `'clear'`, `'cloudy'`, `'rainy'`, `'snowy'`
- `Day of Week`: `'MONDAY'`, `'TUESDAY'`, `'WEDNESDAY'`, `'THURSDAY'`, `'FRIDAY'`, `'SATURDAY'`, `'SUNDAY'`
- `Weekday or Weekend`: `'Weekday'`, `'Weekend'`
- `Event`: `'Yes'`, `'No'`
- `Month`: full month name, e.g. `'April'`, `'August'`, …, `'September'`

---

# Streamlit App Spec (from prompt.md)

**User inputs (manual)**:
- `Total Occ %` — float, from weekly hotel email
- `Group Occ %` — float
- `Transient Occ %` — float
- `Event` — Yes / No toggle

**Auto-derived**:
- Date features (`Day of Week`, `Weekday or Weekend`, `Month`) — derive from
  tomorrow's date using Python `datetime`
- Weather (`temp_f`, `precip_in`, `humidity_pct`, `wind_mph`, `weather_condition`)
  — pull from a weather API for location: **229 S Main St Suite 100, Bentonville, AR 72712**
  (lat/lon ≈ 36.3729, -94.2088). Weather API not yet chosen.

**Output**: predicted unit count for each of the 9 categories, displayed as a
table or card layout. Round predictions to nearest integer (can't bake 0.7 of a croissant).

---

# Next Step

Create `streamlit_app.py` with manual weather input first (no API yet) so the
full prediction pipeline is validated end-to-end before adding API complexity.
The app should: accept all inputs including weather fields manually → build the
correct feature vectors (dummy for non-CatBoost, raw for CatBoost) → load and
run the 9 rank-1 pkl models → display predicted counts for all 9 categories.
Wire the weather API in as a second pass once the core app works.
