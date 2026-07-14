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
