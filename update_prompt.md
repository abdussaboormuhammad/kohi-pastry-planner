# Kohi Pastry Planner — Model Rebuild, Weather Pipeline & UI Overhaul

## Goal

Fix the Goal 1 pastry-demand models and their evaluation metrics, rebuild the
results website used to visually verify them, replace the Streamlit app's
live weather API calls with a pre-computed daily cache (to respect free-tier
rate limits and use correct business-hour averages), and overhaul the app's
UI/naming/layout so the pastry chef spends less time scrolling. Ship all four
workstreams together — the UI changes touch the same file as the weather-cache
change, and the model rebuild's outputs (metrics, warnings) feed directly into
the UI copy.

## Context

- **Read `handoff.md` first** for full project history and architecture. Also
  skim `prompt.md` (current Streamlit spec) and `v8_kohi_modeling_v2 copy/prompt.md`
  (original modeling spec) before changing anything — don't restate them back
  to me, just use them as ground truth for existing conventions.
- Stack: Streamlit app (`streamlit_app.py`) deployed on **Streamlit Community
  Cloud, free tier** — no persistent server, no built-in cron, the app can
  sleep after inactivity. Models: scikit-learn / XGBoost / LightGBM / CatBoost
  `.pkl` files in `goal1_trained_models/`. Weather: Open-Meteo (free, no key)
  for Bentonville, AR (36.3729, -94.2088). Kohi's business hours are
  **6 AM–3 PM**.
- The modeling code that produced the current (suspect) models lives in
  `v8_kohi_modeling_v2 copy/` (`goal1_main.py`, `add_sweet_treat.py`). It
  trained on `data/pastry_kohi.csv` (3,979 rows, +504 Sweet Treat rows added
  later → 4,483) and its one-hot-encoded counterpart
  `data/pastry_kohi_dummy.csv` (29 feature columns). Per-category test sets
  are small — a chronological 70/15/15 split means each category's test set
  is roughly 60–90 rows.
- **Correction to check before assuming a bug**: `goal1_main.py` computes R²
  with `sklearn.metrics.r2_score`, i.e. the textbook `1 − SS_res/SS_tot`.
  Out-of-sample R² is *not* bounded at 0 — it goes negative whenever a
  model's test-set errors are larger than the errors from just predicting the
  test-set mean. That's common on small, noisy test sets, which several of
  these categories have. So the negative R² values in the current rank-1
  table (Cookie/Brownie −0.064, Muffin −0.026, Savory −0.075, Sweet Treat
  −2.195) are not automatically evidence of a broken formula. Verify the
  RMSE / MAE / R² / MAPE / Median AE implementations in `compute_metrics` and
  `safe_mape` line-by-line against `v8_kohi_modeling_v2 copy/metrics_explain.md`
  first, and only report/fix something as broken if you find an actual defect
  (wrong axis, train/test leakage, off-by-one in the split indices, etc.). If
  the formulas check out, say so plainly — the real lever for improving these
  numbers is the outlier cleanup below, not the math.
- Outlier-removal method is your call: pick a standard, defensible approach
  (e.g. IQR-based, applied per pastry category to `daily_count` and/or the
  numeric predictors), and document exactly what you removed and why — add a
  short section to `model_explain.md` before retraining. Don't silently drop
  rows without a paper trail.
- The 5 AM weather job needs to run without a persistent server. Use a
  **GitHub Actions scheduled workflow**: it runs on GitHub's infrastructure
  regardless of whether the Streamlit app is awake, calls Open-Meteo, writes
  the aggregated results to a JSON file in this repo, and commits it back.
  The Streamlit app then just reads that file — no live API call on page
  load, so the free-tier rate-limit problem goes away. GitHub Actions cron
  runs in UTC and does not auto-adjust for daylight saving; handle the
  CST/CDT shift explicitly (Central is UTC-6 in winter, UTC-5 in summer)
  rather than hardcoding one UTC offset and drifting an hour twice a year.
- Streamlit Community Cloud containers most likely run in UTC.
  `streamlit_app.py` currently computes `today = datetime.now().date()` with
  no timezone — if the server clock isn't Central, that date can silently
  disagree with the "today" Open-Meteo resolves via its
  `timezone=America/Chicago` parameter. This is a plausible root cause for
  "the weather looks the same every day" and/or subtle date-boundary bugs
  near midnight Central. Investigate it, and make all day-boundary logic
  explicitly timezone-aware to `America/Chicago` (`zoneinfo.ZoneInfo`) rather
  than relying on server-local time.
- The 9 pastry categories are fixed by the trained models
  (`goal1_trained_models/*.pkl`). Do not rename the internal dict keys or
  filenames keyed off `Cookie_Brownie`, `Savory`, `Sweet_Treat`, etc. — only
  the **user-facing display strings** in the Streamlit UI change. The backend
  category identifiers stay as-is so model loading keeps working.

## Deliverables

### 1. Data cleaning + model rebuild
- In `v8_kohi_modeling_v2 copy/`, review the metric formulas in
  `goal1_main.py` (`compute_metrics`, `safe_mape`) against
  `metrics_explain.md`. Report findings either way — bug found and fixed, or
  formulas confirmed correct with an explanation of why negative R² is
  expected here.
- Remove statistical outliers from `data/pastry_kohi.csv` and rebuild
  `data/pastry_kohi_dummy.csv` to match (same 29-column dummy schema
  documented in `handoff.md`). Do this in the top-level `data/` folder — the
  one `streamlit_app.py` and the live models actually use — not just the copy
  inside `v8_kohi_modeling_v2 copy/`.
- Re-run the full training pipeline (6 models × 9 categories, same
  train/val/test methodology, ranking rules, and file-naming convention as
  the existing `goal1_main.py` / `add_sweet_treat.py`) on the cleaned data.
  Overwrite `goal1_trained_models/*.pkl`, `goal1_results/tables/`,
  `goal1_results/predictions/`, `goal1_results/importance/`, and
  `goal1_results_report.html`.
- Update `model_explain.md` and `handoff.md`'s "Rank-1 Model Per Category"
  table with the new results, including the outlier-removal writeup.
- If a different model wins rank-1 for any category after retraining, update
  `RANK1_MODELS` in `streamlit_app.py` (filename + model type) to match —
  check this explicitly rather than assuming the old winners still hold.

### 2. Goal 1 Results website
- Rebuild `goal1_results_report.html` from the retrained models so I can
  visually verify: per-category performance tables (RMSE / MAE / R² / MAPE /
  Median AE for all 6 models), prediction scatter plots, and feature
  importance plots (all models except Linear Regression, per the existing
  convention). Keep it self-contained (base64-embedded images, no external
  assets) like the current report.

### 3. Weather pipeline — correct aggregation + scheduled caching
- Redefine the weather aggregation: pull **hourly** Open-Meteo data (not the
  `daily` block) for `temp_f`, `precip_in`, `humidity_pct`, `wind_mph`, and
  the weather code, restricted to Kohi's business hours (6 AM–3 PM Central).
  Average the four numeric metrics over that window; take the **mode** of the
  mapped weather condition (`clear` / `cloudy` / `rainy` / `snowy`) over the
  same window, since it's categorical and can't be averaged.
- Diagnose and fix why the weather has appeared static day-to-day — start
  with the timezone issue flagged above, and verify end-to-end that the date
  displayed in the app matches the date the API data was actually computed
  for.
- Add a GitHub Actions workflow that runs once daily at 5 AM Central (handle
  CST/CDT as noted above):
  - Fetches the current day's business-hours weather (for the Daily tab) and
    the next 7 days' business-hours weather (for the Weekly tab), using the
    new hourly average/mode aggregation.
  - Writes both to a JSON cache file in the repo (e.g. `data/weather_cache.json`)
    with a generation timestamp, and commits/pushes it.
- Update `streamlit_app.py` to read from that cached file instead of calling
  Open-Meteo on every page load. Keep a fallback: if the cache file is
  missing or older than ~24 hours, fall back to a live API call (or the
  existing manual-entry expander) so the chef is never fully blocked if the
  scheduled job fails one morning.

### 4. UI/UX overhaul (`streamlit_app.py`)
- **Today's Weather**: make the section more compact — ensure both the
  auto-fetched metrics *and* the manual override/entry fields lay out
  horizontally (columns), not stacked vertically, so less scrolling is needed
  to reach the input fields below it.
- **Hotel Occupancy**: audit the current layout (the Daily tab already uses
  3 `st.columns`) and make sure both Daily and Weekly sections use a
  horizontal 3-column layout — fix wherever it isn't already the case.
- Remove the Sweet Treat low-reliability warning banner
  (`sweet_treat_warning()` and its call sites) from both tabs.
- Reorder the "Units to Bake" table/matrix to: Butter Croissant, Chocolate
  Croissant, Muffin, Savory Croissant, Morning Bun, Miso Toast.
- Split out a second table, "Make Sure There Is Enough of:", directly beneath
  "Units to Bake", containing: Cookie, Overnight Oats, Yogurt Parfait (these
  aren't baked fresh every morning). Apply the same split to the Weekly
  matrix output too (two matrices instead of one 9-row matrix).
- Rename **display labels only** (leave backend keys/filenames untouched —
  see Context):
  - "Cookie / Brownie" → "Cookie"
  - "Savory" → "Savory Croissant"
  - "Sweet Treat" → "Miso Toast"
- Change "Is There an Event Today?" (and the Weekly tab's "Event" column) to
  "Event or Holiday" phrasing throughout (both tabs).
- Rename the Weekly tab from "Weekly (Last 7 Days)" to "Weekly (Next 7 Days)"
  — this also fixes a pre-existing label bug, since the tab already computes
  tomorrow through +7 days, not the past.

## Out of scope
- Don't touch Goals 2/3 (peak-hour forecasting, Plotly/Dash dashboards) —
  still deferred per the original modeling prompt.
- Don't change which 6 model architectures are tried per category — only the
  data going into them and, if a genuine formula bug is found, the metric
  computation.

## Before you start
If anything above is ambiguous once you're in the code (exact 6 AM–3 PM hour
boundaries, where the GitHub Actions commit should push if branch protection
blocks direct pushes to `main`, etc.), ask me rather than guessing.
