# Kohi Pastry Planner — Streamlit App Build Prompt

Build a Streamlit web app for Kohi Coffee Shop's head pastry chef. Every morning
the chef opens the app, enters yesterday's hotel occupancy and whether there is an
event, and the app outputs predicted unit counts for 9 pastry categories.

The app has two tabs: Daily and Weekly.

---

## Files to create

- `streamlit_app.py` — the app (project root)
- `requirements.txt` — package list for Streamlit Cloud
- `.streamlit/config.toml` — branding theme

Do not modify any existing files (goal1_main.py, add_sweet_treat.py, data files,
pkl files).

---

## Models

Load ONLY the 9 rank-1 (best) model per category from `goal1_trained_models/`.
There are 27 .pkl files total (3 per category) — ignore `_2.pkl` and `_3.pkl`.
The 9 rank-1 files, confirmed against `Summary of Models.png`:

| Category | File | Type |
|---|---|---|
| Butter Croissant | `Butter_Croissant_LinearRegression_1.pkl` | standard |
| Chocolate Croissant | `Chocolate_Croissant_LinearRegression_1.pkl` | standard |
| Cookie / Brownie | `Cookie_Brownie_CatBoost_1.pkl` | **CatBoost** |
| Morning Bun | `Morning_Bun_LinearRegression_1.pkl` | standard |
| Muffin | `Muffin_RandomForest_1.pkl` | standard |
| Overnight Oats | `Overnight_Oats_LightGBM_1.pkl` | standard |
| Savory | `Savory_LightGBM_1.pkl` | standard |
| Sweet Treat | `Sweet_Treat_LightGBM_1.pkl` | standard |
| Yogurt Parfait | `Yogurt_Parfait_LinearRegression_1.pkl` | standard |

---

## Feature formats (critical)

Two distinct input formats are required depending on model type.

### Standard models (all except CatBoost)
A single-row DataFrame with exactly these 29 columns in this order:

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
# Reference (dropped) levels:
#   weather_condition = clear
#   Day of Week       = FRIDAY
#   Weekday/Weekend   = Weekday
#   Event             = No
#   Month             = April
```

### CatBoost model (Cookie / Brownie only)
A raw 12-column DataFrame with string categoricals — no dummy encoding:

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

Valid categorical values:
- `weather_condition`: `'clear'`, `'cloudy'`, `'rainy'`, `'snowy'`
- `Day of Week`: `'MONDAY'` … `'SUNDAY'` (all caps)
- `Weekday or Weekend`: `'Weekday'`, `'Weekend'`
- `Event`: `'Yes'`, `'No'`
- `Month`: full name e.g. `'June'`, `'July'` …

---

## Weather API (Open-Meteo, free, no key required)

Location: Bentonville AR — lat=36.3729, lon=-94.2088

Base URL (reuse for both calls):
```
https://api.open-meteo.com/v1/forecast
  ?latitude=36.3729&longitude=-94.2088
  &hourly=relative_humidity_2m
  &daily=temperature_2m_max,precipitation_sum,wind_speed_10m_max,weather_code
  &temperature_unit=fahrenheit&wind_speed_unit=mph&precipitation_unit=inch
  &timezone=America%2FChicago
```

⚠️ Do NOT include `time` in the `daily` parameter list — it is returned
automatically and including it causes a 400 error.

**For today's weather (Daily tab):** append `&forecast_days=1` → parse index 0.

**For the 7-day forecast (Weekly tab):** append `&forecast_days=8` → parse
indices 1–7 (index 0 = today, already covered by Daily tab).

**Humidity** is not in Open-Meteo's daily variables. Derive it from the hourly
`relative_humidity_2m` array: for day index `i`, average
`hourly[i*24 : (i+1)*24]`.

**WMO weather code → condition mapping:**
```python
WMO_MAP = {
    0: 'clear',
    1: 'cloudy', 2: 'cloudy', 3: 'cloudy', 45: 'cloudy', 48: 'cloudy',
    51: 'rainy', 53: 'rainy', 55: 'rainy', 56: 'rainy', 57: 'rainy',
    61: 'rainy', 63: 'rainy', 65: 'rainy', 66: 'rainy', 67: 'rainy',
    71: 'snowy', 73: 'snowy', 75: 'snowy', 77: 'snowy',
    80: 'rainy', 81: 'rainy', 82: 'rainy', 85: 'snowy', 86: 'snowy',
    95: 'rainy', 96: 'rainy', 99: 'rainy',
}
```

Cache today's weather with `ttl=1800`. Cache the weekly forecast with `ttl=1800`.

---

## Tab 1 — Daily

**Business logic:**
- Yesterday's hotel guests are today's coffee shop customers, so the chef enters
  **yesterday's** hotel occupancy.
- Weather should reflect **today's** actual conditions, not yesterday's and not
  tomorrow's forecast.
- Day-of-week, Weekday/Weekend, and Month features must be derived from
  **today's date** (the day being predicted).

**Auto-fetched on load:**
- Today's weather via `forecast_days=1`
- Display as 5 metric tiles: Temp, Rain, Humidity, Wind, Condition

**Manual inputs:**
- Yesterday's hotel occupancy: Total Occ %, Group Occ %, Transient Occ %
  (number inputs, 0–100, step 0.5)
- Label the occupancy section with yesterday's date so the chef knows
  which email to pull from, e.g. "Yesterday's Hotel Occupancy (Thu Jun 25)"
- Event today? — Yes / No radio, horizontal layout

**Weather override expander:**
- Collapsed by default when the API succeeds (pre-filled with API values)
- Expanded and required when the API fails (show error message above it)
- The expander values are always the source of truth for predictions

**Output:**
- Predict button: "🥐 Get Today's Bake Numbers"
- Results: simple table — Pastry | Units to Bake (rounded to nearest int, min 0)
- Sweet Treat warning banner (see below)
- Caption: generated time + weather used + occupancy date

---

## Tab 2 — Weekly

**Business logic:**
- The chef receives a weekly hotel email with projected occupancy for each
  upcoming day. For each future day, the relevant occupancy is the previous day's
  projected figure (those guests become the next morning's customers).
- Weather should be the **7-day forecast** for each upcoming day.
- Day-of-week, Weekday/Weekend, and Month features must be derived from each
  specific future day's date.

**Auto-fetched on load:**
- 7-day weather forecast via `forecast_days=8`, indices 1–7
- Dates shown: tomorrow through today+7

**Input method — spreadsheet-style data editor (st.data_editor):**
Pre-fill a 7-row DataFrame. When the weather fetch succeeds, lock the weather
columns (disabled) and let the chef edit only the occupancy and event columns.
If the weather fetch fails, make the weather columns editable too.

Columns:

| Column | Editable? | Notes |
|---|---|---|
| Date | No (always locked) | e.g. "Sat Jun 28" |
| Condition | No (locked when fetch OK) | Capitalize for display |
| Temp (°F) | No (locked when fetch OK) | |
| Rain (in) | No (locked when fetch OK) | |
| Humidity (%) | No (locked when fetch OK) | |
| Wind (mph) | No (locked when fetch OK) | |
| Total Occ % | Yes | NumberColumn, 0–100, step 0.5 |
| Group Occ % | Yes | NumberColumn, 0–100, step 0.5 |
| Transient Occ % | Yes | NumberColumn, 0–100, step 0.5 |
| Event | Yes | SelectboxColumn: ["No", "Yes"] |

Default occupancy values: 70 / 20 / 50. Default event: No.

**Output:**
- Predict button: "🥐 Get Weekly Bake Numbers"
- Results: pastry × day matrix — rows = 9 pastry categories, columns = 7 dates
- Sweet Treat warning banner (see below)
- Caption: generated time + date range

---

## Sweet Treat warning (both tabs)

Show this as a `st.warning()` above the results table in both tabs:

> ⚠️ **Sweet Treat** prediction has low reliability (R² = −2.195 across all
> models). Use as a rough guide only.

---

## UX requirements

- `st.set_page_config(layout="wide")` — use wide layout
- Two tabs at the top: `📅 Daily` and `📆 Weekly (Last 7 Days)`
- Non-technical users (pastry chefs) — keep language plain and friendly
- Show today's date in the app header
- Weather metrics displayed as `st.metric()` tiles (5 columns)
- On page load, weather is fetched automatically — chef should not need to click
  anything extra before seeing the forecast
- If weather fetch fails, show a clear error message and expand the manual input
  section automatically

---

## Deployment files

**`requirements.txt`:**
```
streamlit
pandas
numpy
scikit-learn
lightgbm
catboost
requests
```

**`.streamlit/config.toml`:**
```toml
[theme]
primaryColor = "#8B1A1A"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F5F5F5"
textColor = "#1A1A1A"
font = "sans serif"
```

**Deployment — Streamlit Community Cloud (free hosting):**
1. Push the repo to GitHub. Include: `streamlit_app.py`, `requirements.txt`,
   `.streamlit/config.toml`, and `goal1_trained_models/` (rank-1 pkl files).
   Optionally add a `.gitignore` excluding `*_2.pkl` and `*_3.pkl` to keep the
   repo lean.
2. Go to share.streamlit.io → sign in with GitHub → New app → point at the repo,
   set main file to `streamlit_app.py` → Deploy.
3. Share the generated public URL with the pastry chef as a browser bookmark.
   No installation required on her end.

Note: free tier sleeps after ~7 days of inactivity. First morning load may take
~30 seconds to wake up — acceptable for this use case.
