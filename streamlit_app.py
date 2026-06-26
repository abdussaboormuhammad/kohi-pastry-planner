import streamlit as st
import pandas as pd
import pickle
import requests
from datetime import datetime, timedelta
from catboost import Pool

st.set_page_config(
    page_title="Kohi Pastry Planner",
    page_icon="🥐",
    layout="wide",
)

# ── Constants ──────────────────────────────────────────────────────────────────

MODELS_DIR = "goal1_trained_models"

# Only rank-1 (best) model per category. Ranks 2 and 3 are intentionally ignored.
RANK1_MODELS = {
    "Butter Croissant":    ("Butter_Croissant_LinearRegression_1.pkl",    "standard"),
    "Chocolate Croissant": ("Chocolate_Croissant_LinearRegression_1.pkl", "standard"),
    "Cookie / Brownie":    ("Cookie_Brownie_CatBoost_1.pkl",              "catboost"),
    "Morning Bun":         ("Morning_Bun_LinearRegression_1.pkl",         "standard"),
    "Muffin":              ("Muffin_RandomForest_1.pkl",                  "standard"),
    "Overnight Oats":      ("Overnight_Oats_LightGBM_1.pkl",             "standard"),
    "Savory":              ("Savory_LightGBM_1.pkl",                      "standard"),
    "Sweet Treat":         ("Sweet_Treat_LightGBM_1.pkl",                "standard"),
    "Yogurt Parfait":      ("Yogurt_Parfait_LinearRegression_1.pkl",      "standard"),
}

# 29-column dummy-encoded feature order — non-CatBoost models.
# Reference levels: weather=clear, day=FRIDAY, weekday/weekend=Weekday, event=No, month=April
DUMMY_COLS = [
    "temp_f", "precip_in", "humidity_pct", "wind_mph",
    "Total Occ %", "Group Occ %", "Transient Occ %",
    "weather_condition_cloudy", "weather_condition_rainy", "weather_condition_snowy",
    "Day of Week_MONDAY", "Day of Week_SATURDAY", "Day of Week_SUNDAY",
    "Day of Week_THURSDAY", "Day of Week_TUESDAY", "Day of Week_WEDNESDAY",
    "Weekday or Weekend_Weekend",
    "Event_Yes",
    "Month_August", "Month_December", "Month_February", "Month_January",
    "Month_July", "Month_June", "Month_March", "Month_May",
    "Month_November", "Month_October", "Month_September",
]

# 12-column raw feature order — CatBoost (Cookie/Brownie) only.
RAW_COLS = [
    "temp_f", "precip_in", "humidity_pct", "wind_mph",
    "Total Occ %", "Group Occ %", "Transient Occ %",
    "weather_condition", "Day of Week", "Weekday or Weekend", "Event", "Month",
]
CAT_FEATURES = ["weather_condition", "Day of Week", "Weekday or Weekend", "Event", "Month"]

COND_OPTIONS = ["clear", "cloudy", "rainy", "snowy"]

WMO_MAP = {
    0:  "clear",
    1: "cloudy",  2: "cloudy",  3: "cloudy",
    45: "cloudy", 48: "cloudy",
    51: "rainy",  53: "rainy",  55: "rainy",
    56: "rainy",  57: "rainy",
    61: "rainy",  63: "rainy",  65: "rainy",
    66: "rainy",  67: "rainy",
    71: "snowy",  73: "snowy",  75: "snowy",  77: "snowy",
    80: "rainy",  81: "rainy",  82: "rainy",
    85: "snowy",  86: "snowy",
    95: "rainy",  96: "rainy",  99: "rainy",
}

_BASE_URL = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude=36.3729&longitude=-94.2088"
    "&hourly=relative_humidity_2m"
    "&daily=temperature_2m_max,precipitation_sum,wind_speed_10m_max,weather_code"
    "&temperature_unit=fahrenheit&wind_speed_unit=mph&precipitation_unit=inch"
    "&timezone=America%2FChicago"
)

# ── Weather helpers ────────────────────────────────────────────────────────────

def _parse_daily(d: dict, idx: int) -> dict:
    daily    = d["daily"]
    wmo      = daily["weather_code"][idx]
    hum_vals = d["hourly"]["relative_humidity_2m"][idx * 24 : (idx + 1) * 24]
    humidity = round(sum(hum_vals) / len(hum_vals), 1) if hum_vals else 50.0
    return {
        "date":              daily["time"][idx],
        "temp_f":            round(daily["temperature_2m_max"][idx], 1),
        "precip_in":         round(daily["precipitation_sum"][idx], 2),
        "humidity_pct":      humidity,
        "wind_mph":          round(daily["wind_speed_10m_max"][idx], 1),
        "weather_condition": WMO_MAP.get(wmo, "clear"),
    }


@st.cache_data(ttl=1800)
def fetch_today_weather() -> dict:
    """Today's forecast — used for the Daily tab."""
    r = requests.get(_BASE_URL + "&forecast_days=1", timeout=10)
    r.raise_for_status()
    return _parse_daily(r.json(), 0)


@st.cache_data(ttl=1800)
def fetch_week_weather() -> list[dict]:
    """7-day weather forecast (tomorrow through today+7) — used for the Weekly tab."""
    r = requests.get(_BASE_URL + "&forecast_days=8", timeout=10)
    r.raise_for_status()
    d = r.json()
    # Index 0 = today (covered by Daily tab); indices 1-7 = next 7 days
    return [_parse_daily(d, i) for i in range(1, 8)]

# ── Model loading ──────────────────────────────────────────────────────────────

@st.cache_resource
def load_models() -> dict:
    models = {}
    for name, (fname, mtype) in RANK1_MODELS.items():
        with open(f"{MODELS_DIR}/{fname}", "rb") as f:
            models[name] = (pickle.load(f), mtype)
    return models

# ── Feature engineering ────────────────────────────────────────────────────────

def build_features(w: dict, occ_total: float, occ_group: float,
                   occ_transient: float, event: str, data_date) -> tuple:
    dow      = data_date.strftime("%A").upper()
    is_wkend = "Weekend" if dow in ("SATURDAY", "SUNDAY") else "Weekday"
    month    = data_date.strftime("%B")
    cond     = w["weather_condition"]

    row = dict.fromkeys(DUMMY_COLS, 0)
    row.update({
        "temp_f":          w["temp_f"],
        "precip_in":       w["precip_in"],
        "humidity_pct":    w["humidity_pct"],
        "wind_mph":        w["wind_mph"],
        "Total Occ %":     occ_total,
        "Group Occ %":     occ_group,
        "Transient Occ %": occ_transient,
    })
    if cond != "clear" and f"weather_condition_{cond}" in row:
        row[f"weather_condition_{cond}"] = 1
    if f"Day of Week_{dow}" in row:
        row[f"Day of Week_{dow}"] = 1
    if is_wkend == "Weekend":
        row["Weekday or Weekend_Weekend"] = 1
    if event == "Yes":
        row["Event_Yes"] = 1
    if month != "April" and f"Month_{month}" in row:
        row[f"Month_{month}"] = 1

    X_dummy = pd.DataFrame([row])[DUMMY_COLS]
    X_raw   = pd.DataFrame([{
        "temp_f":            w["temp_f"],
        "precip_in":        w["precip_in"],
        "humidity_pct":     w["humidity_pct"],
        "wind_mph":         w["wind_mph"],
        "Total Occ %":      occ_total,
        "Group Occ %":      occ_group,
        "Transient Occ %":  occ_transient,
        "weather_condition":cond,
        "Day of Week":      dow,
        "Weekday or Weekend":is_wkend,
        "Event":            event,
        "Month":            month,
    }])[RAW_COLS]

    return X_dummy, X_raw


def predict_one(models: dict, X_dummy: pd.DataFrame, X_raw: pd.DataFrame) -> dict:
    results = {}
    for name, (model, mtype) in models.items():
        if mtype == "catboost":
            val = model.predict(Pool(X_raw, cat_features=CAT_FEATURES))
        else:
            val = model.predict(X_dummy)
        results[name] = max(0, round(float(val[0])))
    return results

# ── Shared UI helpers ──────────────────────────────────────────────────────────

def show_weather_metrics(w: dict):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Temp",      f"{w['temp_f']} °F")
    c2.metric("Rain",      f"{w['precip_in']} in")
    c3.metric("Humidity",  f"{w['humidity_pct']}%")
    c4.metric("Wind",      f"{w['wind_mph']} mph")
    c5.metric("Condition", w["weather_condition"].capitalize())


def weather_override_expander(api_weather: dict | None, key: str) -> dict:
    defaults = api_weather or {
        "temp_f": 65.0, "precip_in": 0.0,
        "humidity_pct": 50.0, "wind_mph": 5.0, "weather_condition": "clear",
    }
    with st.expander(
        "Override weather" if api_weather else "Enter weather manually",
        expanded=not bool(api_weather),
    ):
        temp_f       = st.number_input("Temp (°F)",          value=float(defaults["temp_f"]),       step=0.5,  key=f"{key}_temp")
        precip_in    = st.number_input("Precipitation (in)", value=float(defaults["precip_in"]),    step=0.01, format="%.2f", key=f"{key}_precip")
        humidity_pct = st.number_input("Humidity (%)",       value=float(defaults["humidity_pct"]), step=0.5,  key=f"{key}_hum")
        wind_mph     = st.number_input("Wind (mph)",         value=float(defaults["wind_mph"]),     step=0.5,  key=f"{key}_wind")
        weather_cond = st.selectbox(
            "Condition", COND_OPTIONS,
            index=COND_OPTIONS.index(defaults["weather_condition"]),
            key=f"{key}_cond",
        )
    return {
        "temp_f":            temp_f,
        "precip_in":         precip_in,
        "humidity_pct":      humidity_pct,
        "wind_mph":          wind_mph,
        "weather_condition": weather_cond,
    }


def sweet_treat_warning():
    st.warning(
        "⚠️ **Sweet Treat** prediction has low reliability (R² = −2.195 across all models). "
        "Use as a rough guide only."
    )

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    today     = datetime.now().date()
    yesterday = today - timedelta(days=1)

    st.title("🥐 Kohi Pastry Planner")
    st.markdown(f"**Today:** {today.strftime('%A, %B %d, %Y')}")
    st.divider()

    tab_daily, tab_weekly = st.tabs(["📅 Daily", "📆 Weekly (Last 7 Days)"])

    # ══════════════════════════════════════════════════════════════════════════
    # DAILY TAB
    # Uses: yesterday's hotel occupancy + today's weather → today's prediction
    # ══════════════════════════════════════════════════════════════════════════
    with tab_daily:
        st.caption(
            "Enter **yesterday's** hotel occupancy (those guests are today's coffee shop customers). "
            "Today's weather is loaded automatically."
        )
        st.divider()

        # Weather — TODAY's forecast
        st.subheader("☁️ Today's Weather")
        api_wx = None
        try:
            api_wx = fetch_today_weather()
        except Exception as exc:
            st.warning(f"Could not load today's weather — enter it manually below.  \n_(Error: {exc})_")

        if api_wx:
            show_weather_metrics(api_wx)

        weather = weather_override_expander(api_wx, key="daily")
        st.divider()

        # Occupancy — YESTERDAY's numbers
        st.subheader(f"🏨 Yesterday's Hotel Occupancy  ({yesterday.strftime('%a %b %d')})")
        st.caption("From your weekly hotel email.")
        col1, col2, col3 = st.columns(3)
        with col1:
            occ_total     = st.number_input("Total Occ %",     min_value=0.0, max_value=100.0, value=70.0, step=0.5, key="d_tot")
        with col2:
            occ_group     = st.number_input("Group Occ %",     min_value=0.0, max_value=100.0, value=20.0, step=0.5, key="d_grp")
        with col3:
            occ_transient = st.number_input("Transient Occ %", min_value=0.0, max_value=100.0, value=50.0, step=0.5, key="d_trn")
        st.divider()

        # Event — TODAY
        st.subheader("📅 Is There an Event Today?")
        event = st.radio("", ["No", "Yes"], horizontal=True,
                         label_visibility="collapsed", key="d_event")
        st.divider()

        # Predict — date features derived from TODAY
        if st.button("🥐  Get Today's Bake Numbers", type="primary",
                     use_container_width=True, key="d_predict"):
            with st.spinner("Running predictions…"):
                models  = load_models()
                X_dummy, X_raw = build_features(
                    weather, occ_total, occ_group, occ_transient, event,
                    today,      # day-of-week / month from TODAY
                )
                results = predict_one(models, X_dummy, X_raw)

            st.subheader(f"📋 Recommended Units to Bake — {today.strftime('%A, %B %d')}")
            sweet_treat_warning()
            results_df = pd.DataFrame(
                [{"Pastry": k, "Units to Bake": v} for k, v in results.items()]
            )
            st.dataframe(results_df, use_container_width=True, hide_index=True)
            st.caption(
                f"Generated {datetime.now().strftime('%I:%M %p')}  ·  "
                f"Weather: {weather['weather_condition'].capitalize()}, {weather['temp_f']}°F  ·  "
                f"Occupancy from {yesterday.strftime('%a %b %d')}"
            )

    # ══════════════════════════════════════════════════════════════════════════
    # WEEKLY TAB
    # For each of the next 7 days: enter the hotel occ from your weekly email
    # (each day's occ is effectively the previous day's projected figure) +
    # forecasted weather is pulled automatically → 7-day prediction plan
    # ══════════════════════════════════════════════════════════════════════════
    with tab_weekly:
        week_start = today + timedelta(days=1)   # tomorrow
        week_end   = today + timedelta(days=7)   # 7 days out

        st.markdown(f"### {week_start.strftime('%B %d')} – {week_end.strftime('%B %d, %Y')}")
        st.caption(
            "For each upcoming day, enter the projected hotel occupancy from your weekly email "
            "(use the previous day's figure — those guests become the next day's coffee shop customers). "
            "Weather forecasts are loaded automatically."
        )
        st.divider()

        # Fetch 7-day weather forecast (tomorrow through today+7)
        api_week_wx: list[dict] = []
        weather_ok = True
        try:
            api_week_wx = fetch_week_weather()
        except Exception as exc:
            st.warning(
                f"Could not auto-fetch weekly weather forecast — weather columns are editable below.  \n"
                f"_(Error: {exc})_"
            )
            weather_ok = False

        # Build list of 7 future date objects oldest → newest (matches API order)
        week_dates = [today + timedelta(days=i) for i in range(1, 8)]

        # Build the DataFrame for st.data_editor
        rows = []
        for i, d in enumerate(week_dates):
            w = api_week_wx[i] if (weather_ok and i < len(api_week_wx)) else {
                "temp_f": 65.0, "precip_in": 0.0,
                "humidity_pct": 50.0, "wind_mph": 5.0, "weather_condition": "clear",
            }
            rows.append({
                "Date":            d.strftime("%a %b %d"),
                "Condition":       w["weather_condition"].capitalize(),
                "Temp (°F)":       w["temp_f"],
                "Rain (in)":       w["precip_in"],
                "Humidity (%)":    w["humidity_pct"],
                "Wind (mph)":      w["wind_mph"],
                "Total Occ %":     70.0,
                "Group Occ %":     20.0,
                "Transient Occ %": 50.0,
                "Event":           "No",
            })

        edit_df = pd.DataFrame(rows)

        # Weather columns are locked when fetch succeeded; editable when it failed
        disabled_cols = ["Date"] + (["Condition", "Temp (°F)", "Rain (in)", "Humidity (%)", "Wind (mph)"] if weather_ok else [])

        st.markdown("**Fill in occupancy and event for each day:**")
        edited = st.data_editor(
            edit_df,
            disabled=disabled_cols,
            column_config={
                "Condition":       st.column_config.SelectboxColumn(options=[c.capitalize() for c in COND_OPTIONS]) if not weather_ok else None,
                "Temp (°F)":       st.column_config.NumberColumn(format="%.1f"),
                "Rain (in)":       st.column_config.NumberColumn(format="%.2f"),
                "Humidity (%)":    st.column_config.NumberColumn(format="%.1f"),
                "Wind (mph)":      st.column_config.NumberColumn(format="%.1f"),
                "Total Occ %":     st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=0.5, format="%.1f"),
                "Group Occ %":     st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=0.5, format="%.1f"),
                "Transient Occ %": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=0.5, format="%.1f"),
                "Event":           st.column_config.SelectboxColumn(options=["No", "Yes"]),
            },
            hide_index=True,
            use_container_width=True,
            key="weekly_editor",
        )

        st.divider()

        if st.button("🥐  Get Weekly Bake Numbers", type="primary",
                     use_container_width=True, key="w_predict"):
            with st.spinner("Running predictions for all 7 days…"):
                models = load_models()
                weekly_results: dict[str, dict] = {}

                for i, d in enumerate(week_dates):
                    row = edited.iloc[i]
                    w = {
                        "temp_f":            float(row["Temp (°F)"]),
                        "precip_in":         float(row["Rain (in)"]),
                        "humidity_pct":      float(row["Humidity (%)"]),
                        "wind_mph":          float(row["Wind (mph)"]),
                        "weather_condition": str(row["Condition"]).lower(),
                    }
                    X_dummy, X_raw = build_features(
                        w,
                        float(row["Total Occ %"]),
                        float(row["Group Occ %"]),
                        float(row["Transient Occ %"]),
                        str(row["Event"]),
                        d,      # day-of-week / month from each specific day
                    )
                    weekly_results[d.strftime("%a %b %d")] = predict_one(models, X_dummy, X_raw)

            st.subheader("📋 Weekly Bake Plan")
            sweet_treat_warning()

            # Rows = pastry categories, columns = days
            pastries   = list(RANK1_MODELS.keys())
            day_labels = list(weekly_results.keys())
            matrix     = {day: [weekly_results[day][p] for p in pastries] for day in day_labels}
            results_df = pd.DataFrame(matrix, index=pastries)
            results_df.index.name = "Pastry"
            st.dataframe(results_df, use_container_width=True)
            st.caption(
                f"Generated {datetime.now().strftime('%I:%M %p')}  ·  "
                f"{week_start.strftime('%b %d')} – {week_end.strftime('%b %d, %Y')}"
            )


if __name__ == "__main__":
    main()
