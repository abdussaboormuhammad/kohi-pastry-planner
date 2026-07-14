import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st
from catboost import Pool

from fetch_weather_cache import API_URL, aggregate_business_hours

st.set_page_config(
    page_title="Kohi Pastry Planner",
    page_icon="🥐",
    layout="wide",
)

# ── Constants ──────────────────────────────────────────────────────────────────

# All day-boundary logic runs on Kohi local time, never the server clock —
# Streamlit Cloud containers run in UTC and would otherwise flip "today"
# at 6/7 PM Central.
CENTRAL = ZoneInfo("America/Chicago")

MODELS_DIR = "goal1_trained_models"
CACHE_PATH = Path(__file__).parent / "data" / "weather_cache.json"
CACHE_MAX_AGE_HOURS = 24

# Only rank-1 (best) model per category, from the 2026-07-13 retrain on
# outlier-cleaned data. Keys and filenames are backend identifiers — do not
# rename them; user-facing labels live in DISPLAY_NAMES.
RANK1_MODELS = {
    "Butter Croissant":    ("Butter_Croissant_LightGBM_1.pkl",             "standard"),
    "Chocolate Croissant": ("Chocolate_Croissant_LinearRegression_1.pkl",  "standard"),
    "Cookie / Brownie":    ("Cookie_Brownie_CatBoost_1.pkl",               "catboost"),
    "Morning Bun":         ("Morning_Bun_LightGBM_1.pkl",                  "standard"),
    "Muffin":              ("Muffin_DecisionTree_1.pkl",                   "standard"),
    "Overnight Oats":      ("Overnight_Oats_LinearRegression_1.pkl",       "standard"),
    "Savory":              ("Savory_LightGBM_1.pkl",                       "standard"),
    "Sweet Treat":         ("Sweet_Treat_CatBoost_1.pkl",                  "catboost"),
    "Yogurt Parfait":      ("Yogurt_Parfait_DecisionTree_1.pkl",           "standard"),
}

# User-facing names only — backend keys above stay untouched.
DISPLAY_NAMES = {
    "Cookie / Brownie": "Cookie",
    "Savory":           "Savory Croissant",
    "Sweet Treat":      "Miso Toast",
}

def display_name(key: str) -> str:
    return DISPLAY_NAMES.get(key, key)

# Baked fresh every morning — shown first, in this order.
BAKE_FRESH = [
    "Butter Croissant", "Chocolate Croissant", "Muffin",
    "Savory", "Morning Bun", "Sweet Treat",
]
# Not baked fresh daily — stocked ahead instead.
STOCK_AHEAD = ["Cookie / Brownie", "Overnight Oats", "Yogurt Parfait"]

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

# 12-column raw feature order — CatBoost models (Cookie/Brownie, Sweet Treat).
RAW_COLS = [
    "temp_f", "precip_in", "humidity_pct", "wind_mph",
    "Total Occ %", "Group Occ %", "Transient Occ %",
    "weather_condition", "Day of Week", "Weekday or Weekend", "Event", "Month",
]
CAT_FEATURES = ["weather_condition", "Day of Week", "Weekday or Weekend", "Event", "Month"]

COND_OPTIONS = ["clear", "cloudy", "rainy", "snowy"]

DEFAULT_WEATHER = {
    "temp_f": 65.0, "precip_in": 0.0,
    "humidity_pct": 50.0, "wind_mph": 5.0, "weather_condition": "clear",
}

# ── Weather: cached file first, live API fallback ──────────────────────────────
# A GitHub Actions job (.github/workflows/weather_cache.yml) refreshes
# data/weather_cache.json at 5 AM Central daily, so normal page loads make no
# API call at all. Both sources use the same business-hours aggregation
# (hourly 6 AM-2 PM readings: mean of numerics, mode of condition).

def load_cached_days() -> tuple[dict, str] | None:
    """Return ({date_iso: weather}, generated_at) if the cache file exists
    and is fresher than CACHE_MAX_AGE_HOURS, else None."""
    try:
        cache = json.loads(CACHE_PATH.read_text())
        generated = datetime.fromisoformat(cache["generated_at"])
        age_hours = (datetime.now(CENTRAL) - generated).total_seconds() / 3600
        if age_hours > CACHE_MAX_AGE_HOURS or not cache.get("days"):
            return None
        return cache["days"], cache["generated_at"]
    except Exception:
        return None


@st.cache_data(ttl=1800)
def fetch_live_days() -> dict:
    """Live Open-Meteo fallback — same aggregation as the cache builder."""
    r = requests.get(API_URL, timeout=15)
    r.raise_for_status()
    return aggregate_business_hours(r.json()["hourly"])


def get_weather_days(today_iso: str) -> tuple[dict, str]:
    """Weather for today + next 7 days, keyed by ISO date.
    Returns (days, source_note); days is {} if every source failed."""
    cached = load_cached_days()
    if cached and today_iso in cached[0]:
        days, generated_at = cached
        return days, f"cached {generated_at[:16].replace('T', ' ')}"
    try:
        return fetch_live_days(), "live Open-Meteo (cache missing or stale)"
    except Exception:
        return {}, "unavailable"

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
    defaults = api_weather or DEFAULT_WEATHER
    with st.expander(
        "Override weather" if api_weather else "Enter weather manually",
        expanded=not bool(api_weather),
    ):
        c1, c2, c3, c4, c5 = st.columns(5)
        temp_f       = c1.number_input("Temp (°F)",          value=float(defaults["temp_f"]),       step=0.5,  key=f"{key}_temp")
        precip_in    = c2.number_input("Precipitation (in)", value=float(defaults["precip_in"]),    step=0.01, format="%.2f", key=f"{key}_precip")
        humidity_pct = c3.number_input("Humidity (%)",       value=float(defaults["humidity_pct"]), step=0.5,  key=f"{key}_hum")
        wind_mph     = c4.number_input("Wind (mph)",         value=float(defaults["wind_mph"]),     step=0.5,  key=f"{key}_wind")
        weather_cond = c5.selectbox(
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


def show_bake_tables(results: dict, day_label: str):
    """Split one prediction dict into the fresh-bake and stock-ahead tables."""
    st.subheader(f"📋 Units to Bake — {day_label}")
    bake_df = pd.DataFrame(
        [{"Pastry": display_name(k), "Units to Bake": results[k]} for k in BAKE_FRESH]
    )
    st.dataframe(bake_df, use_container_width=True, hide_index=True)

    st.subheader("🧺 Make Sure There Is Enough of:")
    stock_df = pd.DataFrame(
        [{"Item": display_name(k), "Units": results[k]} for k in STOCK_AHEAD]
    )
    st.dataframe(stock_df, use_container_width=True, hide_index=True)


def show_weekly_matrices(weekly_results: dict):
    """Two pastry × day matrices: fresh bakes on top, stock-ahead below."""
    day_labels = list(weekly_results.keys())

    st.subheader("📋 Units to Bake")
    bake_df = pd.DataFrame(
        {day: [weekly_results[day][p] for p in BAKE_FRESH] for day in day_labels},
        index=[display_name(p) for p in BAKE_FRESH],
    )
    bake_df.index.name = "Pastry"
    st.dataframe(bake_df, use_container_width=True)

    st.subheader("🧺 Make Sure There Is Enough of:")
    stock_df = pd.DataFrame(
        {day: [weekly_results[day][p] for p in STOCK_AHEAD] for day in day_labels},
        index=[display_name(p) for p in STOCK_AHEAD],
    )
    stock_df.index.name = "Item"
    st.dataframe(stock_df, use_container_width=True)

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    today     = datetime.now(CENTRAL).date()
    yesterday = today - timedelta(days=1)

    st.title("🥐 Kohi Pastry Planner")
    st.markdown(f"**Today:** {today.strftime('%A, %B %d, %Y')}")
    st.divider()

    weather_days, weather_source = get_weather_days(today.isoformat())

    tab_daily, tab_weekly = st.tabs(["📅 Daily", "📆 Weekly (Next 7 Days)"])

    # ══════════════════════════════════════════════════════════════════════════
    # DAILY TAB
    # Uses: yesterday's hotel occupancy + today's weather → today's prediction
    # ══════════════════════════════════════════════════════════════════════════
    with tab_daily:
        st.caption(
            "Enter **yesterday's** hotel occupancy (those guests are today's coffee shop customers). "
            "Today's weather is loaded automatically."
        )

        # Weather — TODAY, averaged over business hours (6 AM-3 PM)
        st.subheader("☁️ Today's Weather")
        api_wx = weather_days.get(today.isoformat())
        if api_wx:
            show_weather_metrics(api_wx)
            st.caption(f"Business-hours average (6 AM-3 PM) · source: {weather_source}")
        else:
            st.warning("Could not load today's weather — enter it manually below.")

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

        # Event or Holiday — TODAY
        st.subheader("📅 Is There an Event or Holiday Today?")
        event = st.radio("Event or Holiday", ["No", "Yes"], horizontal=True,
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

            show_bake_tables(results, today.strftime("%A, %B %d"))
            st.caption(
                f"Generated {datetime.now(CENTRAL).strftime('%I:%M %p')} Central  ·  "
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

        week_dates = [today + timedelta(days=i) for i in range(1, 8)]
        week_wx    = [weather_days.get(d.isoformat()) for d in week_dates]
        weather_ok = all(w is not None for w in week_wx)
        if not weather_ok:
            st.warning("Could not load the full 7-day forecast — weather columns are editable below.")

        rows = []
        for d, w in zip(week_dates, week_wx):
            w = w or DEFAULT_WEATHER
            rows.append({
                "Date":             d.strftime("%a %b %d"),
                "Condition":        w["weather_condition"].capitalize(),
                "Temp (°F)":        w["temp_f"],
                "Rain (in)":        w["precip_in"],
                "Humidity (%)":     w["humidity_pct"],
                "Wind (mph)":       w["wind_mph"],
                "Total Occ %":      70.0,
                "Group Occ %":      20.0,
                "Transient Occ %":  50.0,
                "Event or Holiday": "No",
            })

        edit_df = pd.DataFrame(rows)

        # Weather columns are locked when the forecast loaded; editable when it failed
        disabled_cols = ["Date"] + (["Condition", "Temp (°F)", "Rain (in)", "Humidity (%)", "Wind (mph)"] if weather_ok else [])

        st.markdown("**Fill in occupancy and event/holiday for each day:**")
        edited = st.data_editor(
            edit_df,
            disabled=disabled_cols,
            column_config={
                "Condition":        st.column_config.SelectboxColumn(options=[c.capitalize() for c in COND_OPTIONS]) if not weather_ok else None,
                "Temp (°F)":        st.column_config.NumberColumn(format="%.1f"),
                "Rain (in)":        st.column_config.NumberColumn(format="%.2f"),
                "Humidity (%)":     st.column_config.NumberColumn(format="%.1f"),
                "Wind (mph)":       st.column_config.NumberColumn(format="%.1f"),
                "Total Occ %":      st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=0.5, format="%.1f"),
                "Group Occ %":      st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=0.5, format="%.1f"),
                "Transient Occ %":  st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=0.5, format="%.1f"),
                "Event or Holiday": st.column_config.SelectboxColumn(options=["No", "Yes"]),
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
                        str(row["Event or Holiday"]),
                        d,      # day-of-week / month from each specific day
                    )
                    weekly_results[d.strftime("%a %b %d")] = predict_one(models, X_dummy, X_raw)

            show_weekly_matrices(weekly_results)
            st.caption(
                f"Generated {datetime.now(CENTRAL).strftime('%I:%M %p')} Central  ·  "
                f"{week_start.strftime('%b %d')} – {week_end.strftime('%b %d, %Y')}"
            )


if __name__ == "__main__":
    main()
