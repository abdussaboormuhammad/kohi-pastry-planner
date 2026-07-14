#!/usr/bin/env python3
"""
fetch_weather_cache.py
Kohi Pastry Planner — daily weather cache builder.

Run by a GitHub Actions schedule at 5 AM Central (see
.github/workflows/weather_cache.yml). Pulls hourly Open-Meteo data for
Bentonville, AR and aggregates each day over Kohi's business hours
(6 AM-3 PM Central; the nine hourly readings 06:00-14:00, each stamped at
the start of its hour). Numeric metrics are averaged; the mapped weather
condition takes the mode. Writes data/weather_cache.json covering today
plus the next 7 days.

Stdlib only — no third-party dependencies, so CI needs no pip install.
"""

import json
import os
import ssl
import sys
import urllib.request
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo

try:  # some local Python installs lack the system CA bundle; CI doesn't need this
    import certifi
    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CONTEXT = ssl.create_default_context()

CENTRAL = ZoneInfo("America/Chicago")
LAT, LON = 36.3729, -94.2088
BUSINESS_HOURS = range(6, 15)  # hourly readings 06:00 through 14:00 inclusive

API_URL = (
    "https://api.open-meteo.com/v1/forecast"
    f"?latitude={LAT}&longitude={LON}"
    "&hourly=temperature_2m,precipitation,relative_humidity_2m,wind_speed_10m,weather_code"
    "&temperature_unit=fahrenheit&wind_speed_unit=mph&precipitation_unit=inch"
    "&timezone=America%2FChicago&forecast_days=8"
)

WMO_MAP = {
    0: "clear",
    1: "cloudy", 2: "cloudy", 3: "cloudy", 45: "cloudy", 48: "cloudy",
    51: "rainy", 53: "rainy", 55: "rainy", 56: "rainy", 57: "rainy",
    61: "rainy", 63: "rainy", 65: "rainy", 66: "rainy", 67: "rainy",
    71: "snowy", 73: "snowy", 75: "snowy", 77: "snowy",
    80: "rainy", 81: "rainy", 82: "rainy", 85: "snowy", 86: "snowy",
    95: "rainy", 96: "rainy", 99: "rainy",
}
# Mode tie-break: prefer the operationally worse condition
CONDITION_SEVERITY = {"clear": 0, "cloudy": 1, "rainy": 2, "snowy": 3}


def aggregate_business_hours(hourly: dict) -> dict:
    """Group hourly arrays by local date, keep 6 AM-2 PM readings, aggregate."""
    days = {}
    for i, ts in enumerate(hourly["time"]):  # e.g. "2026-07-13T06:00"
        date_str, hour = ts[:10], int(ts[11:13])
        if hour not in BUSINESS_HOURS:
            continue
        days.setdefault(date_str, []).append({
            "temp_f":       hourly["temperature_2m"][i],
            "precip_in":    hourly["precipitation"][i],
            "humidity_pct": hourly["relative_humidity_2m"][i],
            "wind_mph":     hourly["wind_speed_10m"][i],
            "condition":    WMO_MAP.get(hourly["weather_code"][i], "clear"),
        })

    out = {}
    for date_str, rows in sorted(days.items()):
        rows = [r for r in rows if r["temp_f"] is not None]
        if not rows:
            continue
        counts = Counter(r["condition"] for r in rows)
        top = max(counts.values())
        condition = max((c for c, n in counts.items() if n == top),
                        key=CONDITION_SEVERITY.get)
        out[date_str] = {
            "temp_f":            round(sum(r["temp_f"] for r in rows) / len(rows), 1),
            "precip_in":         round(sum(r["precip_in"] for r in rows) / len(rows), 3),
            "humidity_pct":      round(sum(r["humidity_pct"] for r in rows) / len(rows), 1),
            "wind_mph":          round(sum(r["wind_mph"] for r in rows) / len(rows), 1),
            "weather_condition": condition,
            "hours_used":        len(rows),
        }
    return out


def main() -> int:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(base_dir, "data", "weather_cache.json")

    with urllib.request.urlopen(API_URL, timeout=30, context=SSL_CONTEXT) as resp:
        payload = json.load(resp)

    days = aggregate_business_hours(payload["hourly"])
    now_central = datetime.now(CENTRAL)
    today_str = now_central.date().isoformat()
    if today_str not in days:
        print(f"ERROR: API response has no business-hours data for today ({today_str})",
              file=sys.stderr)
        return 1

    cache = {
        "generated_at": now_central.isoformat(timespec="seconds"),
        "timezone": "America/Chicago",
        "location": {"lat": LAT, "lon": LON},
        "business_hours": "06:00-14:00 hourly readings (Kohi open 6 AM-3 PM Central)",
        "aggregation": "mean of temp/precip/humidity/wind, mode of condition",
        "days": days,
    }
    with open(out_path, "w") as f:
        json.dump(cache, f, indent=2)

    print(f"Wrote {out_path}")
    print(f"  generated_at: {cache['generated_at']}  ({len(days)} days cached)")
    for d, w in days.items():
        print(f"  {d}: {w['temp_f']}°F  {w['precip_in']}in  {w['humidity_pct']}%  "
              f"{w['wind_mph']}mph  {w['weather_condition']} ({w['hours_used']}h)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
