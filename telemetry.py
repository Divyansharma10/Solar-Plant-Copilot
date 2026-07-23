"""Live-demo ingestion and vendor-neutral telemetry normalization."""

import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from scipy.fft import fft, fftfreq

from config import (
    AZIMUTH, HOURLY_VARS, LIVE_FORECAST_URL, SEASON_MAP, TILT, TIMEZONE, ZONES
)
from preprocessing import (
    add_zenith_angles, simulate_plant_output
)


REQUIRED_TELEMETRY_COLUMNS = [
    "time", "zone", "ac_power_kw", "global_tilted_irradiance",
    "temperature_2m", "cell_temperature"
]


def normalize_telemetry(df, zone_name, source):
    """Normalize an adapter dataframe into the common SCADA-ready schema."""
    normalized = df.copy()
    normalized["time"] = pd.to_datetime(normalized["time"])
    if normalized["time"].dt.tz is None:
        normalized["time"] = normalized["time"].dt.tz_localize(TIMEZONE)
    normalized["zone"] = zone_name
    normalized["source"] = source

    defaults = {
        "inverter_id": "SIMULATED_ZONE_INVERTER",
        "inverter_status": "SIMULATED",
        "grid_available": True,
        "battery_soc_pct": np.nan,
        "alarm_code": None
    }
    for column, value in defaults.items():
        if column not in normalized:
            normalized[column] = value

    return normalized.sort_values("time").drop_duplicates(
        subset=["time", "zone"], keep="last"
    ).reset_index(drop=True)


def validate_telemetry(df, now=None, max_age_hours=2):
    """Validate schema, freshness, duplicates, missing data, and value ranges."""
    now = pd.Timestamp(now or datetime.now().astimezone())
    if now.tzinfo is None:
        now = now.tz_localize(TIMEZONE)
    else:
        now = now.tz_convert(TIMEZONE)

    missing_columns = [
        column for column in REQUIRED_TELEMETRY_COLUMNS if column not in df
    ]
    if missing_columns:
        return {
            "valid": False,
            "fresh": False,
            "latest_timestamp": None,
            "age_hours": None,
            "duplicate_rows": 0,
            "missing_values": 0,
            "range_violations": [],
            "errors": ["Missing columns: " + ", ".join(missing_columns)]
        }

    timestamps = pd.to_datetime(df["time"])
    latest = timestamps.max()
    if latest.tzinfo is None:
        latest = latest.tz_localize(TIMEZONE)
    age_hours = max(0.0, (now - latest).total_seconds() / 3600)
    duplicates = int(df.duplicated(subset=["time", "zone"]).sum())
    missing_values = int(df[REQUIRED_TELEMETRY_COLUMNS].isna().sum().sum())

    range_violations = []
    checks = {
        "global_tilted_irradiance": (0, 1400),
        "temperature_2m": (-20, 65),
        "cell_temperature": (-20, 90),
        "ac_power_kw": (0, 5500)
    }
    for column, (low, high) in checks.items():
        invalid_count = int((~df[column].between(low, high)).sum())
        if invalid_count:
            range_violations.append(
                f"{column}: {invalid_count} values outside {low}-{high}"
            )

    errors = []
    if duplicates:
        errors.append(f"{duplicates} duplicate timestamp/zone rows")
    if missing_values:
        errors.append(f"{missing_values} missing required values")
    errors.extend(range_violations)

    return {
        "valid": not errors,
        "fresh": age_hours <= max_age_hours,
        "latest_timestamp": latest.isoformat(),
        "age_hours": round(age_hours, 2),
        "duplicate_rows": duplicates,
        "missing_values": missing_values,
        "range_violations": range_violations,
        "errors": errors
    }


def fetch_live_demo(zone_name, past_days=7):
    """Fetch recent Open-Meteo weather and simulate read-only plant telemetry."""
    if zone_name not in ZONES:
        raise ValueError(f"Unknown zone: {zone_name}")

    coords = ZONES[zone_name]
    params = {
        "latitude": coords["lat"],
        "longitude": coords["lon"],
        "hourly": ",".join(HOURLY_VARS),
        "timezone": TIMEZONE,
        "tilt": TILT,
        "azimuth": AZIMUTH,
        "past_days": past_days,
        "forecast_days": 1
    }

    last_error = None
    for attempt in range(3):
        try:
            response = requests.get(
                LIVE_FORECAST_URL, params=params, timeout=20
            )
            response.raise_for_status()
            hourly = response.json().get("hourly", {})
            weather = pd.DataFrame(hourly)
            if weather.empty:
                raise ValueError("Open-Meteo returned no hourly telemetry")
            break
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(attempt + 1)
    else:
        raise ConnectionError(
            f"Live weather request failed after 3 attempts: {last_error}"
        )

    # preprocessing.add_zenith_angles owns timezone localization.
    weather["time"] = pd.to_datetime(weather["time"])
    now = pd.Timestamp.now(tz=TIMEZONE)
    local_now_naive = now.tz_localize(None)
    weather = weather[weather["time"] <= local_now_naive].copy()
    weather["zone"] = zone_name
    weather["latitude"] = coords["lat"]
    weather["longitude"] = coords["lon"]

    with_zenith = add_zenith_angles(weather)
    # Retain nighttime zero-output rows so feed freshness represents
    # communication health even when the plant is not producing.
    simulated = simulate_plant_output(with_zenith)
    telemetry = normalize_telemetry(
        simulated, zone_name, source="Open-Meteo live demo"
    )
    quality = validate_telemetry(telemetry, now=now)
    return telemetry, quality


def build_live_summary(live_df, historical_df, zone_name):
    """Build the summary-row contract consumed by the existing pipeline."""
    live = live_df.sort_values("time").copy()
    history = historical_df[
        historical_df["zone"] == zone_name
    ].copy()
    if len(live) < 3:
        raise ValueError("At least three live telemetry rows are required")

    live["hour"] = pd.to_datetime(live["time"]).dt.hour
    history["hour"] = pd.to_datetime(history["time"]).dt.hour
    hourly_average = history.groupby("hour")["ac_power_kw"].mean()
    live["residual"] = (
        live["ac_power_kw"] - live["hour"].map(hourly_average)
    )

    signal = live["ac_power_kw"].to_numpy()
    frequencies = fftfreq(len(signal), d=1)
    amplitudes = np.abs(fft(signal))
    positive = frequencies > 0
    dominant_period = (
        1 / frequencies[positive][np.argmax(amplitudes[positive])]
        if positive.any() else 0.0
    )
    daytime = live[live["hour"].between(8, 17)]
    start = pd.to_datetime(live["time"].iloc[0])
    end = pd.to_datetime(live["time"].iloc[-1])

    return pd.DataFrame([{
        "window_start": start,
        "window_end": end,
        "month": start.month,
        "season": SEASON_MAP.get(start.month, "Unknown"),
        "dominant_period_hours": round(float(dominant_period), 2),
        "dominant_amplitude": round(float(amplitudes[positive].max()), 2),
        "residual_mean": round(float(live["residual"].mean()), 2),
        "residual_std": round(float(live["residual"].std()), 2),
        "residual_max_dev": round(float(live["residual"].max()), 2),
        "residual_min_dev": round(float(live["residual"].min()), 2),
        "abs_max_deviation_kw": round(
            float(live["residual"].abs().max()), 2
        ),
        "daytime_avg_kw": round(float(daytime["ac_power_kw"].mean()), 2),
        "zone": zone_name
    }])
