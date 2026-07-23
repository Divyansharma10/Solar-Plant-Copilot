"""Deterministic, auditable anomaly detection for solar telemetry windows."""

import numpy as np
import pandas as pd

from config import (
    GAMMA_PDC, INVERTER_EFFICIENCY, SYSTEM_LOSSES, ZONE_CAPACITY_KW
)


SENSOR_COLUMNS = [
    "global_tilted_irradiance", "temperature_2m", "cell_temperature",
    "ac_power_kw"
]


def _longest_constant_run(series, decimals=3):
    values = pd.to_numeric(series, errors="coerce").round(decimals)
    groups = values.ne(values.shift()).cumsum()
    counts = values.groupby(groups).transform("size")
    return int(counts.max()) if not counts.empty else 0


def _expected_power(window):
    irradiance = window["global_tilted_irradiance"].clip(lower=0)
    cell_temp = window.get("cell_temperature", window["temperature_2m"])
    temp_factor = (1 + GAMMA_PDC * (cell_temp - 25)).clip(lower=0)
    expected = (
        ZONE_CAPACITY_KW * (irradiance / 1000) * temp_factor *
        INVERTER_EFFICIENCY * (1 - SYSTEM_LOSSES)
    )
    return expected.clip(lower=0, upper=ZONE_CAPACITY_KW)


def analyze_window(window_df, historical_df):
    """Return deterministic health metrics and anomaly evidence for one zone."""
    window = window_df.copy()
    history = historical_df.copy()
    window["time"] = pd.to_datetime(window["time"])
    history["time"] = pd.to_datetime(history["time"])

    missing = {
        column: int(window[column].isna().sum())
        for column in SENSOR_COLUMNS if column in window
    }
    missing_total = sum(missing.values())

    valid = window[window["global_tilted_irradiance"] >= 50].copy()
    if valid.empty:
        return {
            "status": "Critical", "cause": "Data quality",
            "performance_ratio_pct": 0.0, "model_performance_pct": 0.0,
            "rolling_baseline_pct": 0.0, "expected_power_kw": 0.0,
            "missing_values": missing_total, "stuck_sensors": [],
            "clipping_detected": False, "curtailment_detected": False,
            "evidence": ["No usable irradiance samples in the selected window."]
        }

    valid["expected_power_kw"] = _expected_power(valid)
    expected_sum = valid["expected_power_kw"].sum()
    model_performance = (
        valid["ac_power_kw"].sum() / expected_sum * 100
        if expected_sum > 0 else 0.0
    )
    pr_denominator = (
        ZONE_CAPACITY_KW * (valid["global_tilted_irradiance"] / 1000).sum()
    )
    performance_ratio = (
        valid["ac_power_kw"].sum() / pr_denominator * 100
        if pr_denominator > 0 else 0.0
    )

    history["month"] = history["time"].dt.month
    history["hour"] = history["time"].dt.hour
    valid["month"] = valid["time"].dt.month
    valid["hour"] = valid["time"].dt.hour
    baseline_source = history[history["global_tilted_irradiance"] >= 50]
    baselines = baseline_source.groupby(["month", "hour"])[
        ["ac_power_kw", "global_tilted_irradiance"]
    ].median().rename(columns={
        "ac_power_kw": "baseline_power_kw",
        "global_tilted_irradiance": "baseline_gti"
    })
    compared = valid.join(baselines, on=["month", "hour"])
    baseline_power = compared["baseline_power_kw"].mean()
    actual_power = compared["ac_power_kw"].mean()
    baseline_pct = (
        actual_power / baseline_power * 100
        if pd.notna(baseline_power) and baseline_power > 0 else 100.0
    )
    current_gti = compared["global_tilted_irradiance"].mean()
    baseline_gti = compared["baseline_gti"].mean()
    irradiance_pct = (
        current_gti / baseline_gti * 100
        if pd.notna(baseline_gti) and baseline_gti > 0 else 100.0
    )

    stuck_sensors = []
    for column in ("global_tilted_irradiance", "temperature_2m", "ac_power_kw"):
        sensor_values = valid[column] if column in valid else pd.Series(dtype=float)
        if _longest_constant_run(sensor_values) >= 4:
            stuck_sensors.append(column)

    high_gti = valid[valid["global_tilted_irradiance"] >= 800]
    clipping_detected = False
    if len(high_gti) >= 4:
        peak_threshold = valid["ac_power_kw"].quantile(0.98) * 0.995
        clipping_detected = bool(
            (high_gti["ac_power_kw"] >= peak_threshold).mean() >= 0.5 and
            high_gti["ac_power_kw"].std() / high_gti["ac_power_kw"].mean() < 0.03 and
            high_gti["ac_power_kw"].mean() >= ZONE_CAPACITY_KW * 0.9
        )

    strong_sun = valid[valid["global_tilted_irradiance"] >= 600]
    curtailment_detected = bool(
        len(strong_sun) >= 3 and
        (strong_sun["ac_power_kw"] < strong_sun["expected_power_kw"] * 0.8).mean() >= 0.2
    )

    evidence = []
    if missing_total:
        evidence.append(f"{missing_total} required sensor values are missing.")
    if stuck_sensors:
        evidence.append("Possible stuck sensors: " + ", ".join(stuck_sensors) + ".")
    if clipping_detected:
        evidence.append("Output plateaus during sustained high irradiance (possible clipping).")
    if curtailment_detected:
        evidence.append("Output is below the physical expectation during strong irradiance.")

    if missing_total or stuck_sensors:
        cause = "Data quality"
    elif curtailment_detected or (model_performance < 85 and irradiance_pct >= 90):
        cause = "Equipment/grid"
    elif baseline_pct < 90 and irradiance_pct < 90:
        cause = "Weather"
    else:
        cause = "Normal variation"

    if cause == "Weather":
        evidence.append(
            f"Irradiance is {irradiance_pct:.1f}% of its zone/month/hour baseline, "
            "supporting a weather-related explanation."
        )
    elif cause == "Equipment/grid" and not curtailment_detected:
        evidence.append(
            "Irradiance is near its baseline while output is below the physical model."
        )

    if missing_total > len(window) * 0.05 or model_performance < 75:
        status = "Critical"
    elif curtailment_detected or stuck_sensors or model_performance < 85:
        status = "Warning"
    elif clipping_detected or baseline_pct < 95 or model_performance < 95:
        status = "Monitor"
    else:
        status = "Normal"

    if not evidence:
        evidence.append("No deterministic equipment or sensor anomaly detected.")

    return {
        "status": status,
        "cause": cause,
        "performance_ratio_pct": float(round(performance_ratio, 1)),
        "model_performance_pct": float(round(model_performance, 1)),
        "rolling_baseline_pct": float(round(baseline_pct, 1)),
        "irradiance_vs_baseline_pct": float(round(irradiance_pct, 1)),
        "expected_power_kw": float(round(valid["expected_power_kw"].mean(), 1)),
        "missing_values": missing_total,
        "stuck_sensors": stuck_sensors,
        "clipping_detected": clipping_detected,
        "curtailment_detected": curtailment_detected,
        "evidence": evidence
    }


def format_anomaly_context(analysis):
    """Format deterministic evidence for inclusion in the cloud prompt."""
    evidence = " ".join(analysis["evidence"])
    return (
        f"Deterministic status: {analysis['status']}. "
        f"Likely cause: {analysis['cause']}. "
        f"Performance ratio: {analysis['performance_ratio_pct']}%. "
        f"Physical-model performance: {analysis['model_performance_pct']}%. "
        f"Rolling baseline performance: {analysis['rolling_baseline_pct']}%. "
        f"Evidence: {evidence}"
    )
