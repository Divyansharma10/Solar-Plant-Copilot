"""Auditable operator recommendations derived from deterministic evidence."""

import pandas as pd

from config import (
    BATTERY_RESERVE_DURATION_HOURS, BATTERY_RESERVE_THRESHOLD_KW
)


SEVERITY_ORDER = {"Critical": 4, "Warning": 3, "Monitor": 2, "Info": 1}


def _rule(rule_id, severity, title, evidence, action, urgency, clear_condition):
    return {
        "rule_id": rule_id,
        "severity": severity,
        "title": title,
        "evidence": evidence,
        "action": action,
        "urgency": urgency,
        "clear_condition": clear_condition
    }


def _max_low_output_hours(window_df):
    """Count the longest consecutive strong-sun run below the grid threshold."""
    if window_df.empty:
        return 0
    data = window_df.copy().sort_values("time")
    data["time"] = pd.to_datetime(data["time"])
    low = (
        (data["global_tilted_irradiance"] >= 300) &
        (data["ac_power_kw"] < BATTERY_RESERVE_THRESHOLD_KW)
    )
    groups = low.ne(low.shift()).cumsum()
    runs = low.groupby(groups).sum()
    return int(runs.max()) if not runs.empty else 0


def evaluate_recommendations(anomaly, window_df):
    """Evaluate operational rules and return a priority-ordered decision."""
    rules = []
    missing = int(anomaly.get("missing_values", 0))
    stuck = anomaly.get("stuck_sensors", [])
    model_pct = float(anomaly.get("model_performance_pct", 0))
    baseline_pct = float(anomaly.get("rolling_baseline_pct", 0))
    irradiance_pct = float(anomaly.get("irradiance_vs_baseline_pct", 100))

    if missing:
        severity = "Critical" if missing > len(window_df) * 0.05 else "Warning"
        rules.append(_rule(
            "DQ-001", severity, "Missing telemetry",
            f"{missing} required sensor values are missing.",
            "Restore telemetry completeness before diagnosing plant equipment.",
            "Immediate" if severity == "Critical" else "Within 1 hour",
            "Clear after required fields contain no missing values for 3 consecutive samples."
        ))

    if stuck:
        sensor_list = ", ".join(stuck)
        rules.append(_rule(
            "DQ-002", "Warning", "Possible stuck sensor",
            f"Repeated identical readings detected for: {sensor_list}.",
            "Check sensor wiring, gateway communication, and timestamp updates.",
            "Within 1 hour",
            "Clear after each affected sensor changes plausibly for 3 consecutive samples."
        ))

    if anomaly.get("curtailment_detected"):
        rules.append(_rule(
            "OPS-001", "Critical", "Strong-sun output deficit",
            f"Physical-model performance is {model_pct:.1f}% during strong irradiance.",
            "Check inverter alarms, plant availability, breaker state, and grid curtailment signals.",
            "Immediate",
            "Clear when physical-model performance remains at or above 90% for 3 strong-sun samples."
        ))
    elif model_pct < 75:
        rules.append(_rule(
            "PERF-002", "Critical", "Severe physical underperformance",
            f"Actual output is only {model_pct:.1f}% of the irradiance/temperature model.",
            "Inspect inverter availability, DC inputs, protection trips, and grid connection immediately.",
            "Immediate",
            "Clear when physical-model performance remains at or above 85% for 3 samples."
        ))
    elif model_pct < 85:
        rules.append(_rule(
            "PERF-001", "Warning", "Physical underperformance",
            f"Actual output is {model_pct:.1f}% of the irradiance/temperature model.",
            "Review inverter status, MPPT/string measurements, soiling, and grid availability.",
            "Within 1 hour",
            "Clear when physical-model performance remains at or above 90% for 3 samples."
        ))

    if anomaly.get("clipping_detected"):
        rules.append(_rule(
            "OPS-002", "Monitor", "Possible inverter clipping",
            "Output is flat near zone capacity during sustained high irradiance.",
            "Review inverter AC limits and the plant DC/AC sizing ratio; no immediate shutdown is required.",
            "Monitor",
            "Clear when high-irradiance output no longer forms a near-capacity plateau."
        ))

    if anomaly.get("cause") == "Weather" and baseline_pct < 95:
        rules.append(_rule(
            "WX-001", "Monitor", "Weather-driven production reduction",
            f"Irradiance is {irradiance_pct:.1f}% and output is {baseline_pct:.1f}% of baseline.",
            "Continue monitoring; do not dispatch equipment maintenance unless physical-model performance declines.",
            "Monitor",
            "Clear when irradiance and output return to at least 95% of their baselines."
        ))

    low_output_hours = _max_low_output_hours(window_df)
    if low_output_hours > BATTERY_RESERVE_DURATION_HOURS:
        rules.append(_rule(
            "GRID-001", "Warning", "Sustained grid-minimum risk",
            f"Output stayed below {BATTERY_RESERVE_THRESHOLD_KW} kW for "
            f"{low_output_hours} consecutive strong-sun samples.",
            "Verify the forecast and battery state of charge; prepare reserve dispatch under the plant procedure.",
            "Within 1 hour",
            f"Clear after output remains above {BATTERY_RESERVE_THRESHOLD_KW} kW for 2 consecutive strong-sun samples."
        ))

    if not rules:
        rules.append(_rule(
            "OPS-000", "Info", "No operational action",
            "No deterministic equipment, grid, weather, or data-quality rule triggered.",
            "No action required; continue routine monitoring.",
            "None",
            "Re-evaluate when new telemetry arrives."
        ))

    rules.sort(
        key=lambda item: SEVERITY_ORDER[item["severity"]], reverse=True
    )
    primary = rules[0]
    return {
        "primary_rule_id": primary["rule_id"],
        "severity": primary["severity"],
        "action": primary["action"],
        "urgency": primary["urgency"],
        "clear_condition": primary["clear_condition"],
        "triggered_rules": rules
    }


def format_recommendation_context(decision):
    """Format rules for a constrained LLM explanation."""
    rule_lines = [
        (
            f"{rule['rule_id']} [{rule['severity']}]: {rule['evidence']} "
            f"Required action: {rule['action']} Urgency: {rule['urgency']}. "
            f"Clear when: {rule['clear_condition']}"
        )
        for rule in decision["triggered_rules"]
    ]
    return (
        f"Primary rule: {decision['primary_rule_id']}. "
        f"Primary action: {decision['action']} "
        f"Primary urgency: {decision['urgency']}.\n" +
        "\n".join(rule_lines)
    )
