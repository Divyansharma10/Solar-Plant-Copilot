# time_series.py
"""
Solar Plant Copilot — Time Series Analysis
Handles FFT analysis, ARMA on residuals,
and sliding window summary generation.
"""

import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq
from statsmodels.tsa.arima.model import ARIMA
from config import (
    WINDOW_SIZE_HOURS, STEP_SIZE_HOURS,
    SEASON_MAP, MONTH_NAMES, DATA_DIR,
    WINDOW_SUMMARIES_CSV, ALL_ZONES_WINDOW_SUMMARIES_CSV, ZONES
)


def compute_hourly_average(df, zone_name):
    """
    Compute average AC power output per hour of day for a given zone.
    Used for deseasonalizing before ARMA.
    Returns Series indexed by hour (0-23).
    """
    zone_df = df[df["zone"] == zone_name].copy()
    zone_df["hour"] = pd.to_datetime(zone_df["time"]).dt.hour
    return zone_df.groupby("hour")["ac_power_kw"].mean()


def deseasonalize(df, hourly_avg):
    """
    Remove daily seasonal pattern by subtracting hour-of-day average.
    Returns DataFrame with residual column added.
    """
    df = df.copy()
    df["hour"] = pd.to_datetime(df["time"]).dt.hour
    df["hourly_avg"] = df["hour"].map(hourly_avg)
    df["residual"] = df["ac_power_kw"] - df["hourly_avg"]
    return df


def run_fft(signal, sample_spacing_hours=1):
    """
    Run FFT on a signal and return a DataFrame of periods and amplitudes.
    Only returns positive frequencies, sorted by amplitude descending.
    """
    n = len(signal)
    fft_values = fft(signal)
    frequencies = fftfreq(n, d=sample_spacing_hours)

    positive_mask = frequencies > 0
    frequencies_positive = frequencies[positive_mask]
    amplitudes = np.abs(fft_values[positive_mask])
    periods_hours = 1 / frequencies_positive

    fft_df = pd.DataFrame({
        "period_hours": periods_hours,
        "period_days": periods_hours / 24,
        "amplitude": amplitudes
    }).sort_values("amplitude", ascending=False)

    return fft_df


def run_arma(residual_series, order=(2, 0, 2)):
    """
    Fit ARMA model on deseasonalized residuals.
    Drops night hours (0-5, 20-23) before fitting.
    Returns fitted model summary.
    """
    # Drop structural zeros (night hours)
    if "hour" in residual_series.index.names:
        series = residual_series
    else:
        series = residual_series

    series = series.reset_index(drop=True)

    model = ARIMA(series, order=order)
    fitted = model.fit()
    return fitted


def compute_window_summaries(zone_df, hourly_avg, zone_name):
    """
    Generate sliding window summaries for a single zone.
    Each window = 7 days, slides forward 1 day at a time.
    Returns DataFrame of window summaries.
    """
    # Sort and reset
    zone_df = zone_df.sort_values("time").reset_index(drop=True)

    # Add residual
    zone_df["hour"] = pd.to_datetime(zone_df["time"]).dt.hour
    zone_df["hourly_avg"] = zone_df["hour"].map(hourly_avg)
    zone_df["residual"] = zone_df["ac_power_kw"] - zone_df["hourly_avg"]

    window_summaries = []

    for start_idx in range(
        0,
        len(zone_df) - WINDOW_SIZE_HOURS,
        STEP_SIZE_HOURS
    ):
        window = zone_df.iloc[start_idx: start_idx + WINDOW_SIZE_HOURS]

        window_start = window["time"].iloc[0]
        window_end = window["time"].iloc[-1]
        signal = window["ac_power_kw"].values
        residual = window["residual"].values

        # FFT on window
        n = len(signal)
        fft_vals = np.abs(fft(signal))
        freqs = fftfreq(n, d=1)

        pos_mask = freqs > 0
        pos_freqs = freqs[pos_mask]
        pos_amps = fft_vals[pos_mask]

        dominant_freq = pos_freqs[np.argmax(pos_amps)]
        dominant_period_hours = 1 / dominant_freq
        dominant_amplitude = pos_amps.max()

        # Residual statistics
        residual_mean = residual.mean()
        residual_std = residual.std()
        residual_max_dev = residual.max()
        residual_min_dev = residual.min()
        abs_max_dev = np.abs(residual).max()

        # Daytime average (hours 8-17)
        daytime_window = window[window["hour"].between(8, 17)]
        daytime_avg = daytime_window["ac_power_kw"].mean()

        # Season metadata
        window_start_dt = pd.to_datetime(window_start)
        month = window_start_dt.month
        season = SEASON_MAP.get(month, "Unknown")

        window_summaries.append({
            "window_start": window_start,
            "window_end": window_end,
            "month": month,
            "season": season,
            "dominant_period_hours": round(dominant_period_hours, 2),
            "dominant_amplitude": round(dominant_amplitude, 2),
            "residual_mean": round(residual_mean, 2),
            "residual_std": round(residual_std, 2),
            "residual_max_dev": round(residual_max_dev, 2),
            "residual_min_dev": round(residual_min_dev, 2),
            "abs_max_deviation_kw": round(abs_max_dev, 2),
            "daytime_avg_kw": round(daytime_avg, 2),
            "zone": zone_name,
        })

    summaries_df = pd.DataFrame(window_summaries)
    print(f"Generated {len(summaries_df)} window summaries for {zone_name}")
    return summaries_df


def run_time_series_analysis(
    zone_df,
    zone_name="Zone_A",
    run_fft_analysis=True,
    run_arma_analysis=True,
    save=True
):
    """
    Full time series analysis pipeline for one zone:
    1. Compute hourly averages
    2. Deseasonalize
    3. FFT analysis (optional)
    4. ARMA on residuals (optional)
    5. Sliding window summaries
    Returns (hourly_avg, summaries_df, fft_df, arma_model)
    """

    print(f"\n── Time Series Analysis: {zone_name} ──")

    # Step 1 — Hourly averages
    print("Computing hourly averages...")
    hourly_avg = compute_hourly_average(zone_df, zone_name)

    # Step 2 — Deseasonalize
    print("Deseasonalizing...")
    zone_full = zone_df[zone_df["zone"] == zone_name].copy()
    zone_full = deseasonalize(zone_full, hourly_avg)

    # Step 3 — FFT
    fft_df = None
    if run_fft_analysis:
        print("Running FFT...")
        signal = zone_full["ac_power_kw"].values
        fft_df = run_fft(signal)
        print("Top 5 dominant periods:")
        print(fft_df.head())

    # Step 4 — ARMA
    arma_model = None
    if run_arma_analysis:
        print("Fitting ARMA(2,0,2)...")
        daytime_residual = zone_full[
            zone_full["hour"].between(6, 19)
        ]["residual"].reset_index(drop=True)
        arma_model = run_arma(daytime_residual, order=(2, 0, 2))
        print(arma_model.summary())

    # Step 5 — Sliding window summaries
    print("Computing sliding window summaries...")
    summaries_df = compute_window_summaries(zone_full, hourly_avg, zone_name)

    if save:
        summaries_df.to_csv(WINDOW_SUMMARIES_CSV, index=False)
        print(f"Saved -> {WINDOW_SUMMARIES_CSV}")

    return hourly_avg, summaries_df, fft_df, arma_model


if __name__ == "__main__":
    import pandas as pd
    from config import OUTPUT_CSV

    print("Loading output dataset...")
    output_df = pd.read_csv(OUTPUT_CSV)

    all_summaries = []
    for zone_name in ZONES:
        hourly_avg, summaries_df, fft_df, arma_model = run_time_series_analysis(
            output_df,
            zone_name=zone_name,
            run_fft_analysis=(zone_name == "Zone_A"),
            run_arma_analysis=(zone_name == "Zone_A"),
            save=(zone_name == "Zone_A")
        )
        all_summaries.append(summaries_df)

    summaries_df = pd.concat(all_summaries, ignore_index=True)
    summaries_df.to_csv(ALL_ZONES_WINDOW_SUMMARIES_CSV, index=False)
    print(f"Saved -> {ALL_ZONES_WINDOW_SUMMARIES_CSV}")

    print("\nTime series analysis complete.")
    print(summaries_df.head())
