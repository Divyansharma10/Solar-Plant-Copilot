# app.py
"""
Solar Plant Copilot — Streamlit Interface
Dashboard + embedded chat interface for solar plant operators.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from config import (
    GROQ_API_KEY, CLOUD_LLM_MODEL, CLOUD_LLM_TEMPERATURE,
    SEASON_MAP, OUTPUT_CSV, WINDOW_SUMMARIES_CSV, ZONES, DATA_DIR
)
from edge_llm import format_window_for_summary, get_local_summary
from rag import setup_rag, load_embeddings, get_rag_context, build_rag_query
from pipeline import (
    build_llmsense_prompt, get_cloud_reasoning,
    run_pipeline, load_cloud_llm
)
from telemetry import fetch_live_demo, build_live_summary

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Solar Plant Copilot",
    page_icon="🌞",
    layout="wide"
)

# ── Load Data & Models (cached) ────────────────────────────────
@st.cache_resource
def load_resources():
    """Load all heavy resources once and cache them."""
    required_files = [
        OUTPUT_CSV,
        f"{DATA_DIR}/all_zones_window_summaries.csv"
    ]
    missing_files = [path for path in required_files if not os.path.exists(path)]
    if missing_files:
        raise FileNotFoundError(
            "Missing generated data: " + ", ".join(missing_files) +
            ". Run data_extraction.py, preprocessing.py, and time_series.py."
        )

    daylight_df = pd.read_csv(OUTPUT_CSV)
    summaries_df = pd.read_csv(f"{DATA_DIR}/all_zones_window_summaries.csv")
    summaries_df["month"] = pd.to_datetime(
        summaries_df["window_start"]
    ).dt.month
    summaries_df["season"] = summaries_df["month"].map(SEASON_MAP)

    embeddings = load_embeddings()
    hybrid_retriever, seasonal_lookup, _ = setup_rag(
        summaries_df,
        embeddings=embeddings,
        load_existing=True
    )
    llm = load_cloud_llm()
    return daylight_df, summaries_df, hybrid_retriever, seasonal_lookup, llm


@st.cache_data(ttl=900)
def load_live_demo(zone_name):
    """Cache live-demo telemetry briefly to limit external API traffic."""
    return fetch_live_demo(zone_name)


# ── Helper: Parse Reasoning Output ────────────────────────────
def parse_reasoning(reasoning_text):
    """Extract structured fields from LLM reasoning output."""
    fields = {
        "STATUS": "Unknown",
        "PERFORMANCE": "Unknown",
        "ANOMALY": "Unknown",
        "HISTORICAL MATCH": "Unknown",
        "RECOMMENDATION": "Unknown",
        "URGENCY": "Unknown",
        "NARRATIVE": ""
    }

    lines = reasoning_text.split("\n")
    current_field = None

    for line in lines:
        line = line.strip()
        if not line:
            continue
        for field in fields:
            if line.startswith(f"{field}:"):
                current_field = field
                fields[field] = line[len(field)+1:].strip()
                break
        else:
            if current_field == "NARRATIVE":
                fields["NARRATIVE"] += " " + line

    return fields


# ── Helper: Status Color ───────────────────────────────────────
def status_color(status):
    return {
        "Normal": "🟢",
        "Monitor": "🟡",
        "Warning": "🟠",
        "Critical": "🔴"
    }.get(status, "⚪")


# ── Helper: Output Trend Chart ─────────────────────────────────
def plot_output_trend(window_data, zone_name):
    """Plot 7-day AC power output trend using Plotly."""
    window_data = window_data.copy()
    window_data["time"] = pd.to_datetime(window_data["time"])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=window_data["time"],
        y=window_data["ac_power_kw"],
        mode="lines",
        name="AC Output (kW)",
        line=dict(color="#f59e0b", width=2)
    ))

    fig.add_trace(go.Scatter(
        x=window_data["time"],
        y=window_data["global_tilted_irradiance"],
        mode="lines",
        name="GTI (W/m²)",
        line=dict(color="#3b82f6", width=1.5, dash="dot"),
        yaxis="y2"
    ))

    fig.add_hline(
        y=1500,
        line_dash="dash",
        line_color="red",
        annotation_text="Grid Min (1500 kW)"
    )

    fig.update_layout(
        title=f"{zone_name} — 7-Day Output Trend",
        xaxis_title="Time",
        yaxis_title="AC Power (kW)",
        yaxis2=dict(
            title="GTI (W/m²)",
            overlaying="y",
            side="right"
        ),
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="white")
    )

    return fig


# ── Chat Response ──────────────────────────────────────────────
def get_chat_response(
    user_question,
    current_reasoning,
    window_stats,
    zone_name,
    llm
):
    """
    Answer operator's conversational question using current
    window context + existing Copilot reasoning as background.
    """
    context_prompt = f"""You are a Solar Plant Operations Copilot assistant.
The operator is asking a question about the current plant status.

Current Plant Analysis:
Zone: {zone_name}
Window: {window_stats.get('window_start')} to {window_stats.get('window_end')}
Season: {window_stats.get('season')}
Actual Output: {window_stats.get('daytime_avg_kw')} kW

Recent Copilot Analysis:
{current_reasoning}

Operator Question: {user_question}

Answer the operator's question directly and concisely using the context above.
Use exact numbers where relevant. Keep the response under 4 sentences."""

    messages = [HumanMessage(content=context_prompt)]
    response = llm.invoke(messages, max_tokens=300)
    return response.content


# ── Main App ───────────────────────────────────────────────────
def main():

    # Header
    st.title("🌞 Solar Plant Copilot")
    st.caption("Jaipur, Rajasthan, India — LLMSense + RAG Powered")

    # Load resources
    try:
        with st.spinner("Loading models and data..."):
            daylight_df, summaries_df, hybrid_retriever, seasonal_lookup, llm = \
                load_resources()
    except FileNotFoundError as exc:
        st.error(f"Data setup incomplete: {exc}")
        st.code(
            "python data_extraction.py\n"
            "python preprocessing.py\n"
            "python time_series.py\n"
            "python rag.py"
        )
        st.stop()
    except Exception as exc:
        st.error(
            "Resource initialization failed. Check the generated data and "
            f"embedding model, then restart the app. ({type(exc).__name__}: {exc})"
        )
        st.stop()

    # ── Sidebar ────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Controls")

        data_mode = st.radio(
            "Data Source",
            options=["Historical", "Live Demo"],
            index=0,
            help="Live Demo uses current Open-Meteo weather and simulated output."
        )

        # Zone selector
        zone_name = st.selectbox(
            "Select Zone",
            options=list(ZONES.keys()),
            index=0
        )

        if data_mode == "Historical":
            # Window indexes are zone-relative.
            zone_summaries = summaries_df[
                summaries_df["zone"] == zone_name
            ].reset_index(drop=True)

            if zone_summaries.empty:
                st.error(f"No window summaries found for {zone_name}.")
                st.stop()

            window_mode = st.radio(
                "Window Mode",
                options=["Latest (most recent 7 days)", "Historical"],
                index=0
            )

            if window_mode == "Historical":
                window_index = st.slider(
                    "Window Index",
                    min_value=0,
                    max_value=len(zone_summaries) - 1,
                    value=len(zone_summaries) - 1,
                    help="0 = Jan 2020, latest = Dec 2024"
                )
                selected_window = zone_summaries.iloc[window_index]
                st.caption(
                    f"📅 {selected_window['window_start'][:10]} → "
                    f"{selected_window['window_end'][:10]}"
                )
            else:
                window_index = len(zone_summaries) - 1
            button_label = "🔍 Run Analysis"
        else:
            zone_summaries = None
            window_index = 0
            st.caption(
                "🟢 Read-only demo feed · current weather · simulated plant output"
            )
            button_label = "🔄 Refresh Live Analysis"

        # Run button
        run_analysis = st.button(
            button_label,
            type="primary",
            use_container_width=True
        )

        st.divider()
        st.caption("Solar Plant Copilot v1.0")
        st.caption("Built with LLMSense + RAG")

    # ── Initialize session state ───────────────────────────────
    if "result" not in st.session_state:
        st.session_state.result = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "last_zone" not in st.session_state:
        st.session_state.last_zone = None
    if "last_window" not in st.session_state:
        st.session_state.last_window = None
    if "last_mode" not in st.session_state:
        st.session_state.last_mode = None

    # ── Run Pipeline ───────────────────────────────────────────
    should_run = (
        run_analysis or st.session_state.result is None or
        st.session_state.last_zone != zone_name or
        st.session_state.last_mode != data_mode
    )
    if should_run:
        try:
            with st.spinner("Running LLMSense pipeline..."):
                if data_mode == "Live Demo":
                    if run_analysis:
                        load_live_demo.clear()
                    analysis_df, telemetry_quality = load_live_demo(zone_name)
                    analysis_summaries = build_live_summary(
                        analysis_df, daylight_df, zone_name
                    )
                    result = run_pipeline(
                        zone_name=zone_name,
                        window_index=0,
                        summaries_df=analysis_summaries,
                        daylight_df=analysis_df,
                        hybrid_retriever=hybrid_retriever,
                        seasonal_lookup=seasonal_lookup,
                        llm=llm,
                        historical_df=daylight_df
                    )
                    result["telemetry_quality"] = telemetry_quality
                    result["telemetry_data"] = analysis_df
                else:
                    result = run_pipeline(
                        zone_name=zone_name,
                        window_index=window_index,
                        summaries_df=zone_summaries,
                        daylight_df=daylight_df,
                        hybrid_retriever=hybrid_retriever,
                        seasonal_lookup=seasonal_lookup,
                        llm=llm
                    )
                if result:
                    result["data_mode"] = data_mode
                    st.session_state.result = result
                    st.session_state.last_zone = zone_name
                    st.session_state.last_window = window_index
                    st.session_state.last_mode = data_mode
                    st.session_state.chat_history = []
        except Exception as exc:
            st.error(
                "Live telemetry could not be prepared. Historical mode remains "
                f"available. ({type(exc).__name__}: {exc})"
            )
            if st.session_state.result is None:
                return

    result = st.session_state.result

    if result is None:
        st.warning("No analysis yet. Click Run Analysis.")
        return

    for warning in result.get("warnings", []):
        st.warning(warning)

    quality = result.get("telemetry_quality")
    if quality:
        freshness = (
            f"Fresh · {quality['age_hours']:.1f}h old"
            if quality["fresh"] else
            f"Stale · {quality['age_hours']:.1f}h old"
        )
        if quality["valid"] and quality["fresh"]:
            st.success(
                f"Live demo telemetry: {freshness} · "
                f"latest {quality['latest_timestamp']}"
            )
        else:
            st.warning(
                f"Live demo telemetry quality: {freshness}. " +
                ("; ".join(quality["errors"]) or "No schema errors.")
            )

    # Parse reasoning into structured fields
    fields = parse_reasoning(result["reasoning"])
    status = fields["STATUS"].split("—")[0].strip().split()[0]

    # ── TOP — Dashboard Metrics ────────────────────────────────
    st.subheader("📊 Plant Status Dashboard")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Status",
            f"{status_color(status)} {status}"
        )
    with col2:
        st.metric(
            "Actual Output",
            f"{result['actual_output_kw']:.0f} kW"
        )
    with col3:
        perf = round(
            (result['actual_output_kw'] / result['expected_output_kw']) * 100,
            1
        )
        st.metric(
            "Performance",
            f"{perf}%",
            delta=f"{perf - 100:.1f}% vs expected"
        )
    with col4:
        st.metric(
            "Season",
            result["season"]
        )
    with col5:
        st.metric(
            "Zone",
            result["zone"]
        )

    # Output trend chart
    window_stats = {
        "window_start": result["window_start"],
        "window_end": result["window_end"],
        "daytime_avg_kw": result["actual_output_kw"],
        "season": result["season"]
    }
    window_start = pd.to_datetime(result["window_start"])
    window_end = pd.to_datetime(result["window_end"])

    if result.get("data_mode") == "Live Demo":
        zone_data = result["telemetry_data"].copy()
    else:
        zone_data = daylight_df[daylight_df["zone"] == result["zone"]].copy()
    zone_data["time"] = pd.to_datetime(zone_data["time"])
    window_data = zone_data[
        (zone_data["time"] >= window_start) &
        (zone_data["time"] <= window_end)
    ]

    if not window_data.empty:
        st.plotly_chart(
            plot_output_trend(window_data, result["zone"]),
            use_container_width=True
        )

    st.divider()

    # ── Deterministic anomaly evidence ──────────────────────────
    anomaly = result.get("anomaly_analysis")
    if anomaly:
        st.subheader("🔬 Deterministic Health Checks")
        health1, health2, health3, health4 = st.columns(4)
        with health1:
            st.metric("Performance Ratio", f"{anomaly['performance_ratio_pct']:.1f}%")
        with health2:
            st.metric("Physical Model", f"{anomaly['model_performance_pct']:.1f}%")
        with health3:
            st.metric("Rolling Baseline", f"{anomaly['rolling_baseline_pct']:.1f}%")
        with health4:
            st.metric("Likely Cause", anomaly["cause"])

        for evidence in anomaly["evidence"]:
            st.caption(f"• {evidence}")

        st.divider()

    # ── Auditable operator decision rules ───────────────────────
    decision = result.get("recommendation_analysis")
    if decision:
        st.subheader("🧭 Operator Decision Rules")
        primary_message = (
            f"{decision['primary_rule_id']} · {decision['severity']} · "
            f"{decision['action']} Urgency: {decision['urgency']}."
        )
        if decision["severity"] == "Critical":
            st.error(primary_message)
        elif decision["severity"] == "Warning":
            st.warning(primary_message)
        elif decision["severity"] == "Monitor":
            st.info(primary_message)
        else:
            st.success(primary_message)

        with st.expander("View triggered rules and clearing conditions"):
            rule_rows = [{
                "Rule": rule["rule_id"],
                "Severity": rule["severity"],
                "Evidence": rule["evidence"],
                "Action": rule["action"],
                "Clear condition": rule["clear_condition"]
            } for rule in decision["triggered_rules"]]
            st.dataframe(rule_rows, hide_index=True, width="stretch")

        st.divider()

    # ── MIDDLE — Copilot Analysis ──────────────────────────────
    st.subheader("🤖 Copilot Analysis")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown(f"**STATUS:** {status_color(status)} {fields['STATUS']}")
        st.markdown(f"**PERFORMANCE:** {fields['PERFORMANCE']}")
        st.markdown(f"**ANOMALY:** {fields['ANOMALY']}")

    with col_right:
        st.markdown(f"**HISTORICAL MATCH:** {fields['HISTORICAL MATCH']}")
        st.markdown(f"**RECOMMENDATION:** {fields['RECOMMENDATION']}")
        st.markdown(f"**URGENCY:** {fields['URGENCY']}")

    # Narrative
    st.info(f"📋 {fields['NARRATIVE'].strip()}")

    # Edge LLM summary expander
    with st.expander("🔍 Edge LLM Summary (Phi-4-Mini)"):
        st.write(result["local_summary"])

    with st.expander("📚 RAG Historical Context"):
        st.write(result["rag_context"])

    st.divider()

    # ── BOTTOM — Chat Interface ────────────────────────────────
    st.subheader("💬 Ask the Copilot")

    if llm is None:
        st.info("Chat is disabled until GROQ_API_KEY is configured.")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    user_input = st.chat_input(
        "Ask about plant performance, anomalies, recommendations...",
        disabled=llm is None
    )

    if user_input:
        # Add user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.write(user_input)

        # Get Copilot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = get_chat_response(
                        user_question=user_input,
                        current_reasoning=result["reasoning"],
                        window_stats={
                            **window_stats,
                            "season": result["season"]
                        },
                        zone_name=result["zone"],
                        llm=llm
                    )
                except Exception as exc:
                    response = (
                        "Chat is temporarily unavailable. The dashboard "
                        f"analysis remains valid. ({type(exc).__name__})"
                    )
            st.write(response)

        # Add assistant response to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response
        })


if __name__ == "__main__":
    main()
