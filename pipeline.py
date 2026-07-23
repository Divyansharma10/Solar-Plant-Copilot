# pipeline.py
"""
Solar Plant Copilot — Main Pipeline
Orchestrates the full LLMSense + RAG pipeline:
Edge LLM (Phi-4-Mini) -> RAG retrieval -> Cloud LLM (Llama 3.3 70B)
"""

import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from config import (
    GROQ_API_KEY, CLOUD_LLM_MODEL, CLOUD_LLM_TEMPERATURE,
    SEASON_MAP, GRID_CONTRACT_MIN_KW,
    BATTERY_RESERVE_THRESHOLD_KW, BATTERY_RESERVE_DURATION_HOURS,
    NORMAL_PR_MIN, NORMAL_PR_MAX,
    PERFORMANCE_NORMAL, PERFORMANCE_MONITOR, PERFORMANCE_WARNING,
    DATA_DIR
)
from edge_llm import format_window_for_summary, get_local_summary
from rag import get_rag_context, build_rag_query
from anomaly_detection import analyze_window, format_anomaly_context
from recommendation_engine import (
    evaluate_recommendations, format_recommendation_context
)


# ── Cloud LLM Setup ────────────────────────────────────────────
def load_cloud_llm():
    """Initialize ChatGroq, or return None when cloud mode is unconfigured."""
    if not GROQ_API_KEY:
        print("Cloud LLM disabled: GROQ_API_KEY is not configured.")
        return None

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=CLOUD_LLM_MODEL,
        temperature=CLOUD_LLM_TEMPERATURE,
        timeout=30,
        max_retries=2
    )
    print(f"Cloud LLM loaded: {CLOUD_LLM_MODEL}")
    return llm


# ── Prompt Building ────────────────────────────────────────────
def build_llmsense_prompt(
    local_summary,
    window_stats,
    zone_name,
    rag_context,
    expected_output_kw,
    anomaly_context="",
    recommendation_context=""
):
    """
    Build the full LLMSense prompt following:
    Prompt = Objective + Context + Data + Historical Context + Format

    Parameters:
        local_summary      — Phi-4-Mini edge summary text
        window_stats       — dict from summaries_df row
        zone_name          — e.g. "Zone_A"
        rag_context        — retrieved historical context string
        expected_output_kw — seasonal expected output from lookup
    """
    actual_output = window_stats["daytime_avg_kw"]
    performance_pct = round((actual_output / expected_output_kw) * 100, 1)
    season = window_stats.get("season", "Unknown")

    # ── OBJECTIVE ──────────────────────────────────────────────
    objective = """You are a Solar Plant Operations Copilot.
Your job is to assess current plant performance, identify anomalies,
and recommend specific operator actions based on sensor data.
Be precise, factual, and actionable. Only use information provided."""

    # ── CONTEXT ────────────────────────────────────────────────
    context = f"""
Plant Information:
- Location: Jaipur, Rajasthan, India (Latitude 26.9°N)
- Zone: {zone_name}
- Nameplate Capacity: 5000 kW (5 MW) per zone
- Panel Type: Fixed-tilt crystalline silicon, tilt angle 24°, south-facing
- Grid Contract Minimum: {GRID_CONTRACT_MIN_KW} kW during daylight hours
- Battery Reserve Threshold: Activate if output expected below \
{BATTERY_RESERVE_THRESHOLD_KW} kW for >{BATTERY_RESERVE_DURATION_HOURS} hours
- Normal Performance Ratio: {NORMAL_PR_MIN * 100:.0f}% - {NORMAL_PR_MAX * 100:.0f}%
- Season context: Jaipur receives peak irradiance in March-April,
  lowest in December-January, monsoon cloud disruption June-September
"""

    # ── CURRENT DATA ───────────────────────────────────────────
    data = f"""
Current Window Data:
Period: {window_stats['window_start']} to {window_stats['window_end']}
Season: {season}

Edge LLM Summary (Phi-4-Mini):
{local_summary}

Statistical Summary (7-day sliding window):
- Dominant cycle period: {window_stats['dominant_period_hours']} hours
- Actual daytime average output: {actual_output} kW
- Seasonal expected output (5-year average for this month): \
{expected_output_kw} kW
- Performance vs seasonal expectation: {performance_pct}%
- Residual std deviation: {window_stats['residual_std']} kW
- Maximum single-hour deviation: {window_stats['abs_max_deviation_kw']} kW
- Residual mean vs seasonal norm: {window_stats['residual_mean']} kW

Deterministic Anomaly Engine (treat this measured evidence as authoritative):
{anomaly_context}

Deterministic Operator Rules (authoritative):
{recommendation_context}
Do not invent a different action or urgency. The RECOMMENDATION and URGENCY
fields must preserve the primary rule's action and urgency exactly.

Performance Assessment Guide:
- Above {PERFORMANCE_NORMAL}%: Normal — plant performing well
- {PERFORMANCE_MONITOR}-{PERFORMANCE_NORMAL}%: Monitor — slight underperformance
- {PERFORMANCE_WARNING}-{PERFORMANCE_MONITOR}%: Warning — investigate cause
- Below {PERFORMANCE_WARNING}%: Critical — immediate action required
"""

    # ── HISTORICAL CONTEXT (RAG) ────────────────────────────────
    historical = f"""
Historical Context (retrieved from 5-year pattern database):
{rag_context}
"""

    # ── FORMAT ─────────────────────────────────────────────────
    output_format = """
Respond in exactly this structure:

STATUS: [Normal / Monitor / Warning / Critical]
PERFORMANCE: [actual] kW vs [expected] kW = [performance]%
ANOMALY: [None detected / describe anomaly with exact numbers]
HISTORICAL MATCH: [Most similar historical pattern and what happened next]
RECOMMENDATION: [Specific operator action, or "No action required"]
URGENCY: [Immediate / Within 1 hour / Monitor / None]

NARRATIVE:
[One paragraph: what is happening, why, what happened in similar past
situations, what the operator should do now.
Use exact numbers. Maximum 5 sentences.]
"""

    return f"{objective}\n{context}\n{data}\n{historical}\n{output_format}"


# ── Cloud Reasoning ────────────────────────────────────────────
def get_cloud_reasoning(prompt, llm, max_tokens=600):
    """
    Send the full LLMSense prompt to the cloud LLM for reasoning.
    Returns response string.
    """
    from langchain_core.messages import HumanMessage
    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages, max_tokens=max_tokens)
    return response.content


def build_fallback_reasoning(
    window_stats, expected_output_kw, recommendation_analysis=None
):
    """Create an auditable rules-only assessment when cloud AI is unavailable."""
    actual = float(window_stats["daytime_avg_kw"])
    expected = float(expected_output_kw)
    performance = round((actual / expected) * 100, 1) if expected else 0.0

    if performance >= PERFORMANCE_NORMAL:
        status, urgency = "Normal", "None"
        anomaly = "None detected"
        recommendation = "No action required"
    elif performance >= PERFORMANCE_MONITOR:
        status, urgency = "Monitor", "Monitor"
        anomaly = f"Output is {expected - actual:.1f} kW below expectation"
        recommendation = "Continue monitoring; investigate if performance declines"
    elif performance >= PERFORMANCE_WARNING:
        status, urgency = "Warning", "Within 1 hour"
        anomaly = f"Output is {expected - actual:.1f} kW below expectation"
        recommendation = "Check irradiance sensors, inverter status, and grid availability"
    else:
        status, urgency = "Critical", "Immediate"
        anomaly = f"Output is {expected - actual:.1f} kW below expectation"
        recommendation = "Inspect inverter alarms, plant availability, and grid connection"

    if recommendation_analysis:
        recommendation = recommendation_analysis["action"]
        urgency = recommendation_analysis["urgency"]

    return f"""STATUS: {status}
PERFORMANCE: {actual:.1f} kW vs {expected:.1f} kW = {performance:.1f}%
ANOMALY: {anomaly}
HISTORICAL MATCH: Unavailable in rules-only mode
RECOMMENDATION: {recommendation}
URGENCY: {urgency}

NARRATIVE:
Rules-only assessment: the plant is producing {actual:.1f} kW against an expected {expected:.1f} kW ({performance:.1f}%). {recommendation}."""


# ── Full Pipeline ──────────────────────────────────────────────
def run_pipeline(
    zone_name,
    window_index,
    summaries_df,
    daylight_df,
    hybrid_retriever,
    seasonal_lookup,
    llm,
    historical_df=None
):
    """
    Full LLMSense + RAG pipeline for a given zone and window index.

    Steps:
    1. Get window statistics
    2. Get raw daylight data for window
    3. Pre-aggregate → Phi-4-Mini edge summary
    4. RAG retrieval
    5. Build LLMSense prompt
    6. Cloud LLM reasoning
    7. Return structured result

    Parameters:
        zone_name         — e.g. "Zone_A"
        window_index      — index into summaries_df
        summaries_df      — sliding window summaries DataFrame
        daylight_df       — daylight-filtered plant output DataFrame
        hybrid_retriever  — BM25 + FAISS hybrid retriever
        seasonal_lookup   — dict {month: expected_output_kw}
        llm               — ChatGroq LangChain LLM instance

    Returns dict with all intermediate and final outputs.
    """

    # Step 1 — Window stats. Filter defensively so callers cannot combine one
    # zone's raw telemetry with another zone's summary row.
    if "zone" in summaries_df.columns:
        zone_summaries = summaries_df[
            summaries_df["zone"] == zone_name
        ].reset_index(drop=True)
    else:
        zone_summaries = summaries_df.reset_index(drop=True)

    if zone_summaries.empty:
        raise ValueError(f"No window summaries found for {zone_name}")
    if not 0 <= window_index < len(zone_summaries):
        raise IndexError(
            f"Window index {window_index} is out of range for {zone_name} "
            f"(0-{len(zone_summaries) - 1})"
        )

    window_stats = zone_summaries.iloc[window_index].to_dict()
    window_start = pd.to_datetime(window_stats["window_start"])
    window_end = pd.to_datetime(window_stats["window_end"])
    month = int(window_start.month)
    season = SEASON_MAP.get(month, "Unknown")
    window_stats["season"] = season

    # Step 2 — Raw data for this window
    zone_data = daylight_df[daylight_df["zone"] == zone_name].copy()
    zone_data["time"] = pd.to_datetime(zone_data["time"])
    window_data = zone_data[
        (zone_data["time"] >= window_start) &
        (zone_data["time"] <= window_end)
    ]

    if len(window_data) == 0:
        print(f"No data found for {zone_name}, window {window_index}")
        return None

    if historical_df is not None:
        anomaly_history = historical_df[
            historical_df["zone"] == zone_name
        ].copy()
    else:
        anomaly_history = zone_data
    anomaly_analysis = analyze_window(window_data, anomaly_history)
    recommendation_analysis = evaluate_recommendations(
        anomaly_analysis, window_data
    )

    warnings = []

    # Step 3 — Edge LLM summarization
    structured_text = format_window_for_summary(window_data)
    try:
        local_summary = get_local_summary(structured_text)
    except Exception as exc:
        local_summary = structured_text
        warnings.append(
            "Ollama is unavailable; using the deterministic sensor summary "
            f"instead ({type(exc).__name__})."
        )

    # Step 4 — RAG retrieval
    rag_query = build_rag_query(window_stats, month)
    try:
        rag_context = get_rag_context(rag_query, hybrid_retriever)
    except Exception as exc:
        rag_context = "Historical retrieval unavailable."
        warnings.append(
            "Historical retrieval failed; continuing without RAG context "
            f"({type(exc).__name__})."
        )

    # Step 5 — Seasonal expected output
    expected_output_kw = seasonal_lookup.get(month, 2000.0)

    # Step 6 — Build prompt
    prompt = build_llmsense_prompt(
        local_summary=local_summary,
        window_stats=window_stats,
        zone_name=zone_name,
        rag_context=rag_context,
        expected_output_kw=expected_output_kw,
        anomaly_context=format_anomaly_context(anomaly_analysis),
        recommendation_context=format_recommendation_context(
            recommendation_analysis
        )
    )

    # Step 7 — Cloud LLM reasoning, with a deterministic operational fallback.
    if llm is None:
        reasoning = build_fallback_reasoning(
            window_stats, expected_output_kw, recommendation_analysis
        )
        warnings.append(
            "Groq is not configured; showing a deterministic rules-only assessment."
        )
    else:
        try:
            reasoning = get_cloud_reasoning(prompt, llm)
        except Exception as exc:
            reasoning = build_fallback_reasoning(
                window_stats, expected_output_kw, recommendation_analysis
            )
            warnings.append(
                "Groq request failed; showing a deterministic rules-only "
                f"assessment ({type(exc).__name__})."
            )

    return {
        "zone": zone_name,
        "window_start": str(window_stats["window_start"]),
        "window_end": str(window_stats["window_end"]),
        "month": month,
        "season": season,
        "actual_output_kw": window_stats["daytime_avg_kw"],
        "expected_output_kw": expected_output_kw,
        "structured_text": structured_text,
        "local_summary": local_summary,
        "rag_query": rag_query,
        "rag_context": rag_context,
        "prompt": prompt,
        "reasoning": reasoning,
        "warnings": warnings,
        "reasoning_mode": "rules" if any("Groq" in w for w in warnings) else "ai",
        "anomaly_analysis": anomaly_analysis,
        "recommendation_analysis": recommendation_analysis
    }


def print_result(result):
    """Pretty print a pipeline result."""
    print("=" * 60)
    print(f"Zone: {result['zone']} | Season: {result['season']}")
    print(f"Window: {result['window_start']} → {result['window_end']}")
    print(
        f"Actual: {result['actual_output_kw']} kW | "
        f"Expected: {result['expected_output_kw']} kW"
    )
    print()
    print("── Edge Summary (Phi-4-Mini) ──")
    print(result["local_summary"])
    print()
    print("── Cloud Reasoning (Llama 3.3 70B + RAG) ──")
    print(result["reasoning"])
    print("=" * 60)


if __name__ == "__main__":
    import pandas as pd
    from config import OUTPUT_CSV, WINDOW_SUMMARIES_CSV
    from rag import setup_rag, load_embeddings

    print("Loading data...")
    daylight_df = pd.read_csv(OUTPUT_CSV)
    summaries_df = pd.read_csv(WINDOW_SUMMARIES_CSV)

    print("Setting up RAG...")
    embeddings = load_embeddings()
    hybrid_retriever, seasonal_lookup, _ = setup_rag(
        summaries_df,
        embeddings=embeddings,
        load_existing=True
    )

    print("Loading cloud LLM...")
    llm = load_cloud_llm()

    # Test on 3 windows
    for idx in [0, 90, 180]:
        result = run_pipeline(
            zone_name="Zone_A",
            window_index=idx,
            summaries_df=summaries_df,
            daylight_df=daylight_df,
            hybrid_retriever=hybrid_retriever,
            seasonal_lookup=seasonal_lookup,
            llm=llm
        )
        if result:
            print_result(result)
            print()
