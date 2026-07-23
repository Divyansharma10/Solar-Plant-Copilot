# Solar Plant Copilot

Solar Plant Copilot is a Streamlit dashboard for solar plant operators. It combines:

- weather and solar feature extraction
- PV output simulation
- sliding-window time-series analysis
- local edge summarization with Ollama
- hybrid RAG retrieval with FAISS + BM25
- cloud reasoning with Groq-hosted LLMs

The app is designed to help operators review a zone's recent performance, compare it against historical patterns, and get concise action-oriented guidance.

## Project Structure

- `app.py` - Streamlit UI and top-level app entry point
- `pipeline.py` - Orchestrates the edge summary, retrieval, and cloud reasoning
- `data_extraction.py` - Downloads historical weather data from Open-Meteo
- `preprocessing.py` - Adds solar geometry and simulates plant output
- `time_series.py` - Builds sliding window statistics and residual features
- `edge_llm.py` - Prepares and summarizes window data with a local model
- `rag.py` - Builds and queries the hybrid retriever
- `location_manager.py` - Experimental first-run location bootstrap flow
- `config.py` - Central constants, thresholds, paths, and model settings
- `main.ipynb` - Notebook for exploration and experimentation
- `data/` - Generated CSVs and the FAISS index

## How It Works

1. `data_extraction.py` pulls hourly weather and solar data for each zone from Open-Meteo.
2. `preprocessing.py` computes solar zenith/elevation, filters daylight rows, and simulates AC output.
3. `time_series.py` creates 7-day sliding window summaries.
4. `rag.py` turns window summaries into documents and builds a FAISS + BM25 retriever.
5. `edge_llm.py` compresses the current window into a short local summary with Ollama.
6. `pipeline.py` combines the edge summary, retrieved history, and current metrics into a final reasoning prompt.
7. `app.py` renders the result in Streamlit and supports a simple operator chat experience.

## Prerequisites

- Python 3.10 or newer
- An active Groq API key
- Ollama installed locally if you want to run the edge summarizer
- The generated CSV files under `data/`

## Setup

Create and activate a virtual environment, then install the required packages.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you already have the environment set up, install only the missing packages.

## Environment Variables

Create a `.env` file in the project root with at least:

```env
GROQ_API_KEY=your_groq_api_key
```

Optional values can be added if you want to customize model behavior, but the app reads its main settings from `config.py`.

## Data Files

The Streamlit app expects the generated datasets in `data/`:

- `all_zones_weather_combined.csv`
- `all_zones_weather_with_zenith.csv`
- `all_zones_daylight_only.csv`
- `all_zones_with_output.csv`
- `all_zones_window_summaries.csv`
- `faiss_index/index.faiss`

Some zone-specific CSVs are also present for inspection and debugging.

## Running the Pipeline

If you want to rebuild the data products from scratch, run the scripts in this order:

```powershell
python data_extraction.py
python preprocessing.py
python time_series.py
python rag.py
```

That produces the CSV files and retrieval index used by the app.

## Running the App

Start the Streamlit dashboard with:

```powershell
streamlit run app.py
```

Then choose a zone, choose a window, and click Run Analysis.

## Data Modes

The sidebar supports two read-only data modes:

- **Historical** uses the generated 2020-2024 CSV files and saved windows.
- **Live Demo** fetches recent hourly weather from Open-Meteo, simulates zone
  output with the existing PV model, validates freshness and schema quality,
  and runs the same anomaly, RAG, and reasoning pipeline.

Live Demo is not actual plant telemetry. It is an integration test for the
vendor-neutral telemetry contract in `telemetry.py`. A future SCADA, MQTT,
OPC-UA, database, or inverter adapter should produce the same core fields:

- `time`, `zone`, `ac_power_kw`, `dc_power_kw`
- `global_tilted_irradiance`, `temperature_2m`, `cell_temperature`
- `inverter_id`, `inverter_status`, `grid_available`
- optional `battery_soc_pct` and `alarm_code`

The live adapter retries transient API failures, rejects invalid schemas and
out-of-range values, reports feed freshness, and retains Historical mode when
the external feed is unavailable.

## Operator Decision Rules

`recommendation_engine.py` converts deterministic anomaly evidence into
auditable operator decisions. Every triggered rule includes a stable rule ID,
severity, measured evidence, required action, urgency, and clearing condition.
The cloud LLM may explain these decisions but is instructed not to replace the
primary rule's action or urgency.

Current rule families cover data completeness, stuck sensors, physical
underperformance, strong-sun output deficits, inverter clipping, weather-driven
reductions, and sustained grid-minimum risk. Run the durable rule tests with:

```bash
python -m unittest tests/test_recommendation_engine.py
```

## Notes

- The edge summarizer in `edge_llm.py` uses `ollama.chat`, so Ollama must be running locally for that step to work.
- The cloud reasoning step in `pipeline.py` uses Groq's ChatGroq integration.
- `location_manager.py` contains a newer location-bootstrap flow, but it is more experimental than the main app path.
- The notebook is useful for exploration, but the Streamlit app is the primary user-facing entry point.

## Suggested Workflow

1. Generate or refresh the data under `data/`.
2. Confirm your `.env` file contains `GROQ_API_KEY`.
3. Make sure Ollama is available if you want the edge summary step.
4. Launch `streamlit run app.py`.
