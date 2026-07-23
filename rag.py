# rag.py
"""
Solar Plant Copilot — RAG Layer
Handles document creation, FAISS indexing, BM25,
hybrid retrieval, and historical context formatting.
"""

import pandas as pd
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from config import (
    EMBEDDING_MODEL, BM25_WEIGHT, FAISS_WEIGHT,
    RAG_TOP_K, SEASON_MAP, MONTH_NAMES,
    FAISS_INDEX_PATH, DATA_DIR
)


# ── Embeddings ─────────────────────────────────────────────────
def load_embeddings():
    """Load HuggingFace sentence transformer embeddings."""
    print(f"Loading embeddings: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    print("Embeddings loaded.")
    return embeddings


# ── Document Creation ──────────────────────────────────────────
def window_row_to_document(row):
    """
    Convert a single window summary row into a LangChain Document.
    Includes rich metadata for filtering and retrieval.
    """
    window_start = pd.to_datetime(row["window_start"])
    month = window_start.month
    season = SEASON_MAP.get(month, "Unknown")

    content = (
        f"Window: {row['window_start']} to {row['window_end']}. "
        f"Season: {season}. "
        f"Dominant cycle: {row['dominant_period_hours']} hours. "
        f"Daytime average output: {row['daytime_avg_kw']} kW. "
        f"Residual std deviation: {row['residual_std']} kW. "
        f"Max deviation from typical: {row['abs_max_deviation_kw']} kW. "
        f"Residual mean vs seasonal norm: {row['residual_mean']} kW."
    )

    metadata = {
        "type": "window_summary",
        "window_start": str(row["window_start"]),
        "window_end": str(row["window_end"]),
        "month": month,
        "season": season,
        "daytime_avg_kw": float(row["daytime_avg_kw"]),
        "residual_std": float(row["residual_std"]),
        "abs_max_deviation_kw": float(row["abs_max_deviation_kw"])
    }

    return Document(page_content=content, metadata=metadata)


def create_window_documents(summaries_df):
    """
    Convert all window summaries into LangChain Documents.
    Returns list of Documents.
    """
    docs = [
        window_row_to_document(row)
        for _, row in summaries_df.iterrows()
    ]
    print(f"Created {len(docs)} window summary documents.")
    return docs


def create_seasonal_documents(summaries_df):
    """
    Generate one seasonal baseline document per month
    from 5-year historical window summary statistics.
    Returns list of 12 Documents.
    """
    seasonal_docs = []

    summaries_df = summaries_df.copy()
    summaries_df["month"] = pd.to_datetime(
        summaries_df["window_start"]
    ).dt.month

    for month_num in range(1, 13):
        month_data = summaries_df[summaries_df["month"] == month_num]

        if len(month_data) == 0:
            continue

        avg_output = month_data["daytime_avg_kw"].mean()
        std_output = month_data["residual_std"].mean()
        avg_max_dev = month_data["abs_max_deviation_kw"].mean()
        min_output = month_data["daytime_avg_kw"].min()
        max_output = month_data["daytime_avg_kw"].max()
        season = SEASON_MAP.get(month_num, "Unknown")
        month_name = MONTH_NAMES.get(month_num, str(month_num))

        content = (
            f"Seasonal Baseline for {month_name} ({season} season). "
            f"Based on 5 years of historical data (2020-2024). "
            f"Typical daytime average output: {avg_output:.1f} kW. "
            f"Output range: {min_output:.1f} to {max_output:.1f} kW. "
            f"Typical residual std deviation: {std_output:.1f} kW. "
            f"Typical max deviation from norm: {avg_max_dev:.1f} kW. "
            f"Plant nameplate capacity: 5000 kW per zone. "
            f"Grid contract minimum: 1500 kW during daylight hours."
        )

        metadata = {
            "type": "seasonal_baseline",
            "month": month_num,
            "month_name": month_name,
            "season": season,
            "avg_daytime_output_kw": round(avg_output, 1),
            "min_output_kw": round(min_output, 1),
            "max_output_kw": round(max_output, 1)
        }

        seasonal_docs.append(
            Document(page_content=content, metadata=metadata)
        )

    print(f"Created {len(seasonal_docs)} seasonal baseline documents.")
    return seasonal_docs


def build_seasonal_lookup(seasonal_docs):
    """
    Build a month -> expected output kW lookup dictionary
    from seasonal baseline documents.
    Returns dict {month_int: avg_output_kw}.
    """
    return {
        doc.metadata["month"]: doc.metadata["avg_daytime_output_kw"]
        for doc in seasonal_docs
    }


# ── Index Building ─────────────────────────────────────────────
def build_index(all_docs, embeddings, save=True):
    """
    Build FAISS vector store and BM25 retriever from documents.
    Optionally saves FAISS index to disk.
    Returns (vectorstore, bm25_retriever, hybrid_retriever).
    """
    print(f"Building FAISS index from {len(all_docs)} documents...")
    vectorstore = FAISS.from_documents(all_docs, embeddings)
    print("FAISS index built.")

    if save:
        vectorstore.save_local(FAISS_INDEX_PATH)
        print(f"FAISS index saved -> {FAISS_INDEX_PATH}")

    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = RAG_TOP_K

    faiss_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RAG_TOP_K}
    )

    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[BM25_WEIGHT, FAISS_WEIGHT]
    )

    print("Hybrid retriever ready.")
    return vectorstore, bm25_retriever, hybrid_retriever


def load_index(embeddings):
    """
    Load a previously saved FAISS index from disk.
    Returns vectorstore.
    """
    print(f"Loading FAISS index from {FAISS_INDEX_PATH}...")
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print(f"Loaded {vectorstore.index.ntotal} vectors.")
    return vectorstore


# ── Retrieval ──────────────────────────────────────────────────
def get_rag_context(query, retriever, top_k=RAG_TOP_K):
    """
    Retrieve relevant historical context for a given query.
    Returns formatted string ready to inject into LLMSense prompt.
    """
    docs = retriever.invoke(query)
    docs = docs[:top_k]

    formatted = []
    for i, doc in enumerate(docs):
        doc_type = doc.metadata.get("type", "unknown")
        season = doc.metadata.get("season", "unknown")
        formatted.append(
            f"[{i+1}] ({doc_type} | {season})\n{doc.page_content}"
        )

    return "\n\n".join(formatted)


def build_rag_query(window_stats, month):
    """
    Build a RAG retrieval query from current window statistics.
    Returns query string.
    """
    season = SEASON_MAP.get(month, "Unknown")
    return (
        f"{season} season month {month} "
        f"daytime output {window_stats['daytime_avg_kw']:.0f} kW "
        f"deviation {window_stats['residual_mean']:.0f} kW"
    )


# ── Full RAG Setup ─────────────────────────────────────────────
def setup_rag(summaries_df, embeddings=None, load_existing=False):
    """
    Full RAG setup pipeline:
    1. Load or initialize embeddings
    2. Create window + seasonal documents
    3. Build or load FAISS index
    4. Return hybrid retriever + seasonal lookup
    """
    if embeddings is None:
        embeddings = load_embeddings()

    # Create documents
    window_docs = create_window_documents(summaries_df)
    seasonal_docs = create_seasonal_documents(summaries_df)
    all_docs = window_docs + seasonal_docs
    print(f"Total documents: {len(all_docs)}")

    # Build seasonal lookup
    seasonal_lookup = build_seasonal_lookup(seasonal_docs)

    if load_existing:
        try:
            vectorstore = load_index(embeddings)
        except Exception as exc:
            print(
                "Existing FAISS index unavailable; rebuilding it "
                f"({type(exc).__name__})."
            )
            vectorstore = FAISS.from_documents(all_docs, embeddings)
            vectorstore.save_local(FAISS_INDEX_PATH)
        bm25_retriever = BM25Retriever.from_documents(all_docs)
        bm25_retriever.k = RAG_TOP_K
        faiss_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": RAG_TOP_K}
        )
        hybrid_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[BM25_WEIGHT, FAISS_WEIGHT]
        )
    else:
        _, _, hybrid_retriever = build_index(all_docs, embeddings, save=True)

    return hybrid_retriever, seasonal_lookup, embeddings


if __name__ == "__main__":
    import pandas as pd
    from config import ALL_ZONES_WINDOW_SUMMARIES_CSV

    print("Loading window summaries...")
    summaries_df = pd.read_csv(ALL_ZONES_WINDOW_SUMMARIES_CSV)

    embeddings = load_embeddings()
    hybrid_retriever, seasonal_lookup, _ = setup_rag(
        summaries_df,
        embeddings=embeddings,
        load_existing=False
    )

    # Quick test
    test_query = "monsoon season July low output cloud cover"
    context = get_rag_context(test_query, hybrid_retriever)
    print(f"\nTest query: '{test_query}'")
    print(context)
