"""
Configuration settings and constants for the GST Circular RAG Assistant.

This module serves as the single source of truth for all paths, model identifiers,
retrieval hyperparameters, chunking constraints, and threshold parameters.
"""

from __future__ import annotations

import os
from pathlib import Path

# ==============================================================================
# 1. Base Paths & Directory Layout
# ==============================================================================

# Project root directory: .../Rag_ProJ
BASE_DIR: Path = Path(__file__).resolve().parent.parent

# Raw data directories
DATA_DIR: Path = BASE_DIR / "data"
PDF_DIR: Path = DATA_DIR / "pdfs"

# SQLite metadata & raw chunk store
DB_DIR: Path = BASE_DIR / "db"
DB_PATH: Path = DB_DIR / "chunks.db"

# Retrieval index storage (Qdrant on-disk vector store & serialized BM25 index)
INDEX_DIR: Path = BASE_DIR / "indexes"
QDRANT_PATH: Path = INDEX_DIR / "qdrant_data"
BM25_INDEX_PATH: Path = INDEX_DIR / "bm25_index.pkl"

# Evaluation data & output directories
EVAL_DIR: Path = BASE_DIR / "eval"
EVAL_QUESTIONS_PATH: Path = EVAL_DIR / "questions.json"
CALIBRATION_PAIRS_PATH: Path = EVAL_DIR / "calibration_pairs.json"
EVAL_RESULTS_DIR: Path = EVAL_DIR / "results"

# Ensure runtime directories exist upon module initialization
for directory in [DATA_DIR, PDF_DIR, DB_DIR, INDEX_DIR, EVAL_DIR, EVAL_RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 2. Environment Variables & API Keys
# ==============================================================================

# Load environment variables from .env file if present
ENV_PATH: Path = BASE_DIR / ".env"
if ENV_PATH.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=ENV_PATH)
    except ImportError:
        # Fallback basic parser if python-dotenv is not yet installed
        with open(ENV_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))

# Gemini API Key required for generation & citation answering
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")


# ==============================================================================
# 3. Model Identifiers (CPU-Optimized for v1)
# ==============================================================================

# Bi-encoder dense embedding model (384-dimensional, 33M params, English optimized, runs fast on CPU)
EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM: int = 384

# Cross-encoder reranker model (22M params, runs in <100ms on CPU for top-20 candidates)
RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# LLM generation model via Google Generative AI API
LLM_MODEL: str = "gemini-1.5-flash"


# ==============================================================================
# 4. Chunking Constraints
# ==============================================================================

# Maximum allowed tokens per chunk before sub-paragraph splitting is triggered
MAX_CHUNK_TOKENS: int = 1000

# Minimum token threshold below which orphan or tiny chunks are merged with preceding chunks
MIN_CHUNK_TOKENS: int = 30


# ==============================================================================
# 5. Retrieval & Fusion Hyperparameters
# ==============================================================================

# Number of candidate chunks retrieved by keyword search (BM25)
BM25_TOP_K: int = 20

# Number of candidate chunks retrieved by dense vector search
DENSE_TOP_K: int = 20

# Weights for simple linear hybrid fusion
BM25_WEIGHT: float = 0.5
DENSE_WEIGHT: float = 0.5

# Number of final top chunks returned by hybrid retriever
HYBRID_TOP_K: int = 20

# Smoothing constant for Reciprocal Rank Fusion (RRF): score = 1 / (RRF_K + rank)
RRF_K: int = 60

# Number of top reranked chunks passed to the LLM context after cross-encoder scoring
RERANK_TOP_K: int = 5


# ==============================================================================
# 6. Hallucination & Refusal Thresholds
# ==============================================================================

# Cosine similarity cutoff for sentence-level claim-to-context grounding.
# Default is None until determined empirically via scripts/calibrate_threshold.py.
GROUNDING_THRESHOLD: float | None = None

# If all top-reranked chunks have a cross-encoder score below this threshold,
# the query is flagged as having insufficient context and the system refuses to answer.
RERANKER_REFUSAL_THRESHOLD: float = 0.3

# Overall confidence score cutoff (below 0.6 flags warning to user)
CONFIDENCE_WARNING_THRESHOLD: float = 0.6


# ==============================================================================
# 7. Project Guardrails (Scope Locks)
# ==============================================================================

# Strict v1 limit on number of ingested PDF circulars
MAX_CORPUS_SIZE: int = 25

# Strict limit on total evaluation benchmark questions
MAX_EVAL_QUESTIONS: int = 60
