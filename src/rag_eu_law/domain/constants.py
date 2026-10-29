"""Single home for constants shared across modules."""

from __future__ import annotations

DEFAULT_RAW_DIR = "data/raw"
DEFAULT_PROCESSED_DIR = "data/processed"
DEFAULT_LOG_LEVEL = "INFO"

DEFAULT_VECTOR_STORE = "chroma"
DEFAULT_EMBEDDER = "sentence-transformers"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

DEFAULT_TOP_K = 5
DEFAULT_CANDIDATES_PER_RETRIEVER = 20
RRF_RANK_CONSTANT = 60

DEFAULT_LLM_PROVIDER = "ollama"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5-5"
DEFAULT_OPENAI_MODEL = "gpt-6-luna"
DEFAULT_OPENAI_REASONING_EFFORT = "low"

REFUSAL_MESSAGE = "Not covered by the provided documents"
DISCLAIMER = (
    "This output is generated from the ingested texts and is not legal advice. "
    "Verify against the official sources."
)

CHUNK_ID_SEPARATOR = ":"

# --- ingestion ---------------------------------------------------------------
PDF_GLOB = "*.pdf"
CHUNKS_FILENAME = "chunks.jsonl"
IDENTIFICATION_PAGE_COUNT = 2
MAX_WHOLE_ARTICLE_CHARS = 2500
MAX_CHUNK_CHARS = 2000
# Spans at or below this fraction of the page's body font size are superscripts/footnotes.
SMALL_FONT_RATIO = 0.8
# Footnote blocks only start in the lower part of a page (fraction of page height).
FOOTNOTE_ZONE_START = 0.6
# Footnote number and text share a baseline but differ in font size, so their tops differ.
FOOTNOTE_TOP_TOLERANCE_PT = 6.0
REGULATION_SIGNATURES: dict[str, tuple[str, ...]] = {
    "GDPR": ("general data protection regulation", "2016/679"),
    "AI Act": ("artificial intelligence act",),
}

# --- indexing ----------------------------------------------------------------
CHROMA_DIRNAME = "chroma"
CHROMA_COLLECTION = "eu_law_chunks"
KEYWORD_INDEX_FILENAME = "keyword_index.jsonl"
EMBEDDING_BATCH_SIZE = 64
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
HASHING_EMBEDDER_DIMENSION = 256
VECTOR_STORE_CHROMA = "chroma"
VECTOR_STORE_MEMORY = "memory"
EMBEDDER_SENTENCE_TRANSFORMERS = "sentence-transformers"
EMBEDDER_HASHING = "hashing"
EMBEDDER_FASTEMBED = "fastembed"

# --- retrieval ---------------------------------------------------------------
REFERENCE_LIST_WEIGHT = 2.0
REFERENCE_CANDIDATES = 5
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-base"
RERANK_POOL_MULTIPLIER = 4

# --- generation --------------------------------------------------------------
LLM_TIMEOUT_SECONDS = 180.0
LLM_MAX_OUTPUT_TOKENS = 900
LLM_TEMPERATURE = 0.0
# Reasoning models count thinking tokens against the limit, so leave room beyond the answer.
OPENAI_MAX_COMPLETION_TOKENS = 2500
LLM_PROVIDER_OLLAMA = "ollama"
LLM_PROVIDER_ANTHROPIC = "anthropic"

# --- evaluation --------------------------------------------------------------
DEFAULT_QUESTIONS_PATH = "eval/questions.yaml"
DEFAULT_EVAL_OUTPUT_DIR = "eval"
EVAL_REPORT_FILENAME = "report.md"
EVAL_RESULTS_FILENAME = "results.json"

# --- api ---------------------------------------------------------------------
API_MIN_QUESTION_CHARS = 3
API_MAX_QUESTION_CHARS = 2000
API_MAX_TOP_K = 20

# --- history -----------------------------------------------------------------
DEFAULT_HISTORY_DB_PATH = "data/history.sqlite"
DEFAULT_USER = "local"
RATING_UP = 1
RATING_DOWN = -1
HISTORY_PAGE_SIZE = 200
INDEX_VERSION_LENGTH = 12
UNKNOWN_INDEX_VERSION = "unknown"

# --- cloud storage and auth --------------------------------------------------
RUN_STORE_SQLITE = "sqlite"
RUN_STORE_SUPABASE = "supabase"
RUNS_TABLE = "runs"
