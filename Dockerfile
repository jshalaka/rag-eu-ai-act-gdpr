# Setup A image: Streamlit UI + ONNX embeddings (no PyTorch) + Chroma/BM25 index built at build time.
# Secrets (OpenAI, Supabase) are NOT baked in: pass them as environment variables at run time.
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /bin/uv

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    PATH="/app/.venv/bin:$PATH" \
    RAG_EMBEDDER=fastembed \
    RAG_EMBEDDING_CACHE_DIR=/opt/models \
    RAG_LLM_PROVIDER=openai

WORKDIR /app

# Dependencies first, so code edits do not reinstall them.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project \
    --extra index-lite --extra llm --extra ui --extra cloud

COPY src ./src
RUN uv sync --frozen --no-dev --extra index-lite --extra llm --extra ui --extra cloud

# Download the embedding model into the image so start-up is fast and works offline.
RUN python -c "from rag_eu_law.indexing.embedder import FastEmbedEmbedder; \
FastEmbedEmbedder('BAAI/bge-small-en-v1.5', '/opt/models').embed(['warm up'])"

# Build the index now if the PDFs are in the build context; otherwise the entrypoint
# builds it on first start when PDFs are mounted at data/raw.
COPY data/raw ./data/raw
RUN if ls data/raw/*.pdf >/dev/null 2>&1; then RAG_LLM_PROVIDER=ollama rag-eu-law ingest; fi

COPY docker/entrypoint.sh ./docker/entrypoint.sh
RUN useradd --create-home app && mkdir -p data/processed \
    && chown -R app:app /app/data /opt/models
USER app

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://localhost:%s/_stcore/health' % os.environ.get('PORT','8501'))"
ENTRYPOINT ["sh", "docker/entrypoint.sh"]
