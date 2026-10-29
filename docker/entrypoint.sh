#!/bin/sh
set -e
# Build the index on first start if the image was built without PDFs but they are mounted now.
if [ ! -s data/processed/chunks.jsonl ] && ls data/raw/*.pdf >/dev/null 2>&1; then
    RAG_LLM_PROVIDER=ollama rag-eu-law ingest   # ingestion never calls the LLM
fi
# Hosting platforms (Cloud Run, Render, Fly) tell the app which port to use through $PORT.
exec streamlit run src/rag_eu_law/app/ui.py \
    --server.address=0.0.0.0 \
    --server.port="${PORT:-8501}" \
    --server.headless=true \
    --browser.gatherUsageStats=false
