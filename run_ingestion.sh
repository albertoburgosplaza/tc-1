#!/bin/bash

# Script para ejecutar la ingesta con Python usando contenedor temporal

# Crear contenedor temporal con Python y ejecutar la ingesta
docker run --rm \
    --network turingchallenge-reto-1_rag_internal \
    -v $(pwd)/ingest_simple.py:/app/ingest_simple.py \
    -v $(pwd)/docs:/app/docs \
    -e QDRANT_URL=http://qdrant:6333 \
    -e COLLECTION_NAME=corpus_pdf \
    -e DOCUMENTS_DIR=/app/docs \
    -e EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
    -e CHUNK_SIZE=1200 \
    -e CHUNK_OVERLAP=180 \
    --workdir /app \
    python:3.11-slim \
    bash -c "
        pip install --no-cache-dir langchain-community qdrant-client sentence-transformers pypdf && \
        python ingest_simple.py
    "