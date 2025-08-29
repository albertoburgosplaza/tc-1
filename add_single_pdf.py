#!/usr/bin/env python3
"""
Script para añadir un solo PDF al vector database sin re-procesar todos los documentos
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from embedding_factory import EmbeddingFactory
import uuid
import hashlib
from typing import List

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuración
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("COLLECTION_NAME", "corpus_pdf")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "180"))
EMB_MODEL = os.getenv("EMBEDDING_MODEL", "jina-embeddings-v4")
EMB_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "jina")

def validate_pdf(pdf_path: str) -> bool:
    """Validar que el PDF sea procesable"""
    path = Path(pdf_path)
    
    if not path.exists():
        logger.error(f"File does not exist: {pdf_path}")
        return False
    
    if not path.suffix.lower() == '.pdf':
        logger.error(f"File is not a PDF: {pdf_path}")
        return False
    
    if path.stat().st_size == 0:
        logger.error(f"File is empty: {pdf_path}")
        return False
    
    # Verificar header PDF
    try:
        with open(pdf_path, 'rb') as f:
            header = f.read(5)
            if not header.startswith(b'%PDF-'):
                logger.error(f"Invalid PDF format: {pdf_path}")
                return False
    except Exception as e:
        logger.error(f"Error reading PDF: {e}")
        return False
    
    return True

def check_if_document_exists(client: QdrantClient, pdf_path: str) -> bool:
    """Verificar si el documento ya existe en la base de datos"""
    try:
        # Usar hash del nombre del archivo como identificador
        doc_id = hashlib.md5(Path(pdf_path).name.encode()).hexdigest()
        
        # Buscar documentos con este source
        search_result = client.scroll(
            collection_name=COLLECTION,
            scroll_filter={"must": [{"key": "source", "match": {"value": Path(pdf_path).name}}]},
            limit=1
        )
        
        return len(search_result[0]) > 0
        
    except Exception as e:
        logger.warning(f"Could not check if document exists: {e}")
        return False

def process_single_pdf(pdf_path: str, skip_existing: bool = True) -> bool:
    """Procesar un solo PDF y añadirlo al vector database"""
    
    logger.info(f"Processing PDF: {pdf_path}")
    
    # Validar PDF
    if not validate_pdf(pdf_path):
        return False
    
    # Conectar a Qdrant
    try:
        client = QdrantClient(url=QDRANT_URL)
        
        # Verificar si la colección existe
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if COLLECTION not in collection_names:
            logger.error(f"Collection '{COLLECTION}' does not exist. Please run full ingestion first.")
            return False
            
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        return False
    
    # Verificar si el documento ya existe
    if skip_existing and check_if_document_exists(client, pdf_path):
        logger.info(f"Document already exists in database: {Path(pdf_path).name}")
        return True
    
    try:
        # Cargar PDF
        logger.info("Loading PDF content...")
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        
        if not pages:
            logger.error("No pages found in PDF")
            return False
        
        logger.info(f"Loaded {len(pages)} pages")
        
        # Procesar páginas
        documents = []
        for page_num, page in enumerate(pages, 1):
            if not page.page_content.strip():
                continue
            
            # Añadir metadatos
            page.metadata.update({
                "source": Path(pdf_path).name,
                "title": Path(pdf_path).stem.replace('_', ' ').title(),
                "page": page_num,
                "total_pages": len(pages)
            })
            
            documents.append(page)
        
        if not documents:
            logger.error("No valid content found in PDF")
            return False
        
        # Dividir en chunks
        logger.info("Splitting documents into chunks...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", ".", " "]
        )
        
        splits = splitter.split_documents(documents)
        logger.info(f"Created {len(splits)} chunks")
        
        # Crear embeddings
        logger.info("Creating embeddings...")
        embeddings = EmbeddingFactory.create_embedding(
            model_name=EMB_MODEL,
            provider=EMB_PROVIDER,
            task_type="retrieval.passage",  # Para documentos
            late_chunking=True  # Activar late chunking para documentos
        )
        
        # Generar vectores
        texts = [doc.page_content for doc in splits]
        vectors = embeddings.embed_documents(texts)
        
        logger.info(f"Generated {len(vectors)} embedding vectors")
        
        # Preparar points para Qdrant
        points = []
        for i, (doc, vector) in enumerate(zip(splits, vectors)):
            point_id = str(uuid.uuid4())
            
            # Payload con metadata
            payload = {
                "page_content": doc.page_content,
                **doc.metadata
            }
            
            points.append(PointStruct(
                id=point_id,
                vector=vector,
                payload=payload
            ))
        
        # Insertar en Qdrant
        logger.info(f"Inserting {len(points)} points into Qdrant...")
        client.upsert(
            collection_name=COLLECTION,
            points=points
        )
        
        logger.info(f"✅ Successfully added PDF to vector database: {Path(pdf_path).name}")
        logger.info(f"   - Pages: {len(documents)}")
        logger.info(f"   - Chunks: {len(splits)}")
        logger.info(f"   - Vectors: {len(vectors)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to process PDF: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Add a single PDF to the vector database")
    parser.add_argument("pdf_path", help="Path to the PDF file to add")
    parser.add_argument("--force", action="store_true", help="Force re-processing even if document exists")
    
    args = parser.parse_args()
    
    # Verificar variables de entorno
    required_vars = ["GOOGLE_API_KEY", "JINA_API_KEY"]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        logger.error(f"Missing required environment variables: {missing}")
        sys.exit(1)
    
    success = process_single_pdf(args.pdf_path, skip_existing=not args.force)
    
    if success:
        logger.info("🎉 PDF successfully added to vector database!")
        sys.exit(0)
    else:
        logger.error("💥 Failed to add PDF to vector database")
        sys.exit(1)

if __name__ == "__main__":
    main()