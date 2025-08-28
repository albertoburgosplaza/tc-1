#!/usr/bin/env python3
"""
Script de ingesta con embeddings reales usando sentence-transformers
"""
import os
import json
import requests
from pathlib import Path
from sentence_transformers import SentenceTransformer

def simple_text_from_pdf(pdf_path):
    """Extrae texto de PDF usando pymupdf"""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        texts = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                texts.append({
                    'content': text.strip(),
                    'metadata': {
                        'source': Path(pdf_path).name,
                        'page': page_num + 1,
                        'title': Path(pdf_path).stem.replace('_', ' ').replace('-', ' ')
                    }
                })
        doc.close()
        return texts
    except ImportError:
        print(f"PyMuPDF not available. Skipping {pdf_path}")
        return []

def chunk_text(text, chunk_size=1000, overlap=100):
    """División de texto en chunks con solapamiento"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        if chunk_words:
            chunks.append(' '.join(chunk_words))
    
    return chunks

def main():
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    COLLECTION = os.getenv("COLLECTION_NAME", "corpus_pdf")
    DOCS_DIR = os.getenv("DOCUMENTS_DIR", "docs")
    
    print("Loading embedding model...")
    # Usar el mismo modelo que la aplicación
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    print(f"Connecting to Qdrant at {QDRANT_URL}")
    
    # Crear colección con 384 dimensiones (all-MiniLM-L6-v2)
    collection_data = {
        "vectors": {
            "size": 384,
            "distance": "Cosine"
        }
    }
    
    try:
        # Eliminar colección existente si existe
        requests.delete(f"{QDRANT_URL}/collections/{COLLECTION}")
        
        # Crear nueva colección
        response = requests.put(f"{QDRANT_URL}/collections/{COLLECTION}", json=collection_data)
        if response.status_code in [200, 409]:
            print(f"Collection {COLLECTION} ready")
        else:
            print(f"Error creating collection: {response.text}")
            return
    except Exception as e:
        print(f"Error with Qdrant: {e}")
        return
    
    # Procesar PDFs
    pdf_files = list(Path(DOCS_DIR).glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files")
    
    all_points = []
    point_id = 0
    
    for pdf_path in pdf_files:
        print(f"Processing {pdf_path.name}")
        
        # Extraer texto del PDF
        pages = simple_text_from_pdf(str(pdf_path))
        
        for page_data in pages:
            # Dividir en chunks
            chunks = chunk_text(page_data['content'], chunk_size=800, overlap=100)
            
            for chunk_idx, chunk in enumerate(chunks):
                if len(chunk.strip()) < 10:
                    continue
                
                # Generar embedding real con sentence-transformers
                vector = model.encode(chunk, convert_to_numpy=True).tolist()
                
                # Crear punto para Qdrant
                point = {
                    "id": point_id,
                    "vector": vector,
                    "payload": {
                        "page_content": chunk,
                        "metadata": {
                            "source": page_data['metadata']['source'],
                            "page": page_data['metadata']['page'],
                            "title": page_data['metadata']['title'],
                            "chunk_id": chunk_idx,
                            "doc_id": page_data['metadata']['source']
                        }
                    }
                }
                
                all_points.append(point)
                point_id += 1
                
                # Insertar en lotes para evitar problemas de memoria
                if len(all_points) >= 100:
                    print(f"Inserting batch of {len(all_points)} points...")
                    data = {"points": all_points}
                    
                    try:
                        response = requests.put(f"{QDRANT_URL}/collections/{COLLECTION}/points", json=data)
                        if response.status_code != 200:
                            print(f"Error inserting batch: {response.text}")
                    except Exception as e:
                        print(f"Error inserting batch: {e}")
                    
                    all_points = []
    
    # Insertar puntos restantes
    if all_points:
        print(f"Inserting final batch of {len(all_points)} points...")
        data = {"points": all_points}
        
        try:
            response = requests.put(f"{QDRANT_URL}/collections/{COLLECTION}/points", json=data)
            if response.status_code != 200:
                print(f"Error inserting final batch: {response.text}")
        except Exception as e:
            print(f"Error inserting final batch: {e}")
    
    print(f"Successfully ingested {point_id} chunks into Qdrant!")
    
    # Verificar
    try:
        response = requests.get(f"{QDRANT_URL}/collections/{COLLECTION}")
        if response.status_code == 200:
            info = response.json()
            points_count = info['result']['points_count']
            print(f"Verification: Collection now has {points_count} points")
        else:
            print("Could not verify collection status")
    except Exception as e:
        print(f"Error verifying: {e}")

if __name__ == "__main__":
    main()