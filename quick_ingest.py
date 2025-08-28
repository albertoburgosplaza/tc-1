#!/usr/bin/env python3
"""
Script mínimo para ingestar PDFs usando solo Qdrant nativo y requests
"""
import os
import json
import requests
import uuid
from pathlib import Path

def simple_text_from_pdf(pdf_path):
    """Extrae texto muy básico de PDF usando pymupdf si está disponible"""
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

def chunk_text(text, chunk_size=1000):
    """División simple de texto en chunks"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        current_chunk.append(word)
        current_size += len(word) + 1
        
        if current_size >= chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_size = 0
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def get_simple_embedding(text):
    """Genera un embedding simple usando hash del texto"""
    import hashlib
    
    # Usar hash del texto para generar vector consistente
    hash_obj = hashlib.sha256(text.encode())
    hash_bytes = hash_obj.digest()
    
    # Convertir a 384 floats normalizados
    vector = []
    for i in range(384):
        # Usar bytes del hash de forma cíclica
        byte_idx = i % len(hash_bytes)
        # Convertir byte a float entre -1 y 1
        val = (hash_bytes[byte_idx] / 255.0) * 2.0 - 1.0
        vector.append(val)
    
    # Normalizar vector para que tenga magnitud 1
    magnitude = sum(x*x for x in vector) ** 0.5
    if magnitude > 0:
        vector = [x/magnitude for x in vector]
    else:
        # Vector por defecto si magnitude es 0
        vector = [1.0/384**0.5] * 384
    
    return vector

def main():
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    COLLECTION = os.getenv("COLLECTION_NAME", "corpus_pdf")
    DOCS_DIR = os.getenv("DOCUMENTS_DIR", "docs")
    
    print(f"Connecting to Qdrant at {QDRANT_URL}")
    
    # Crear colección
    collection_data = {
        "vectors": {
            "size": 384,
            "distance": "Cosine"
        }
    }
    
    try:
        response = requests.put(f"{QDRANT_URL}/collections/{COLLECTION}", json=collection_data)
        if response.status_code in [200, 409]:  # 409 = ya existe
            print(f"Collection {COLLECTION} ready")
        else:
            print(f"Error creating collection: {response.text}")
            return
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")
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
            chunks = chunk_text(page_data['content'])
            
            for chunk_idx, chunk in enumerate(chunks):
                if len(chunk.strip()) < 10:  # Saltar chunks muy cortos
                    continue
                
                # Generar embedding simple
                vector = get_simple_embedding(chunk)
                
                # Crear punto para Qdrant
                point = {
                    "id": point_id,
                    "vector": vector,
                    "payload": {
                        "page_content": chunk,  # LangChain espera este campo
                        "metadata": {
                            "source": page_data['metadata']['source'],
                            "page": page_data['metadata']['page'],
                            "title": page_data['metadata']['title'],
                            "chunk_id": chunk_idx
                        }
                    }
                }
                
                all_points.append(point)
                point_id += 1
    
    print(f"Generated {len(all_points)} chunks")
    
    # Insertar en lotes
    batch_size = 100
    for i in range(0, len(all_points), batch_size):
        batch = all_points[i:i + batch_size]
        
        data = {"points": batch}
        
        try:
            response = requests.put(f"{QDRANT_URL}/collections/{COLLECTION}/points", json=data)
            if response.status_code == 200:
                print(f"Inserted batch {i//batch_size + 1}/{(len(all_points) + batch_size - 1)//batch_size}")
            else:
                print(f"Error inserting batch: {response.text}")
                return
        except Exception as e:
            print(f"Error inserting batch: {e}")
            return
    
    print(f"Successfully ingested {len(all_points)} chunks into Qdrant!")
    
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