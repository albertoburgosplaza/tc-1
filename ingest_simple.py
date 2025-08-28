import os, glob, logging
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuración
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("COLLECTION_NAME", "corpus_pdf")
DOCS_DIR = os.getenv("DOCUMENTS_DIR", "docs")
EMB_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "180"))

def main():
    # 1. Cargar PDFs
    pdf_paths = glob.glob(str(Path(DOCS_DIR) / "*.pdf"))
    logger.info(f"Found {len(pdf_paths)} PDF files")
    
    documents = []
    for pdf_path in pdf_paths:
        logger.info(f"Loading {Path(pdf_path).name}")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        # Añadir metadatos
        for i, doc in enumerate(docs):
            doc.metadata["source"] = Path(pdf_path).name
            doc.metadata["page"] = i + 1
        
        documents.extend(docs)
    
    logger.info(f"Loaded {len(documents)} pages from {len(pdf_paths)} PDFs")
    
    # 2. Dividir en chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks")
    
    # 3. Generar embeddings
    logger.info("Initializing embedding model")
    embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL)
    
    # 4. Conectar a Qdrant
    logger.info("Connecting to Qdrant")
    client = QdrantClient(url=QDRANT_URL)
    
    # 5. Crear colección si no existe
    collections = client.get_collections().collections
    existing = {c.name for c in collections}
    
    if COLLECTION not in existing:
        logger.info(f"Creating collection {COLLECTION}")
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
    else:
        logger.info(f"Using existing collection {COLLECTION}")
    
    # 6. Insertar vectores en lotes
    batch_size = 50
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    
    for i in range(0, len(chunks), batch_size):
        batch_num = (i // batch_size) + 1
        batch = chunks[i:i + batch_size]
        
        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)")
        
        # Generar embeddings para el batch
        texts = [doc.page_content for doc in batch]
        vectors = embeddings.embed_documents(texts)
        
        # Crear puntos
        points = []
        for j, (text, vector, doc) in enumerate(zip(texts, vectors, batch)):
            point_id = i + j
            points.append(PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "text": text,
                    "source": doc.metadata.get("source", ""),
                    "page": doc.metadata.get("page", 0)
                }
            ))
        
        # Insertar en Qdrant
        client.upsert(collection_name=COLLECTION, points=points)
        logger.info(f"Inserted batch {batch_num}")
    
    logger.info(f"Successfully ingested {len(chunks)} chunks into Qdrant")

if __name__ == "__main__":
    main()