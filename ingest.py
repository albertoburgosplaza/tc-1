
import os, glob, logging, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.http import models
from embedding_factory import EmbeddingFactory

# Configurar logging estructurado
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("COLLECTION_NAME", "corpus_pdf")
DOCS_DIR = os.getenv("DOCUMENTS_DIR", "docs")
# Configuración de embeddings (alineada con app.py)
EMB_MODEL = os.getenv("EMBEDDING_MODEL", "models/embedding-001")
EMB_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "google")

# Configuración de procesamiento
MAX_PDF_SIZE_MB = int(os.getenv("MAX_PDF_SIZE_MB", "100"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "180"))
MIN_CONTENT_LENGTH = int(os.getenv("MIN_CONTENT_LENGTH", "10"))

# Configuración de paralelización
MAX_WORKERS = int(os.getenv("MAX_WORKERS", min(4, max(2, multiprocessing.cpu_count() - 1))))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "50"))  # Tamaño de batch para inserción de vectores

def validate_pdf_file(pdf_path: str) -> tuple[bool, str]:
    """Valida si un archivo PDF es válido y procesable"""
    path = Path(pdf_path)
    
    # Verificar existencia
    if not path.exists():
        return False, f"El archivo {pdf_path} no existe"
    
    # Verificar que no esté vacío
    if path.stat().st_size == 0:
        return False, f"El archivo {pdf_path} está vacío"
    
    # Verificar extensión
    if path.suffix.lower() != '.pdf':
        return False, f"El archivo {pdf_path} no es un PDF"
    
    # Verificar tamaño razonable
    max_size_bytes = MAX_PDF_SIZE_MB * 1024 * 1024
    if path.stat().st_size > max_size_bytes:
        return False, f"El archivo {pdf_path} es demasiado grande (>{MAX_PDF_SIZE_MB}MB)"
    
    # Verificar que sea un PDF válido (intentar leer los primeros bytes)
    try:
        with open(pdf_path, 'rb') as f:
            header = f.read(5)
            if not header.startswith(b'%PDF-'):
                return False, f"El archivo {pdf_path} no tiene formato PDF válido"
    except Exception as e:
        return False, f"Error al leer {pdf_path}: {str(e)}"
    
    return True, ""

pdf_paths = sorted(glob.glob(str(Path(DOCS_DIR) / "*.pdf")))
if not pdf_paths:
    logger.error(f"No PDF files found in {DOCS_DIR}")
    raise SystemExit(f"No se encontraron PDFs en {DOCS_DIR}")

# Validar todos los PDFs antes de procesar
valid_pdfs = []
for pdf_path in pdf_paths:
    is_valid, error_msg = validate_pdf_file(pdf_path)
    if is_valid:
        valid_pdfs.append(pdf_path)
    else:
        logger.warning(f"Skipping invalid PDF: {error_msg}")

if not valid_pdfs:
    logger.error("No valid PDF files found after validation")
    raise SystemExit("No se encontraron PDFs válidos para procesar")

pdf_paths = valid_pdfs

def validate_document_content(document) -> bool:
    """Valida que el contenido del documento no esté vacío"""
    if not document.page_content or not document.page_content.strip():
        return False
    # Verificar que tenga contenido significativo (más de solo espacios/saltos)
    if len(document.page_content.strip()) < MIN_CONTENT_LENGTH:
        return False
    return True

def process_single_pdf(pdf_path: str) -> tuple[list, int, int, int]:
    """
    Procesa un único PDF y retorna documentos válidos, páginas totales, páginas válidas, páginas vacías
    """
    logger.info(f"Processing PDF: {Path(pdf_path).name}")
    documents = []
    total_pages = 0
    valid_pages = 0
    empty_pages = 0
    
    try:
        # Cargar PDF con manejo de errores específicos
        pdf_documents = PyPDFLoader(pdf_path).load()
        if not pdf_documents:
            logger.warning(f"No pages extracted from {Path(pdf_path).name}")
            return documents, total_pages, valid_pages, empty_pages
            
        logger.info(f"Loaded {len(pdf_documents)} pages from {Path(pdf_path).name}")
        
        # Extraer metadatos mejorados
        file_path = Path(pdf_path)
        clean_title = file_path.stem  # Sin extensión
        # Limpiar guiones bajos y caracteres especiales del título
        clean_title = clean_title.replace('_', ' ').replace('-', ' ')
        # Capitalizar cada palabra
        clean_title = ' '.join(word.capitalize() for word in clean_title.split())
        
        # Procesar cada página con validación
        for page_index, d in enumerate(pdf_documents):
            # Calcular número de página correcto (1-based)
            page_num = page_index + 1
            
            # Asignar metadatos mejorados
            d.metadata["doc_id"] = file_path.stem
            d.metadata["title"] = clean_title
            d.metadata["page"] = page_num
            d.metadata["source"] = file_path.name  # Nombre original del archivo
            
            # Añadir metadatos adicionales si están disponibles en el documento
            if hasattr(d, 'metadata') and d.metadata:
                # Preservar metadatos originales del PDF si existen
                original_metadata = d.metadata.copy()
                if 'author' in original_metadata:
                    d.metadata["author"] = original_metadata["author"]
                if 'creation_date' in original_metadata:
                    d.metadata["creation_date"] = original_metadata["creation_date"]
            
            # Validar contenido de la página
            if validate_document_content(d):
                documents.append(d)
                valid_pages += 1
            else:
                empty_pages += 1
                logger.debug(f"Skipped empty page {page_num} from {clean_title}")
        
        total_pages = len(pdf_documents)
        logger.info(f"Completed {Path(pdf_path).name}: {valid_pages}/{total_pages} valid pages")
        
    except Exception as e:
        logger.error(f"Failed to load PDF {Path(pdf_path).name}: {str(e)}")
        # Retornar valores vacíos en caso de error
    
    return documents, total_pages, valid_pages, empty_pages

logger.info(f"Starting PDF ingestion with parallelization - found {len(pdf_paths)} valid PDF(s) in {DOCS_DIR}")
logger.info(f"Using {MAX_WORKERS} workers for parallel PDF processing")

documents = []
total_pages = 0
empty_pages = 0

# Procesamiento paralelo de PDFs usando ThreadPoolExecutor
# Use ThreadPoolExecutor ya que el IO del PDF es el cuello de botella, no el CPU
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    # Enviar todas las tareas
    future_to_pdf = {executor.submit(process_single_pdf, pdf_path): pdf_path for pdf_path in pdf_paths}
    
    # Recopilar resultados
    for future in as_completed(future_to_pdf):
        pdf_path = future_to_pdf[future]
        try:
            pdf_documents, pdf_total_pages, pdf_valid_pages, pdf_empty_pages = future.result()
            
            # Agregar documentos válidos a la lista principal
            documents.extend(pdf_documents)
            total_pages += pdf_total_pages
            empty_pages += pdf_empty_pages
            
            logger.info(f"Aggregated results from {Path(pdf_path).name}: {pdf_valid_pages} valid pages")
            
        except Exception as e:
            logger.error(f"Failed to process PDF {Path(pdf_path).name}: {str(e)}")
            continue

if not documents:
    logger.error("No valid documents extracted from any PDF files")
    raise SystemExit("No se pudieron extraer documentos válidos de los PDFs")

logger.info(f"PDF processing completed - total_pages: {total_pages}, valid_pages: {len(documents)}, empty_pages: {empty_pages}")

logger.info(f"Starting text splitting - {len(documents)} documents loaded")
logger.info(f"Chunking parameters - CHUNK_SIZE: {CHUNK_SIZE}, CHUNK_OVERLAP: {CHUNK_OVERLAP}")
try:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(documents)
    if not chunks:
        logger.error("No chunks generated during text splitting")
        raise SystemExit("No se pudieron generar chunks de texto")
    
    # Validar tamaños de chunks
    chunk_sizes = [len(chunk.page_content) for chunk in chunks]
    avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes) if chunks else 0
    max_chunk_size = max(chunk_sizes) if chunks else 0
    min_chunk_size = min(chunk_sizes) if chunks else 0
    
    logger.info(f"Text splitting completed - generated {len(chunks)} chunks")
    logger.info(f"Chunk size statistics - avg: {avg_chunk_size:.1f}, min: {min_chunk_size}, max: {max_chunk_size}")
except Exception as e:
    logger.error(f"Text splitting failed: {str(e)}")
    raise SystemExit(f"Error durante el splitting de texto: {str(e)}")

def retry_operation(operation_name: str, operation_func, max_retries: int = 3, delay: float = 2.0):
    """Ejecuta una operación con reintentos en caso de error"""
    for attempt in range(max_retries):
        try:
            return operation_func()
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"{operation_name} failed after {max_retries} attempts: {str(e)}")
                raise
            else:
                logger.warning(f"{operation_name} failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Backoff exponencial

logger.info(f"Initializing embedding model: {EMB_MODEL} ({EMB_PROVIDER})")
# Usar factory para crear embeddings configurables con proveedor específico
# Para documentos usar retrieval.passage con late chunking habilitado
task_config = {
    "task_type": "retrieval.passage" if EMB_PROVIDER == "jina" else "retrieval_document",
    "late_chunking": True if EMB_PROVIDER == "jina" else False,
}
emb = EmbeddingFactory.create_embedding(
    model_name=EMB_MODEL, 
    provider=EMB_PROVIDER,
    **task_config
)
# Obtener dimensiones del modelo seleccionado
dim = EmbeddingFactory.get_model_dimensions(EMB_MODEL, EMB_PROVIDER)

def create_qdrant_client():
    """Crear cliente Qdrant con manejo de errores"""
    try:
        client = QdrantClient(url=QDRANT_URL)
        # Verificar conexión
        client.get_collections()
        logger.info(f"Successfully connected to Qdrant at {QDRANT_URL}")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {str(e)}")
        raise

client = retry_operation("Qdrant client creation", create_qdrant_client)

def setup_qdrant_collection():
    """Configurar colección de Qdrant con manejo de errores"""
    collections = client.get_collections().collections
    existing = {c.name for c in collections}
    if COLLECTION not in existing:
        logger.info(f"Creating new Qdrant collection: {COLLECTION}")
        # Usar DOT distance si el proveedor es Jina (normalized=True)
        # Usar COSINE para otros proveedores
        distance_metric = Distance.DOT if EMB_PROVIDER == "jina" else Distance.COSINE
        logger.info(f"Using distance metric: {distance_metric.name} for provider: {EMB_PROVIDER}")
        
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=dim, distance=distance_metric),
        )
        logger.info(f"Collection {COLLECTION} created successfully with {dim} dimensions")
    else:
        logger.info(f"Using existing collection: {COLLECTION}")

retry_operation("Qdrant collection setup", setup_qdrant_collection)

def insert_vectors_to_qdrant():
    """Insertar vectores en Qdrant usando el cliente nativo"""
    logger.info(f"Starting vector insertion into Qdrant - {len(chunks)} chunks")
    
    # Procesar en lotes para eficiencia
    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    logger.info(f"Processing in {total_batches} batches of {BATCH_SIZE} chunks each")
    
    for i in range(0, len(chunks), BATCH_SIZE):
        batch_num = (i // BATCH_SIZE) + 1
        batch = chunks[i:i + BATCH_SIZE]
        
        logger.info(f"Inserting batch {batch_num}/{total_batches} ({len(batch)} chunks)")
        
        try:
            # Generar embeddings y puntos para insertar
            texts = [doc.page_content for doc in batch]
            metadatas = [doc.metadata for doc in batch]
            
            # Generar embeddings para los textos
            embeddings = emb.embed_documents(texts)
            
            # Crear puntos para Qdrant
            points = []
            for j, (text, metadata, embedding) in enumerate(zip(texts, metadatas, embeddings)):
                point_id = i + j  # ID único para cada punto
                # Combinar metadata con page_content
                payload = {**metadata, "page_content": text}
                points.append(models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                ))
            
            # Insertar puntos en Qdrant
            client.upsert(collection_name=COLLECTION, points=points)
            logger.debug(f"Batch {batch_num} inserted successfully")
            
        except Exception as e:
            logger.error(f"Failed to insert batch {batch_num}: {str(e)}")
            raise
    
    logger.info(f"Vector insertion completed - {len(chunks)} chunks inserted")

retry_operation("Vector insertion to Qdrant", insert_vectors_to_qdrant)

logger.info(f"Ingestion completed successfully - collection: '{COLLECTION}', chunks: {len(chunks)}")
