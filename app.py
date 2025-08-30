
import os, re, time, requests, logging, uuid
from typing import List, Tuple, Dict
import gradio as gr
from threading import Thread
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import hashlib
from functools import lru_cache
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from functools import partial
import base64
from pathlib import Path

from langchain_community.vectorstores import Qdrant
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from qdrant_client import QdrantClient
from embedding_factory import EmbeddingFactory, embedding_metrics
from jina_reranker import create_jina_reranker
from multimodal_schema import MULTIMODAL_COLLECTION_CONFIG
from image_extractor import image_extraction_metrics
from image_storage import image_storage_metrics

# Configurar logging estructurado
logging.basicConfig(
    level=logging.DEBUG if os.getenv("DEBUG", "false").lower() == "true" else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class ParallelRetriever:
    """
    Wrapper para el retriever que puede realizar búsquedas paralelas cuando es beneficioso
    """
    def __init__(self, base_retriever, max_workers: int = None):
        self.base_retriever = base_retriever
        # Set default value at runtime when the variable is available
        self.max_workers = max_workers if max_workers is not None else int(os.getenv("MAX_RETRIEVAL_WORKERS", "2"))
        self.parallel_enabled = os.getenv("PARALLEL_RETRIEVAL", "true").lower() == "true"
        logger.info(f"Initialized parallel retriever - parallel_enabled: {self.parallel_enabled}, max_workers: {self.max_workers}")
    
    def get_relevant_documents(self, query: str, k: int = None):
        """
        Obtiene documentos relevantes para una consulta individual
        """
        # Para consultas individuales, usar el retriever base directamente
        # La paralelización es más útil para búsquedas por lotes
        if k is not None and hasattr(self.base_retriever, 'search_kwargs'):
            # Configurar k temporalmente si se especifica
            original_k = self.base_retriever.search_kwargs.get('k', 5)
            self.base_retriever.search_kwargs['k'] = k
            try:
                docs = self.base_retriever.get_relevant_documents(query)
            finally:
                self.base_retriever.search_kwargs['k'] = original_k
        else:
            docs = self.base_retriever.get_relevant_documents(query)
        
        return docs
    
    def batch_get_relevant_documents(self, queries: List[str], k: int = None) -> List[List]:
        """
        Obtiene documentos relevantes para múltiples consultas en paralelo
        """
        if not self.parallel_enabled or len(queries) == 1:
            # Usar procesamiento secuencial si no está habilitado o hay una sola consulta
            logger.debug("Using sequential batch processing")
            return [self.get_relevant_documents(query, k) for query in queries]
        
        logger.info(f"Processing {len(queries)} queries in parallel with {self.max_workers} workers")
        
        results = [None] * len(queries)  # Mantener orden original
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Crear partial function para get_relevant_documents con k
            get_docs_func = partial(self.get_relevant_documents, k=k) if k else self.get_relevant_documents
            
            # Enviar todas las consultas
            future_to_index = {
                executor.submit(get_docs_func, query): i 
                for i, query in enumerate(queries)
            }
            
            # Recopilar resultados manteniendo el orden
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                    logger.debug(f"Completed batch query {index + 1}/{len(queries)}")
                except Exception as e:
                    logger.error(f"Failed to process batch query {index}: {str(e)}")
                    results[index] = []  # Retornar lista vacía en caso de error
        
        logger.info(f"Completed {len(queries)} parallel queries")
        return results

# Entorno
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("COLLECTION_NAME", "corpus_pdf")
PYEXEC_URL = os.getenv("PYEXEC_URL", "http://localhost:8001")

# Configuración del reranker
JINA_RERANKER_ENABLED = os.getenv("JINA_RERANKER_ENABLED", "true").lower() == "true"
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "30"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "15"))
JINA_RERANKER_TIMEOUT = int(os.getenv("JINA_RERANKER_TIMEOUT", "20"))


# Configuración optimizada de conexiones HTTP
HTTP_TIMEOUT_FAST = 3     # Para health checks rápidos
HTTP_TIMEOUT_NORMAL = 5   # Para operaciones regulares (pyexec, etc)
HTTP_TIMEOUT_SLOW = 8     # Para operaciones lentas si es necesario

# Configuración de connection pooling
HTTP_POOL_CONNECTIONS = 10  # Número de pools de conexión
HTTP_POOL_MAXSIZE = 20     # Tamaño máximo por pool
HTTP_MAX_RETRIES = 2       # Reintentos en caso de error

# Configuración de paralelización para retrieval
PARALLEL_RETRIEVAL_ENABLED = os.getenv("PARALLEL_RETRIEVAL", "true").lower() == "true"
MAX_RETRIEVAL_WORKERS = int(os.getenv("MAX_RETRIEVAL_WORKERS", "2"))  # Conservative for embeddings

def create_optimized_session(timeout: int = HTTP_TIMEOUT_NORMAL, max_retries: int = HTTP_MAX_RETRIES) -> requests.Session:
    """
    Crea una sesión HTTP optimizada con connection pooling y retry logic
    
    Args:
        timeout: Timeout por defecto para las requests
        max_retries: Número máximo de reintentos
    
    Returns:
        requests.Session configurada con pooling optimizado
    """
    session = requests.Session()
    
    # Configurar retry strategy - más conservadora para evitar latencia
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=0.1,  # Backoff rápido: 0.1s, 0.2s
        status_forcelist=[500, 502, 503, 504],  # Solo retry en errores de servidor
        allowed_methods=["GET", "POST"]  # Retry solo en métodos seguros
    )
    
    # Crear adapter con connection pooling optimizado
    adapter = HTTPAdapter(
        pool_connections=HTTP_POOL_CONNECTIONS,
        pool_maxsize=HTTP_POOL_MAXSIZE,
        max_retries=retry_strategy,
        pool_block=False  # No bloquear si el pool está lleno
    )
    
    # Montar adapter para HTTP y HTTPS
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # Configurar timeout por defecto
    session.timeout = timeout
    
    logger.debug(f"Created HTTP session with timeout={timeout}s, pool_size={HTTP_POOL_MAXSIZE}, connections={HTTP_POOL_CONNECTIONS}")
    
    return session

# Crear sesiones HTTP optimizadas para diferentes tipos de operaciones
health_session = create_optimized_session(timeout=HTTP_TIMEOUT_FAST, max_retries=1)  # Health checks rápidos
api_session = create_optimized_session(timeout=HTTP_TIMEOUT_NORMAL, max_retries=HTTP_MAX_RETRIES)  # APIs normales
compute_session = create_optimized_session(timeout=HTTP_TIMEOUT_NORMAL, max_retries=1)  # Computación (pyexec)

def create_llm(provider="google", model=None):
    """Crea una instancia de LLM según el proveedor especificado"""
    if provider.lower() == "google":
        if not GOOGLE_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY is required for Google provider. "
                "Please configure it in environment variables."
            )
        # Usar Gemini 2.5 Flash Lite por defecto
        gemini_model = model or "gemini-2.5-flash-lite"
        return ChatGoogleGenerativeAI(
            model=gemini_model,
            temperature=0.05,
            google_api_key=GOOGLE_API_KEY,
            max_tokens=512,
            convert_system_message_to_human=True  # Gemini no soporta system messages
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}. Only 'google' is supported.")

# Variables globales para el modelo actual
current_llm = None
current_provider = LLM_PROVIDER
current_model = "gemini-2.5-flash-lite"

def switch_llm(provider="google", model_name=None):
    """Cambia el modelo LLM dinámicamente"""
    global current_llm, current_provider, current_model
    
    logger.info(f"Switching LLM from {current_provider} to {provider}")
    
    try:
        new_model = model_name or "gemini-2.5-flash-lite"
        new_llm = create_llm(provider, new_model)
        
        # Test básico del nuevo modelo
        test_response = new_llm.invoke("Responde solo 'OK'").content
        if test_response:
            current_llm = new_llm
            current_provider = provider
            current_model = new_model
            logger.info(f"Successfully switched to {provider} model: {new_model}")
            return True
        else:
            logger.error(f"Failed to switch to {provider}: No response from model")
            return False
    except Exception as e:
        logger.error(f"Failed to switch to {provider}: {str(e)}")
        return False

# Inicializar LLM de forma segura
try:
    current_llm = create_llm(LLM_PROVIDER)
    logger.info(f"Successfully initialized {LLM_PROVIDER} model: {current_model}")
except Exception as e:
    logger.critical(f"Failed to initialize {LLM_PROVIDER} model: {str(e)}")
    logger.critical("Make sure GOOGLE_API_KEY is configured correctly")
    raise e

llm = current_llm  # Mantener compatibilidad con código existente

# Inicializar embeddings usando factory configurable
# Para búsqueda multimodal usar jina-embeddings-v4
embedding_model = os.getenv("EMBEDDING_MODEL", "jina-embeddings-v4")
embedding_provider = os.getenv("EMBEDDING_PROVIDER", "jina")

logger.info(f"Initializing embeddings for multimodal search: {embedding_model} ({embedding_provider})")
# Para queries usar retrieval.query sin late chunking
query_task_config = {
    "task_type": "retrieval.query" if embedding_provider == "jina" else "retrieval_document",
    "late_chunking": False,  # Nunca usar late chunking en queries
}
base_emb = EmbeddingFactory.create_embedding(
    model_name=embedding_model,
    provider=embedding_provider,
    **query_task_config
)

# Usar embeddings de Google directamente (son API calls, no necesitan cache local)
emb = base_emb

client = QdrantClient(url=QDRANT_URL)

# Custom retriever that directly uses Qdrant client to handle metadata properly
class CustomQdrantRetriever:
    """Custom retriever that properly handles Qdrant payload metadata"""
    
    def __init__(self, client, collection_name, embeddings, k=15):
        self.client = client
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.k = k
    
    def get_relevant_documents(self, query: str, k: int = None):
        """Get relevant documents with proper multimodal metadata handling"""
        from langchain_core.documents import Document
        
        # Use provided k or default
        search_k = k if k is not None else self.k
        
        # Generate embedding for query
        query_embedding = self.embeddings.embed_query(query)
        
        # Search in Qdrant
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=search_k,
            with_payload=True
        )
        
        # Convert to LangChain documents with multimodal metadata
        documents = []
        for result in search_result.points:
            # Extract metadata and content from payload
            payload = result.payload.copy()  # Make a copy to avoid modifying original
            
            # Extract modality-specific content
            modality = payload.get('modality', 'text')
            page_content = payload.pop('page_content', '') or ''  # Handle None values from images
            
            # Preserve similarity score for reranking
            metadata = payload.copy()
            metadata['similarity_score'] = result.score
            
            # Handle content based on modality
            if modality == 'image':
                # For images, use thumbnail info as content or description
                thumbnail_uri = payload.get('thumbnail_uri', '')
                width = payload.get('width', 0)
                height = payload.get('height', 0)
                image_index = payload.get('image_index', 0)
                bbox = payload.get('bbox', {})
                
                # Ensure all required image metadata is preserved in metadata
                metadata.update({
                    'thumbnail_uri': thumbnail_uri,
                    'image_index': image_index,
                    'bbox': bbox,
                    'width': width,
                    'height': height
                })
                
                # Create descriptive content for image
                if thumbnail_uri:
                    page_content = f"Imagen {image_index + 1} en página {payload.get('page_number', 'N/A')} (dimensiones: {width}x{height}px, thumbnail: {thumbnail_uri})"
                else:
                    page_content = f"Imagen {image_index + 1} en página {payload.get('page_number', 'N/A')} (dimensiones: {width}x{height}px)"
            
            # Create document with enhanced metadata
            doc = Document(
                page_content=page_content,
                metadata=metadata  # All fields including modality, similarity_score, etc.
            )
            documents.append(doc)
        
        return documents

base_retriever = CustomQdrantRetriever(client, MULTIMODAL_COLLECTION_CONFIG["collection_name"], emb, k=RETRIEVAL_TOP_K)

# Crear retriever paralelo que envuelve el retriever base
retriever = ParallelRetriever(base_retriever, max_workers=MAX_RETRIEVAL_WORKERS)

# Inicializar reranker si está habilitado
reranker = None
if JINA_RERANKER_ENABLED:
    try:
        reranker = create_jina_reranker(
            enabled=True,
            timeout=JINA_RERANKER_TIMEOUT
        )
        if reranker:
            logger.info(f"Jina Reranker enabled: retrieval_k={RETRIEVAL_TOP_K}, rerank_k={RERANK_TOP_K}")
        else:
            logger.warning("Jina Reranker is disabled via environment variable")
    except Exception as e:
        logger.error(f"Failed to initialize Jina Reranker: {str(e)}")
        logger.warning("Continuing without reranker")
        reranker = None
else:
    logger.info("Jina Reranker disabled via configuration")

# Memoria simple
MAX_HISTORY_CHARS = int(os.getenv("MAX_HISTORY_CHARS", "8000"))
SLIDING_WINDOW_TURNS = int(os.getenv("SLIDING_WINDOW_TURNS", "6"))

# Configuración de validación
MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", "2000"))
MIN_QUERY_LENGTH = int(os.getenv("MIN_QUERY_LENGTH", "1"))

# Caracteres peligrosos básicos (no configurable por seguridad)
DANGEROUS_CHARS = ['<script', '</script', 'javascript:', 'onload=', 'onerror=', 'eval(']

# Métricas y contadores globales
request_count = 0
error_count = 0
rag_latency_sum = 0.0
rag_request_count = 0
python_request_count = 0
start_time = time.time()

def check_service_health(url: str, endpoint: str, timeout: int = HTTP_TIMEOUT_FAST) -> Dict:
    """Verifica la salud de un servicio externo usando sesión optimizada"""
    try:
        # Usar sesión optimizada para health checks con timeout reducido
        response = health_session.get(f"{url}{endpoint}", timeout=timeout)
        if response.status_code == 200:
            latency_ms = response.elapsed.total_seconds() * 1000
            logger.debug(f"Health check successful for {url}{endpoint} - latency: {latency_ms:.1f}ms")
            return {"status": "healthy", "latency_ms": latency_ms}
        else:
            logger.warning(f"Health check failed for {url}{endpoint} - HTTP {response.status_code}")
            return {"status": "unhealthy", "error": f"HTTP {response.status_code}"}
    except requests.exceptions.Timeout:
        logger.warning(f"Health check timeout for {url}{endpoint} after {timeout}s")
        return {"status": "unhealthy", "error": "timeout"}
    except requests.exceptions.ConnectionError:
        logger.warning(f"Health check connection error for {url}{endpoint}")
        return {"status": "unhealthy", "error": "connection_error"}
    except Exception as e:
        logger.error(f"Health check failed for {url}{endpoint}: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}

def get_app_status() -> Dict:
    """Retorna el estado completo de la aplicación y sus dependencias"""
    uptime = time.time() - start_time
    
    # Verificar servicios críticos
    services = {
        "qdrant": check_service_health(QDRANT_URL, "/healthz"),
        "pyexec": check_service_health(PYEXEC_URL, "/health")
    }
    
    # Calcular latencia promedio de RAG
    avg_rag_latency = (rag_latency_sum / rag_request_count) if rag_request_count > 0 else 0.0
    
    # Obtener estadísticas de caché de embeddings (deshabilitado temporalmente)
    cache_stats = {"hits": 0, "misses": 0, "hit_rate": 0.0, "current_size": 0, "max_size": 200}
    
    # Obtener métricas del reranker si está disponible
    reranker_metrics = {}
    if reranker:
        try:
            reranker_metrics = reranker.get_metrics()
        except Exception as e:
            logger.warning(f"Failed to get reranker metrics: {e}")
            reranker_metrics = {"error": "Failed to get metrics"}
    else:
        reranker_metrics = {"status": "disabled"}
    
    # Obtener métricas multimodales
    multimodal_metrics = {}
    try:
        # Métricas de extracción de imágenes
        extraction_metrics = image_extraction_metrics.get_metrics_summary()
        
        # Métricas de embeddings de imágenes
        embedding_metrics_summary = embedding_metrics.get_metrics_summary()
        
        # Métricas de almacenamiento de imágenes
        storage_metrics_summary = image_storage_metrics.get_metrics_summary()
        
        multimodal_metrics = {
            "image_extraction": extraction_metrics,
            "image_embeddings": embedding_metrics_summary,
            "image_storage": storage_metrics_summary,
            "multimodal_enabled": True
        }
        
    except Exception as e:
        logger.warning(f"Failed to get multimodal metrics: {e}")
        multimodal_metrics = {
            "error": f"Failed to get multimodal metrics: {str(e)}",
            "multimodal_enabled": False
        }
    
    # Estado general
    all_healthy = all(service["status"] == "healthy" for service in services.values())
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "uptime_seconds": uptime,
        "metrics": {
            "total_requests": request_count,
            "total_errors": error_count,
            "rag_requests": rag_request_count,
            "python_requests": python_request_count,
            "avg_rag_latency_ms": round(avg_rag_latency * 1000, 2),
            "error_rate": round((error_count / request_count) * 100, 2) if request_count > 0 else 0.0
        },
        "embedding_cache": {
            "hits": cache_stats["hits"],
            "misses": cache_stats["misses"],
            "hit_rate_percent": round(cache_stats["hit_rate"] * 100, 2),
            "current_size": cache_stats["current_size"],
            "max_size": cache_stats["max_size"],
            "utilization_percent": round((cache_stats["current_size"] / cache_stats["max_size"]) * 100, 2) if cache_stats["max_size"] > 0 else 0.0
        },
        "multimodal": multimodal_metrics,
        "reranker": reranker_metrics,
        "services": services,
        "timestamp": time.time()
    }

class HealthHandler(BaseHTTPRequestHandler):
    """Handler para endpoints de salud"""
    def do_GET(self):
        if self.path == "/health":
            # Health check básico
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            response = {"status": "healthy", "timestamp": time.time()}
            self.wfile.write(json.dumps(response).encode())
        
        elif self.path == "/status":
            # Status completo con dependencias
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            response = get_app_status()
            self.wfile.write(json.dumps(response, indent=2).encode())
        
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        # Silenciar logs de health checks
        pass

def start_health_server():
    """Inicia el servidor de health checks en un thread separado"""
    try:
        server = HTTPServer(('0.0.0.0', 8080), HealthHandler)
        logger.info("Health check server started on port 8080")
        server.serve_forever()
    except Exception as e:
        logger.error(f"Failed to start health server: {e}")

def validate_input(text: str, field_name: str = "input") -> tuple[bool, str]:
    """Valida entrada del usuario y retorna (es_válido, mensaje_error)"""
    if not text or not text.strip():
        return False, f"{field_name} no puede estar vacío"
    
    if len(text) > MAX_QUERY_LENGTH:
        return False, f"{field_name} excede el límite de {MAX_QUERY_LENGTH} caracteres"
    
    if len(text.strip()) < MIN_QUERY_LENGTH:
        return False, f"{field_name} debe tener al menos {MIN_QUERY_LENGTH} carácter"
    
    # Verificar caracteres peligrosos
    text_lower = text.lower()
    for dangerous in DANGEROUS_CHARS:
        if dangerous in text_lower:
            return False, "El texto contiene contenido no permitido"
    
    return True, ""

def sanitize_input(text: str) -> str:
    """Sanitiza la entrada básicamente"""
    # Remover caracteres de control básicos
    sanitized = ''.join(char for char in text if ord(char) >= 32 or char in ['\n', '\t'])
    # Limitar longitud
    return sanitized[:MAX_QUERY_LENGTH]

def fmt_hist(h: List[Tuple[str, str]], enhanced: bool = False):
    """
    Formatea el historial de conversación
    
    Args:
        h: Lista de tuplas (usuario, asistente) 
        enhanced: Si True, usa formato mejorado con marcadores visuales
    """
    if not enhanced:
        # Formato básico para compatibilidad
        return "\n".join([f"Usuario: {u}\nAsistente: {a}" for u, a in h])
    
    # Formato mejorado con marcadores visuales y metadatos
    formatted_turns = []
    for i, (u, a) in enumerate(h, 1):
        # Detectar si es un resumen
        is_summary = u == "RESUMEN"
        
        if is_summary:
            formatted_turn = f"""
═══════ RESUMEN DE CONVERSACIÓN ANTERIOR ═══════
{a}
═══════════════════════════════════════════════
"""
        else:
            # Detectar tipo de consulta para mejor contexto
            is_python = "python:" in u.lower() or any(word in u.lower() for word in ["calcular", "media", "percentil"])
            query_type = "[Python] " if is_python else "[RAG] "
            
            formatted_turn = f"""
┌─ Turno {i} {query_type}──────────────────────────────────────
│ Usuario: {u.strip()}
├─────────────────────────────────────────────────────────────
│ Asistente: {a.strip()}
└─────────────────────────────────────────────────────────────"""
        
        formatted_turns.append(formatted_turn)
    
    return "\n".join(formatted_turns)

def calculate_prompt_size(history: List[Tuple[str, str]], context: str = "", sys_prompt: str = "") -> int:
    """Calcula el tamaño total del prompt considerando todos los componentes"""
    # Historial formateado (usar formato enhanced para cálculos precisos)
    hist_text = fmt_hist(history, enhanced=True)
    
    # Sistema prompt base optimizado (de RAG)
    base_sys = """Responde usando SOLO el contexto proporcionado. Si no hay información, responde: "No hay información suficiente". Cita fuentes: [Documento - Página]. Sin inferencias ni conocimiento previo. Respuestas concisas."""
    
    # Overhead del template de prompt optimizado
    template_overhead = len("""

Contexto:


Historial:


Pregunta: 

Respuesta:""")
    
    # Calcular tamaño total
    total_size = (
        len(base_sys if not sys_prompt else sys_prompt) +
        len(context) +
        len(hist_text) +
        template_overhead +
        100  # Buffer para query del usuario y variaciones
    )
    
    return total_size

def identify_critical_info(history: List[Tuple[str, str]]) -> str:
    """Identifica y extrae información crítica que debe preservarse"""
    critical_patterns = [
        (r'\b\d{4}-\d{2}-\d{2}\b', 'FECHAS'),  # Fechas formato YYYY-MM-DD
        (r'\b\d{1,2}/\d{1,2}/\d{4}\b', 'FECHAS'),  # Fechas formato DD/MM/YYYY
        (r'\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+ [A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\b', 'NOMBRES'),  # Nombres propios
        (r'\b(?:prefiero|quiero|necesito|decidí|opté por|elegiré|mi decisión es)\b.*?(?:\.|!|$)', 'DECISIONES'),  # Decisiones explícitas
        (r'\b(?:configurar|establecer|setear|ajustar)\s+\w+\s+(?:en|a|como)\s+\w+', 'CONFIGURACIONES'),  # Configuraciones
        (r'\b\d+(?:\.\d+)?(?:\s*%|\s*euros?|\s*dólares?|\s*km|\s*metros?)\b', 'NÚMEROS'),  # Números con unidades
    ]
    
    critical_info = {
        'FECHAS': [],
        'NOMBRES': [],
        'DECISIONES': [],
        'CONFIGURACIONES': [], 
        'NÚMEROS': []
    }
    
    text = fmt_hist(history)
    
    for pattern, category in critical_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        if matches:
            # Limitar a los más importantes para evitar saturación
            critical_info[category].extend(matches[:3])  
    
    # Formatear información crítica para incluir en el prompt
    critical_sections = []
    for category, items in critical_info.items():
        if items:
            items_text = ', '.join(set(items))  # Eliminar duplicados
            critical_sections.append(f"{category}: {items_text}")
    
    if critical_sections:
        return "\nCRÍTICO: " + "; ".join(critical_sections) + "\n"
    
    return ""


def summarize_if_needed(history: List[Tuple[str, str]], context: str = ""):
    # Usar métrica más precisa considerando el prompt completo
    prompt_size = calculate_prompt_size(history, context)
    if prompt_size <= MAX_HISTORY_CHARS:
        return history
    
    text = fmt_hist(history)
    critical_info = identify_critical_info(history)
    
    prompt = f"""Resume conservando elementos esenciales:
{critical_info}
Conservar: nombres, fechas, números, decisiones, problemas técnicos, contexto del dominio.
Omitir: saludos, conversación casual.

{text}

Resumen:"""
    summary = current_llm.invoke(prompt).content.strip()
    return [("RESUMEN", summary)] + history[-SLIDING_WINDOW_TURNS:]

# Heurística para Python
NEEDS_PYTHON = re.compile(r"(calcula|calcular|media|mediana|desviación|percentil|varianza|promedio|sumatorio|suma|sumar|resta|restar|división|dividir|multiplicación|multiplicar|raíz|potencia|factorial|porcentaje|por\s+ciento|cuánto\s+es|resultado\s+de|operación|elevar|elevar\s+al|cuadrado|cubo)", re.I)


def validate_python_expr(expr: str) -> tuple[bool, str]:
    """Valida expresión Python básica"""
    if not expr.strip():
        return False, "Expresión Python vacía"
    
    if len(expr) > 500:  # Límite más restrictivo para Python
        return False, "Expresión Python demasiado larga"
    
    # Verificar caracteres peligrosos específicos para Python
    dangerous_python = ['import ', '__', 'exec', 'eval', 'open(', 'file(', 'input(', 'raw_input(']
    expr_lower = expr.lower()
    for dangerous in dangerous_python:
        if dangerous in expr_lower:
            return False, "Expresión Python contiene funciones no permitidas"
    
    return True, ""

def extract_numbers_and_lists(text: str):
    """Extrae números y listas del texto de manera más inteligente"""
    # Buscar listas explícitas como [1, 2, 3] o (1, 2, 3)
    list_matches = re.findall(r'[\[\(]([0-9.,\s-]+)[\]\)]', text)
    if list_matches:
        nums = []
        for match in list_matches:
            # Separar por comas o espacios
            parts = re.split(r'[,\s]+', match.strip())
            nums.extend([p.replace(',', '.') for p in parts if p.strip() and re.match(r'-?\d+(?:[.,]\d+)?', p.strip())])
        return nums
    
    # Buscar secuencias como "1, 2, 3, 4" o "números: 10 20 30"
    after_colon = re.search(r':\s*([0-9.,\s-]+)', text)
    if after_colon:
        parts = re.split(r'[,\s]+', after_colon.group(1).strip())
        nums = [p.replace(',', '.') for p in parts if p.strip() and re.match(r'-?\d+(?:[.,]\d+)?', p.strip())]
        if len(nums) > 1:
            return nums
    
    # Extracción básica mejorada
    nums_raw = re.findall(r"-?\d+(?:[.,]\d+)?", text.replace(',', '.'))
    nums = [num.replace(',', '.') for num in nums_raw if num.strip()]
    return nums

def load_image_as_base64(image_path: str) -> str:
    """
    Load an image from the filesystem and convert it to base64
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded image string with data URI format
    """
    try:
        # Handle both absolute paths and relative paths
        if image_path.startswith('/var/data/rag/images/'):
            # It's already an absolute path
            full_path = image_path
        elif image_path.startswith('file://'):
            # Remove file:// prefix
            full_path = image_path.replace('file://', '')
        else:
            # Assume it's relative to the images directory
            full_path = f"/var/data/rag/images/{image_path}"
        
        # Check if file exists
        if not os.path.exists(full_path):
            logger.warning(f"Image file not found: {full_path}")
            return None
            
        # Read and encode the image
        with open(full_path, 'rb') as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Determine mime type based on extension
        ext = Path(full_path).suffix.lower()
        mime_type = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg', 
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.webp': 'image/webp'
        }.get(ext, 'image/png')  # Default to PNG
        
        return f"data:{mime_type};base64,{encoded_image}"
        
    except Exception as e:
        logger.error(f"Failed to load image {image_path}: {str(e)}")
        return None

def extract_cited_documents(response: str) -> list[int]:
    """Extrae los números de documentos que el LLM citó en su respuesta"""
    import re
    # Buscar patrones como "DOCUMENTO 1", "DOCUMENTO 7", etc.
    pattern = r'DOCUMENTO\s+(\d+)'
    matches = re.findall(pattern, response, re.IGNORECASE)
    # Convertir a enteros y eliminar duplicados manteniendo orden
    cited_docs = []
    for match in matches:
        doc_num = int(match)
        if doc_num not in cited_docs:
            cited_docs.append(doc_num)
    return cited_docs

def create_citation_mapping(cited_doc_numbers: list[int]) -> dict[int, int]:
    """Crea un mapeo de números de documento originales a números secuenciales"""
    return {original: sequential for sequential, original in enumerate(cited_doc_numbers, 1)}

def rewrite_document_references(response: str, citation_mapping: dict[int, int]) -> str:
    """Reescribe las referencias de documentos con formato de cita inline [N]"""
    import re
    
    def replace_doc_ref(match):
        doc_num = int(match.group(1))
        if doc_num in citation_mapping:
            return f"[{citation_mapping[doc_num]}]"
        else:
            # Si por alguna razón se referenció un documento no mapeado, mantener original
            return match.group(0)
    
    # Reemplazar "DOCUMENTO X" con "[N]"
    updated_response = re.sub(r'DOCUMENTO\s+(\d+)', replace_doc_ref, response, flags=re.IGNORECASE)
    return updated_response

def generate_citations_for_cited_docs(docs: list, cited_doc_numbers: list[int], citation_mapping: dict[int, int]) -> str:
    """Genera la lista de citas solo para los documentos efectivamente citados"""
    if not cited_doc_numbers:
        return ""
    
    citations = []
    
    for original_doc_num in cited_doc_numbers:
        # Los documentos están indexados desde 0, pero numerados desde 1 en el contexto
        doc_index = original_doc_num - 1
        
        if doc_index < len(docs):
            doc = docs[doc_index]
            title = doc.metadata.get('title', 'Documento desconocido')
            page = doc.metadata.get('page_number', 'N/A')
            modality = doc.metadata.get('modality', 'text')
            
            # Extraer nombre del documento de manera más amigable
            display_title = title.replace('_', ' ').title()
            if len(display_title) > 50:
                display_title = display_title[:47] + "..."
            
            # Obtener el número secuencial de la cita
            sequential_num = citation_mapping[original_doc_num]
            
            # Formato de cita diferenciado según modalidad
            if modality == 'image':
                # Para imágenes, incluir metadatos específicos
                image_index = doc.metadata.get('image_index', 0)
                thumbnail_uri = doc.metadata.get('thumbnail_uri', '')
                source_uri = doc.metadata.get('source_uri', '')
                width = doc.metadata.get('width', 0)
                height = doc.metadata.get('height', 0)
                
                citation = f"[{sequential_num}] Imagen {image_index + 1} en {display_title} (pág. {page}) - {width}x{height}px"
                if thumbnail_uri:
                    citation += f" - thumbnail: {thumbnail_uri}"
            else:
                # Para texto, mantener formato existente
                citation = f"[{sequential_num}] {display_title} (pág. {page})"
            
            citations.append(citation)
    
    # Formato final de citas
    if len(citations) == 1:
        return f"\n\n📖 Fuente consultada:\n{citations[0]}"
    else:
        return f"\n\n📚 Fuentes consultadas:\n" + "\n".join(citations)

def process_inline_citations(response: str, docs: list) -> tuple[str, str]:
    """Procesa la respuesta del LLM para generar citas inline y lista de referencias"""
    # Extraer documentos citados
    cited_doc_numbers = extract_cited_documents(response)
    
    if not cited_doc_numbers:
        logger.info("No document citations found in LLM response")
        return response, ""
    
    logger.info(f"Found citations for documents: {cited_doc_numbers}")
    
    # Crear mapeo secuencial
    citation_mapping = create_citation_mapping(cited_doc_numbers)
    
    # Reescribir referencias con formato inline
    processed_response = rewrite_document_references(response, citation_mapping)
    
    # Generar lista de citas
    citations_text = generate_citations_for_cited_docs(docs, cited_doc_numbers, citation_mapping)
    
    logger.info(f"Citation mapping: {citation_mapping}")
    
    return processed_response + citations_text, citations_text

def maybe_python(user_text: str):
    if user_text.lower().startswith("python:"):
        expr = user_text.split(":", 1)[1].strip()
        is_valid, error_msg = validate_python_expr(expr)
        if not is_valid:
            logger.warning(f"Invalid Python expression: {error_msg}")
            return True, f"# Error: {error_msg}"
        return True, expr
        
    if NEEDS_PYTHON.search(user_text) and re.search(r"\d", user_text):
        # Usar la nueva función de extracción mejorada
        nums = extract_numbers_and_lists(user_text)
        if nums:
            if "percentil" in user_text.lower():
                m = re.search(r"percentil\s*(\d+)", user_text.lower())
                if m:
                    p = int(m.group(1))
                    # Usar enfoque más directo para percentiles
                    if p == 50:
                        return True, f"statistics.median([{', '.join(nums)}])"
                    elif p == 25:
                        return True, f"statistics.quantiles([{', '.join(nums)}], n=4)[0]"  # Q1
                    elif p == 75:
                        return True, f"statistics.quantiles([{', '.join(nums)}], n=4)[2]"  # Q3
                    else:
                        # Para otros percentiles, usar fórmula simple
                        return True, f"sorted([{', '.join(nums)}])[min(len([{', '.join(nums)}]) - 1, max(0, int(len([{', '.join(nums)}]) * {p/100})))]"
            if "media" in user_text.lower() or "promedio" in user_text.lower():
                return True, f"statistics.mean([{', '.join(nums)}])"
            if "mediana" in user_text.lower():
                return True, f"statistics.median([{', '.join(nums)}])"
            if "desvi" in user_text.lower():
                return True, f"statistics.pstdev([{', '.join(nums)}])"
            # Manejar operaciones básicas
            if any(word in user_text.lower() for word in ["suma", "sumar"]):
                return True, f"sum([{', '.join(nums)}])"
            if any(word in user_text.lower() for word in ["multiplicación", "multiplicar"]):
                if len(nums) >= 2:
                    return True, f"{' * '.join(nums)}"
            if any(word in user_text.lower() for word in ["división", "dividir"]):
                if len(nums) >= 2:
                    return True, f"{nums[0]} / {nums[1]}"
            if any(word in user_text.lower() for word in ["resta", "restar"]):
                if len(nums) >= 2:
                    return True, f"{nums[0]} - {nums[1]}"
            # Manejar operaciones especiales
            if any(word in user_text.lower() for word in ["porcentaje", "por ciento"]):
                if len(nums) >= 2:
                    return True, f"({nums[0]} / {nums[1]}) * 100"
            if "cuadrado" in user_text.lower():
                if len(nums) >= 1:
                    return True, f"{nums[0]} ** 2"
            if "cubo" in user_text.lower():
                if len(nums) >= 1:
                    return True, f"{nums[0]} ** 3"
            if "raíz" in user_text.lower():
                if len(nums) >= 1:
                    return True, f"math.sqrt({nums[0]})"
            if any(word in user_text.lower() for word in ["potencia", "elevar"]):
                if len(nums) >= 2:
                    return True, f"{nums[0]} ** {nums[1]}"
            if "factorial" in user_text.lower():
                if len(nums) >= 1:
                    return True, f"math.factorial({nums[0]})"
    return False, ""

# RAG

def answer_with_rag(query: str, history: List[Tuple[str, str]]):
    global rag_request_count, rag_latency_sum, error_count
    
    logger.info(f"RAG query initiated - query_length: {len(query)}")
    rag_request_count += 1
    
    # Validar entrada
    is_valid, error_msg = validate_input(query, "consulta")
    if not is_valid:
        logger.warning(f"RAG query validation failed: {error_msg}")
        error_count += 1
        return f"Error en la consulta: {error_msg}", 0.0
    
    # Sanitizar entrada
    query = sanitize_input(query)
    
    try:
        # Obtener estadísticas de caché antes de la consulta (deshabilitado temporalmente)
        cache_stats_before = {"hits": 0, "misses": 0}
        
        docs = retriever.get_relevant_documents(query)
        
        # Aplicar reranking si está habilitado
        rerank_latency_ms = 0.0
        if reranker and len(docs) > 1:
            try:
                logger.info(f"Applying reranking: {len(docs)} → {RERANK_TOP_K} docs")
                reranked_docs, rerank_latency_ms = reranker.rerank_doc_objects(
                    query=query,
                    documents=docs,
                    top_n=RERANK_TOP_K
                )
                docs = reranked_docs
                logger.info(f"Reranking completed: {len(docs)} docs selected, latency: {rerank_latency_ms:.2f}ms")
                
                # Log reranking scores for debugging
                for i, doc in enumerate(docs[:3], 1):
                    score = doc.metadata.get('rerank_score', 'N/A')
                    title = doc.metadata.get('title', 'Unknown')[:50]
                    logger.debug(f"Reranked doc {i}: {title} (score: {score})")
                    
            except Exception as e:
                logger.error(f"Reranking failed, using original order: {str(e)}")
                docs = docs[:RERANK_TOP_K]  # Fallback: truncar sin rerank
        elif len(docs) > RERANK_TOP_K:
            # Si reranker está deshabilitado, truncar a RERANK_TOP_K
            docs = docs[:RERANK_TOP_K]
        
        # Obtener estadísticas después y detectar si fue cache hit (deshabilitado temporalmente)
        cache_stats_after = {"hits": 0, "misses": 0, "hit_rate": 0.0}
        was_cache_hit = False
        
        rerank_info = f", rerank_latency: {rerank_latency_ms:.2f}ms" if rerank_latency_ms > 0 else ""
        logger.info(f"Retrieved {len(docs)} documents for RAG query - Cache {'hit' if was_cache_hit else 'miss'} (hit rate: {cache_stats_after['hit_rate'] * 100:.1f}%){rerank_info}")
        
        # Construir contexto optimizado y recopilar imágenes
        context_parts = []
        image_documents = []  # Para rastrear documentos con imágenes
        
        for i, d in enumerate(docs):
            doc_title = d.metadata.get('title', 'Documento desconocido')
            page_num = d.metadata.get('page_number', 'N/A')
            modality = d.metadata.get('modality', 'text')
            
            # Formato diferenciado según modalidad
            if modality == 'image':
                # Para imágenes, incluir metadatos completos
                thumbnail_uri = d.metadata.get('thumbnail_uri', '')
                image_index = d.metadata.get('image_index', 0)
                width = d.metadata.get('width', 0)
                height = d.metadata.get('height', 0)
                bbox = d.metadata.get('bbox', {})
                source_uri = d.metadata.get('source_uri', '')
                
                # Guardar información de imagen para procesamiento multimodal
                # NOTA: thumbnail_uri contiene la ruta del PNG original (no thumbnail)
                # source_uri solo contiene el nombre del PDF
                image_documents.append({
                    'doc_index': i + 1,
                    'thumbnail_uri': thumbnail_uri,  # Este es realmente el PNG original
                    'source_uri': source_uri,
                    'title': doc_title,
                    'page': page_num,
                    'image_index': image_index
                })
                
                context_part = f"""DOCUMENTO {i+1}: {doc_title}
PÁGINA: {page_num}
TIPO: IMAGEN {image_index + 1}
DIMENSIONES: {width}x{height}px
UBICACIÓN: {thumbnail_uri}
FUENTE: {source_uri}
CONTENIDO: {d.page_content.strip()}
---"""
            else:
                # Para texto, mantener formato existente
                context_part = f"""DOCUMENTO {i+1}: {doc_title}
PÁGINA: {page_num}
CONTENIDO:
{d.page_content.strip()}
---"""
            
            context_parts.append(context_part)
        
        context = "\n\n".join(context_parts)
        
        # Sistema prompt optimizado
        sys = """Eres un asistente experto que responde preguntas basándose en el contexto proporcionado.

INSTRUCCIONES IMPORTANTES:
1. Lee TODO el contexto cuidadosamente antes de responder
2. Busca información relevante en TODOS los documentos proporcionados  
3. Usa conceptos relacionados y sinónimos - por ejemplo, "interpretabilidad", "transparencia", "explicación" están relacionados con "explicabilidad"
4. Si encuentras información parcial o conceptos relacionados, úsalos para construir una respuesta útil
5. SIEMPRE cita las fuentes específicas donde encontraste la información usando el formato DOCUMENTO X
6. Solo responde "No hay información suficiente" si realmente NO HAY nada relacionado con la pregunta

REFERENCIAS A IMÁGENES:
7. Cuando encuentres documentos marcados como "TIPO: IMAGEN", puedes referenciarlos igual que el texto usando DOCUMENTO X
8. Las imágenes pueden contener diagramas, gráficos, fotografías o ilustraciones relevantes para la consulta
9. Describe brevemente el contenido visual basándote en los metadatos proporcionados (dimensiones, ubicación)
10. Menciona imágenes cuando sean relevantes para complementar o ilustrar tu respuesta

IMPORTANTE: Tu objetivo es ser útil y proporcionar información valiosa basada en el contexto disponible."""

        hist_text = fmt_hist(history[-SLIDING_WINDOW_TURNS:], enhanced=True)
        
        # Preparar contenido multimodal si hay imágenes
        if image_documents:
            logger.info(f"Preparing multimodal RAG with {len(image_documents)} images")
            
            # Construir contenido con imágenes
            message_content = []
            
            # Agregar el prompt principal como texto
            text_prompt = f"""{sys}

Contexto:
{context}

Historial:
{hist_text}

Pregunta: {query}

Por favor analiza tanto el texto como las imágenes proporcionadas. Si las imágenes contienen gráficos, tablas o diagramas relevantes para la pregunta, descríbelos y úsalos en tu respuesta.

Respuesta:"""
            
            message_content.append({
                "type": "text",
                "text": text_prompt
            })
            
            # Cargar y agregar imágenes (limitado a las primeras 5 más relevantes)
            images_added = 0
            max_images = 5  # Límite para evitar sobrecarga
            
            for img_doc in image_documents[:max_images]:
                # IMPORTANTE: thumbnail_uri contiene la ruta del PNG original (no es un thumbnail)
                # source_uri solo contiene el nombre del PDF, no la ruta de la imagen
                # Por lo tanto, usar thumbnail_uri que tiene la ruta correcta del PNG
                image_path = img_doc['thumbnail_uri']  # Este es el PNG original de alta resolución
                if image_path:
                    logger.debug(f"Loading image: {image_path}")
                    image_base64 = load_image_as_base64(image_path)
                    
                    if image_base64:
                        message_content.append({
                            "type": "image_url",
                            "image_url": image_base64
                        })
                        images_added += 1
                        logger.info(f"Added image {images_added}: Doc {img_doc['doc_index']}, Page {img_doc['page']}")
                    else:
                        logger.warning(f"Failed to load image: {image_path}")
            
            logger.info(f"Multimodal message prepared with {images_added} images")
            
            # Crear mensaje multimodal para Gemini
            human_message = HumanMessage(content=message_content)
            
            t0 = time.time()
            logger.info("Invoking LLM for multimodal RAG response")
            
            # Debug logs
            logger.debug(f"Context length: {len(context)} chars, docs: {len(docs)}, images: {images_added}")
            logger.debug(f"Query: {query}")
            
            # Invocar con mensaje multimodal
            resp = current_llm.invoke([human_message]).content.strip()
            
        else:
            # Procesamiento tradicional solo texto
            prompt = f"""{sys}

Contexto:
{context}

Historial:
{hist_text}

Pregunta: {query}

Respuesta:"""
            
            t0 = time.time()
            logger.info("Invoking LLM for text-only RAG response")
            
            # Debug: Log context size and first part
            logger.debug(f"Context length: {len(context)} chars, docs: {len(docs)}")
            
            # Log what documents were retrieved
            for i, doc in enumerate(docs[:5], 1):
                page = doc.metadata.get('page_number', 'N/A')
                title = doc.metadata.get('title', 'Unknown')
                content_preview = doc.page_content[:100] if hasattr(doc, 'page_content') else 'No content'
                logger.debug(f"Doc {i}: Page {page}, Title: {title}, Content: {content_preview}...")
            
            logger.debug(f"Query: {query}")
            
            resp = current_llm.invoke(prompt).content.strip()
        latency = time.time() - t0
        rag_latency_sum += latency
        
        # Debug: Log response
        logger.debug(f"LLM response: {resp[:200]}")
        logger.info(f"RAG response generated - latency: {latency:.3f}s, response_length: {len(resp)}")
        
        # Procesar citas inline y generar referencias correspondientes
        processed_response, final_citations = process_inline_citations(resp, docs)
        
        return processed_response, latency
    except Exception as e:
        error_count += 1
        logger.error(f"RAG operation failed - error: {str(e)}")
        return "Lo siento, ha ocurrido un error al procesar su consulta. Intente nuevamente.", 0.0

# Llamada al microservicio pyexec

def generate_correlation_id() -> str:
    """Genera un ID de correlación único para rastrear requests"""
    return str(uuid.uuid4())[:8]

def run_pyexpr(expr: str, correlation_id: str = None):
    global python_request_count, error_count
    
    if correlation_id is None:
        correlation_id = generate_correlation_id()
    
    logger.info(f"Python execution request - correlation_id: {correlation_id}, expr_length: {len(expr)}")
    python_request_count += 1
    
    try:
        # Enviar request con headers de correlation
        headers = {
            'Content-Type': 'application/json',
            'X-Correlation-ID': correlation_id
        }
        
        # Usar sesión optimizada para computación con timeout reducido
        r = compute_session.post(
            f"{PYEXEC_URL}/run", 
            json={"expr": expr}, 
            headers=headers,
            timeout=HTTP_TIMEOUT_NORMAL
        )
        
        if r.status_code == 200:
            response_data = r.json()
            if response_data.get("ok", False):
                result = response_data["result"]
                logger.info(f"Python execution successful - correlation_id: {correlation_id}, result_type: {type(result).__name__}")
                return result
            else:
                # Manejar error estructurado del servicio
                error_info = response_data.get("error", {})
                error_message = error_info.get("message", "Error desconocido")
                error_category = error_info.get("category", "unknown")
                logger.warning(f"Python execution error - correlation_id: {correlation_id}, category: {error_category}")
                error_count += 1
                return f"Error: {error_message}"
        else:
            # Manejar errores HTTP
            try:
                error_data = r.json()
                if "detail" in error_data and isinstance(error_data["detail"], dict):
                    error_info = error_data["detail"].get("error", {})
                    error_message = error_info.get("message", "Error en el servicio de Python")
                    logger.warning(f"Python service HTTP error - correlation_id: {correlation_id}, status: {r.status_code}")
                    return f"Error: {error_message}"
                else:
                    logger.error(f"Python service HTTP error - correlation_id: {correlation_id}, status: {r.status_code}")
                    return f"Error: El servicio de Python respondió con código {r.status_code}"
            except:
                logger.error(f"Python service HTTP error - correlation_id: {correlation_id}, status: {r.status_code}")
                return f"Error: El servicio de Python no está disponible"
                
    except requests.exceptions.Timeout:
        error_count += 1
        logger.error(f"Python execution timeout - correlation_id: {correlation_id}")
        return "Error: El cálculo está tardando demasiado tiempo"
    except requests.exceptions.ConnectionError:
        error_count += 1
        logger.error(f"Python service connection error - correlation_id: {correlation_id}")
        return "Error: No se puede conectar al servicio de Python"
    except Exception as e:
        error_count += 1
        logger.error(f"Python execution failed - correlation_id: {correlation_id}, error: {str(e)}")
        return f"Error al ejecutar Python: {str(e)}"


def chat(user_text, state):
    global request_count, error_count
    
    correlation_id = generate_correlation_id()
    logger.info(f"Chat request received - correlation_id: {correlation_id}, user_text_length: {len(user_text)}")
    request_count += 1
    
    # Validar entrada inicial
    is_valid, error_msg = validate_input(user_text, "mensaje")
    if not is_valid:
        logger.warning(f"Chat input validation failed - correlation_id: {correlation_id}, error: {error_msg}")
        error_count += 1
        history = state or []
        error_response = f"Error: {error_msg}"
        history.append((user_text, error_response))
        return history, history
    
    # Sanitizar entrada
    user_text = sanitize_input(user_text)
    
    history = state or []
    use_py, expr = maybe_python(user_text)
    
    if use_py:
        logger.info(f"Python execution mode triggered - correlation_id: {correlation_id}, expr: {expr[:50]}...")
        if expr.startswith("# Error:"):
            # Error en validación de Python
            answer = f"Resultado de Python: {expr}"
        else:
            result = run_pyexpr(expr, correlation_id)
            # Generar respuesta más contextual basada en la operación
            if "statistics.mean" in expr:
                answer = f"La media/promedio de los números es: {result}"
            elif "statistics.median" in expr:
                answer = f"La mediana de los números es: {result}"
            elif "sum(" in expr:
                answer = f"La suma total es: {result}"
            elif "math.sqrt" in expr:
                answer = f"La raíz cuadrada es: {result}"
            elif "statistics.pstdev" in expr:
                answer = f"La desviación estándar es: {result}"
            elif "**" in expr and "2" in expr:
                answer = f"El resultado al cuadrado es: {result}"
            elif "**" in expr and "3" in expr:
                answer = f"El resultado al cubo es: {result}"
            elif "math.factorial" in expr:
                answer = f"El factorial es: {result}"
            elif "/" in expr:
                answer = f"El resultado de la división es: {result}"
            elif "*" in expr:
                answer = f"El resultado de la multiplicación es: {result}"
            elif "percentil" in user_text.lower():
                answer = f"El percentil solicitado es: {result}"
            else:
                answer = f"Resultado del cálculo: {result}"
    else:
        logger.info(f"RAG mode triggered - correlation_id: {correlation_id}")
        # Obtener contexto para métricas precisas (sin hacer retrieval completo)
        try:
            docs = retriever.get_relevant_documents(user_text)
            context_parts = []
            for i, d in enumerate(docs):
                doc_title = d.metadata.get('title', 'Documento desconocido')
                page_num = d.metadata.get('page_number', 'N/A')
                context_part = f"""[{i+1}] {doc_title} (p.{page_num}):
{d.page_content.strip()}"""
                context_parts.append(context_part)
            context_estimate = "\n\n".join(context_parts)
        except:
            context_estimate = ""
        
        history = summarize_if_needed(history, context_estimate)
        answer, _lat = answer_with_rag(user_text, history)
    
    history.append((user_text, answer))
    logger.info(f"Chat response completed - correlation_id: {correlation_id}, history_length: {len(history)}")
    return history, history

with gr.Blocks() as demo:
    gr.Markdown("## Chatbot RAG (microservicios) • Qdrant + Google Gemini + PyExec")
    
    # Información del modelo (solo informativo, no editable)
    gr.Markdown(f"**🤖 Modelo**: {current_model} • **⚡ Proveedor**: Google Gemini")
    
    chatbox = gr.Chatbot(height=420)
    msg = gr.Textbox(placeholder="Pregunta aquí... (o 'python: 1+2')", autofocus=True)
    state = gr.State([])


    def respond(message, history):
        new_hist, state_out = chat(message, history)
        return new_hist, state_out, ""

    msg.submit(respond, [msg, state], [chatbox, state, msg])

if __name__ == "__main__":
    # Iniciar el servidor de health checks en un thread separado
    health_thread = Thread(target=start_health_server, daemon=True)
    health_thread.start()
    
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        show_api=False,
    )
