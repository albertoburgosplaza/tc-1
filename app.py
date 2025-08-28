
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

from langchain_community.vectorstores import Qdrant
from langchain_google_genai import ChatGoogleGenerativeAI
from qdrant_client import QdrantClient
from embedding_factory import EmbeddingFactory

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
# Obtener modelo de variables de entorno o usar Gemini por defecto
embedding_model = os.getenv("EMBEDDING_MODEL", "models/embedding-001")
embedding_provider = os.getenv("EMBEDDING_PROVIDER", "google")

logger.info(f"Initializing embeddings: {embedding_model} ({embedding_provider})")
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
        """Get relevant documents with proper metadata handling"""
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
        
        # Convert to LangChain documents
        documents = []
        for result in search_result.points:
            # Extract metadata and content from payload
            payload = result.payload.copy()  # Make a copy to avoid modifying original
            page_content = payload.pop('page_content', '')  # Remove page_content from metadata
            
            # Create document with proper metadata
            doc = Document(
                page_content=page_content,
                metadata=payload  # All other fields as metadata
            )
            documents.append(doc)
        
        return documents

base_retriever = CustomQdrantRetriever(client, COLLECTION, emb, k=15)

# Crear retriever paralelo que envuelve el retriever base
retriever = ParallelRetriever(base_retriever, max_workers=MAX_RETRIEVAL_WORKERS)

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
        "ollama": check_service_health(OLLAMA_URL, "/api/tags"),
        "pyexec": check_service_health(PYEXEC_URL, "/health")
    }
    
    # Calcular latencia promedio de RAG
    avg_rag_latency = (rag_latency_sum / rag_request_count) if rag_request_count > 0 else 0.0
    
    # Obtener estadísticas de caché de embeddings (deshabilitado temporalmente)
    cache_stats = {"hits": 0, "misses": 0, "hit_rate": 0.0, "current_size": 0, "max_size": 200}
    
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
        
        # Obtener estadísticas después y detectar si fue cache hit (deshabilitado temporalmente)
        cache_stats_after = {"hits": 0, "misses": 0, "hit_rate": 0.0}
        was_cache_hit = False
        
        logger.info(f"Retrieved {len(docs)} documents for RAG query - Cache {'hit' if was_cache_hit else 'miss'} (hit rate: {cache_stats_after['hit_rate'] * 100:.1f}%)")
        
        # Construir contexto optimizado
        context_parts = []
        for i, d in enumerate(docs):
            doc_title = d.metadata.get('title', 'Documento desconocido')
            page_num = d.metadata.get('page', 'N/A')
            
            # Formato más claro para el contexto
            context_part = f"""DOCUMENTO {i+1}: {doc_title}
PÁGINA: {page_num}
CONTENIDO:
{d.page_content.strip()}
---"""
            
            context_parts.append(context_part)
        
        context = "\n\n".join(context_parts)
        
        # Sistema prompt optimizado
        sys = """Eres un asistente experto que responde preguntas basándose ÚNICAMENTE en el contexto proporcionado. 

INSTRUCCIONES IMPORTANTES:
1. Lee TODO el contexto cuidadosamente antes de responder
2. Busca información relevante en TODOS los documentos proporcionados
3. Si encuentras nombres de empresas, listados, porcentajes o cualquier dato específico, DEBES mencionarlos
4. Solo responde "No hay información suficiente" si después de leer TODO el contexto no encuentras NADA relacionado
5. SIEMPRE cita las fuentes específicas donde encontraste la información
6. Si la pregunta es sobre empresas del S&P 500 y ves una lista de "POSICIONES" o empresas con porcentajes, DEBES listarlas

RECUERDA: Tu trabajo es extraer y presentar TODA la información relevante del contexto."""

        hist_text = fmt_hist(history[-SLIDING_WINDOW_TURNS:], enhanced=True)
        prompt = f"""{sys}

Contexto:
{context}

Historial:
{hist_text}

Pregunta: {query}

Respuesta:"""
        
        t0 = time.time()
        logger.info("Invoking LLM for RAG response")
        
        # Debug: Log context size and first part
        logger.debug(f"Context length: {len(context)} chars, docs: {len(docs)}")
        
        # Log what documents were retrieved
        for i, doc in enumerate(docs[:5], 1):
            page = doc.metadata.get('page', 'N/A')
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
        
        # Generar citas mejoradas con ranking de relevancia (top 1-3 fuentes)
        top_sources = docs[:3]  # Limitamos a top 3 fuentes más relevantes
        citations = []
        
        for i, doc in enumerate(top_sources, 1):
            title = doc.metadata.get('title', 'Documento desconocido')
            page = doc.metadata.get('page', 'N/A')
            source = doc.metadata.get('source', title)
            
            # Extraer nombre del documento de manera más amigable
            display_title = title.replace('_', ' ').title()
            if len(display_title) > 50:
                display_title = display_title[:47] + "..."
            
            # Formato mejorado de cita con ranking
            citation = f"[{i}] {display_title} (pág. {page})"
            citations.append(citation)
        
        # Formato final de citas
        if citations:
            if len(citations) == 1:
                cite_text = f"\n\n📖 Fuente: {citations[0]}"
            else:
                cite_text = f"\n\n📚 Fuentes consultadas:\n" + "\n".join(f"   {cite}" for cite in citations)
        else:
            cite_text = ""
        
        return resp + cite_text, latency
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
                page_num = d.metadata.get('page', 'N/A')
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
