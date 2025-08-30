"""
Jina Embeddings v4 wrapper compatible con LangChain
Proporciona soporte completo para el API de Jina con task adapters, normalización y late chunking
"""

import os
import time
import logging
import requests
from typing import List, Dict, Any, Optional, Union
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class JinaEmbeddings:
    """
    Wrapper para Jina Embeddings v4 API compatible con LangChain.
    
    Soporta:
    - Task adapters (retrieval.query, retrieval.passage, text-matching, code.*)
    - Dimensiones configurables (128-2048) con Matryoshka Representation Learning
    - Normalización de vectores para uso con distance=DOT en Qdrant
    - Late chunking para documentos largos
    - Retry logic con backoff exponencial
    - Rate limiting (500 RPM, 1M TPM)
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "jina-embeddings-v4",
        dimensions: int = 1024,
        normalized: bool = True,
        late_chunking: bool = False,
        embedding_type: str = "float",
        task_type: str = "retrieval.passage",
        max_retries: int = 3,
        timeout: int = 60,
        rate_limit_rpm: int = 500,
        rate_limit_tpm: int = 1000000,
        **kwargs
    ):
        """
        Inicializa el cliente de Jina Embeddings v4
        
        Args:
            api_key: Jina API key (si no se proporciona, lee de JINA_API_KEY)
            model: Modelo a usar (default: jina-embeddings-v4)
            dimensions: Dimensiones del embedding (128-2048, default: 1024)
            normalized: Si normalizar los vectores (recomendado para DOT distance)
            late_chunking: Habilitar late chunking para documentos largos
            embedding_type: Tipo de embedding ("float", "binary", "base64")
            task_type: Task adapter por defecto
            max_retries: Número máximo de reintentos
            timeout: Timeout en segundos para requests HTTP
            rate_limit_rpm: Rate limit requests por minuto
            rate_limit_tpm: Rate limit tokens por minuto
        """
        self.api_key = api_key or os.getenv("JINA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "JINA_API_KEY es requerida. Configúrala en las variables de entorno "
                "o pásala como parámetro api_key."
            )
        
        self.model = model
        self.dimensions = dimensions
        self.normalized = normalized
        self.late_chunking = late_chunking
        self.embedding_type = embedding_type
        self.task_type = task_type
        self.max_retries = max_retries
        self.timeout = timeout
        self.rate_limit_rpm = rate_limit_rpm
        self.rate_limit_tpm = rate_limit_tpm
        
        # Validar parámetros
        if not (128 <= dimensions <= 2048):
            raise ValueError("Las dimensiones deben estar entre 128 y 2048")
        
        if embedding_type not in ["float", "binary", "base64"]:
            raise ValueError("embedding_type debe ser 'float', 'binary' o 'base64'")
        
        # URL del API
        self.api_url = "https://api.jina.ai/v1/embeddings"
        
        # Headers por defecto
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "JinaEmbeddings-Python/1.0"
        }
        
        # Configurar sesión HTTP con retry logic
        self.session = self._create_session()
        
        # Rate limiting
        self._last_request_time = 0
        self._request_count = 0
        self._token_count = 0
        self._minute_start = time.time()
        
        logger.info(f"Initialized Jina Embeddings: model={model}, dims={dimensions}, "
                   f"normalized={normalized}, late_chunking={late_chunking}")
    
    def _create_session(self) -> requests.Session:
        """Crear sesión HTTP con retry logic optimizado"""
        session = requests.Session()
        
        # Configurar retry strategy
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=1,  # 1s, 2s, 4s
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"],
            raise_on_status=False  # Manejar status codes manualmente
        )
        
        # Crear adapter
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        
        return session
    
    def _rate_limit(self, token_count: int):
        """Aplicar rate limiting básico"""
        current_time = time.time()
        
        # Reset counters cada minuto
        if current_time - self._minute_start >= 60:
            self._request_count = 0
            self._token_count = 0
            self._minute_start = current_time
        
        # Verificar límites
        if self._request_count >= self.rate_limit_rpm:
            sleep_time = 60 - (current_time - self._minute_start)
            if sleep_time > 0:
                logger.warning(f"Rate limit reached, sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
                self._request_count = 0
                self._token_count = 0
                self._minute_start = time.time()
        
        if self._token_count + token_count >= self.rate_limit_tpm:
            sleep_time = 60 - (current_time - self._minute_start)
            if sleep_time > 0:
                logger.warning(f"Token limit reached, sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
                self._request_count = 0
                self._token_count = 0
                self._minute_start = time.time()
        
        # Actualizar contadores
        self._request_count += 1
        self._token_count += token_count
    
    def _make_request(
        self, 
        inputs: List[Union[str, Dict]], 
        task: str,
        late_chunking: Optional[bool] = None
    ) -> List[List[float]]:
        """Hacer request al API de Jina con manejo de errores"""
        
        # Estimar tokens para rate limiting (aproximado)
        estimated_tokens = sum(len(inp) // 4 if isinstance(inp, str) 
                             else len(inp.get("text", "")) // 4 for inp in inputs)
        
        # Aplicar rate limiting
        self._rate_limit(estimated_tokens)
        
        # Preparar payload mínimo (solo campos básicos requeridos)
        payload = {
            "model": self.model,
            "input": inputs,
            "dimensions": self.dimensions  # Especificar dimensiones siempre
        }
        
        # Añadir task solo si es válido
        if task:
            payload["task"] = task
        
        try:
            logger.debug(f"Making request to Jina API: {len(inputs)} inputs, task={task}")
            response = self.session.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=self.timeout
            )
            
            # Manejar rate limiting específico de Jina
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 60))
                logger.warning(f"Rate limited by API, waiting {retry_after}s")
                time.sleep(retry_after)
                # Reintento después del rate limit
                response = self.session.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout
                )
            
            # Verificar respuesta exitosa
            response.raise_for_status()
            
            # Parsear respuesta
            data = response.json()
            if "data" not in data:
                raise ValueError(f"Respuesta inválida del API: {data}")
            
            # Extraer embeddings
            embeddings = [item["embedding"] for item in data["data"]]
            
            logger.debug(f"Successfully got {len(embeddings)} embeddings")
            return embeddings
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error en request a Jina API: {e}")
            if hasattr(e.response, 'text'):
                logger.error(f"Response: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error procesando respuesta de Jina API: {e}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generar embeddings para documentos (usar task retrieval.passage)
        
        Args:
            texts: Lista de textos para embebir
            
        Returns:
            Lista de embeddings (vectores de dimensión self.dimensions)
        """
        if not texts:
            return []
        
        # Convertir textos a formato esperado por el API
        inputs = [{"text": text} for text in texts]
        
        # Usar retrieval.passage para documentos y habilitar late_chunking
        return self._make_request(inputs, "retrieval.passage", late_chunking=True)
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generar embedding para una query (usar task retrieval.query)
        
        Args:
            text: Texto de la consulta
            
        Returns:
            Embedding (vector de dimensión self.dimensions)
        """
        if not text:
            raise ValueError("El texto de consulta no puede estar vacío")
        
        # Usar retrieval.query para consultas (sin late_chunking)
        inputs = [{"text": text}]
        embeddings = self._make_request(inputs, "retrieval.query", late_chunking=False)
        
        return embeddings[0] if embeddings else []
    
    def embed_texts(
        self, 
        texts: List[str], 
        task: Optional[str] = None
    ) -> List[List[float]]:
        """
        Método genérico para embebir textos con task específico
        
        Args:
            texts: Lista de textos
            task: Task adapter específico (si None, usa self.task_type)
            
        Returns:
            Lista de embeddings
        """
        if not texts:
            return []
        
        task = task or self.task_type
        inputs = [{"text": text} for text in texts]
        
        return self._make_request(inputs, task)
    
    def embed_images(self, image_paths: List[str]) -> List[List[float]]:
        """
        Generar embeddings para imágenes usando Jina Embeddings v4
        
        Args:
            image_paths: Lista de rutas a archivos de imagen
            
        Returns:
            Lista de embeddings (vectores de dimensión self.dimensions)
        """
        import base64
        
        if not image_paths:
            return []
        
        inputs = []
        for image_path in image_paths:
            try:
                with open(image_path, "rb") as image_file:
                    # Codificar imagen en base64
                    image_data = base64.b64encode(image_file.read()).decode('utf-8')
                    inputs.append({"image": image_data})
            except Exception as e:
                logger.error(f"Error reading image {image_path}: {e}")
                raise ValueError(f"No se pudo leer la imagen: {image_path}")
        
        # Usar el mismo task que para documentos (retrieval.passage)
        return self._make_request(inputs, "retrieval.passage", late_chunking=False)
    
    def embed_images_data(self, image_data_list: List[bytes]) -> List[List[float]]:
        """
        Generar embeddings para datos de imágenes en memoria
        
        Args:
            image_data_list: Lista de datos de imagen en bytes
            
        Returns:
            Lista de embeddings (vectores de dimensión self.dimensions)
        """
        import base64
        
        if not image_data_list:
            return []
        
        inputs = []
        for image_data in image_data_list:
            try:
                # Codificar imagen en base64
                image_b64 = base64.b64encode(image_data).decode('utf-8')
                inputs.append({"image": image_b64})
            except Exception as e:
                logger.error(f"Error encoding image data: {e}")
                raise ValueError(f"No se pudieron codificar los datos de imagen")
        
        # Usar el mismo task que para documentos (retrieval.passage)
        return self._make_request(inputs, "retrieval.passage", late_chunking=False)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Obtener información del modelo configurado"""
        return {
            "model": self.model,
            "dimensions": self.dimensions,
            "normalized": self.normalized,
            "late_chunking": self.late_chunking,
            "embedding_type": self.embedding_type,
            "task_type": self.task_type,
            "max_input_tokens": 32000,  # Jina v4 soporta 32K tokens
            "api_url": self.api_url
        }
    
    def __repr__(self) -> str:
        return (f"JinaEmbeddings(model='{self.model}', dimensions={self.dimensions}, "
                f"normalized={self.normalized}, task='{self.task_type}')")


# Función de conveniencia para crear instancia con configuración común
def create_jina_embeddings(
    task_type: str = "retrieval.passage",
    dimensions: int = 1024,
    **kwargs
) -> JinaEmbeddings:
    """
    Crear instancia de JinaEmbeddings con configuración optimizada
    
    Args:
        task_type: Tipo de task (retrieval.passage, retrieval.query, etc.)
        dimensions: Dimensiones del embedding
        **kwargs: Parámetros adicionales
        
    Returns:
        Instancia configurada de JinaEmbeddings
    """
    return JinaEmbeddings(
        task_type=task_type,
        dimensions=dimensions,
        normalized=True,  # Siempre normalizar para Qdrant DOT
        **kwargs
    )