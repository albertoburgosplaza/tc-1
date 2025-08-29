"""
Jina Reranker m0 (Multimodal) wrapper para RAG
Proporciona reranking multimodal de documentos con soporte para texto e imágenes
"""

import os
import time
import random
import logging
import requests
import re
import base64
from typing import List, Dict, Any, Optional, Union, Tuple
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class JinaReranker:
    """
    Wrapper para Jina Reranker m0 API con soporte multimodal.
    
    Características:
    - Soporte para queries y documentos de texto e imagen
    - Chunking automático para >2048 documentos
    - Retry logic con backoff exponencial y jitter
    - Normalización inteligente de formatos de entrada
    - Fallback seguro ante errores
    - Métricas detalladas de rendimiento
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "jina-reranker-m0",
        timeout: int = 20,
        max_retries: int = 3,
        backoff_base: float = 1.0,
        max_chunk_size: int = 2048,
        **kwargs
    ):
        """
        Inicializa el cliente de Jina Reranker m0
        
        Args:
            api_key: Jina API key (si no se proporciona, lee de JINA_API_KEY)
            model: Modelo a usar (default: jina-reranker-m0)
            timeout: Timeout en segundos para requests HTTP
            max_retries: Número máximo de reintentos
            backoff_base: Base para backoff exponencial
            max_chunk_size: Máximo número de documentos por chunk (API limit: 2048)
        """
        self.api_key = api_key or os.getenv("JINA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "JINA_API_KEY es requerida. Configúrala en las variables de entorno "
                "o pásala como parámetro api_key."
            )
        
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.max_chunk_size = max_chunk_size
        
        # URL del API
        self.api_url = "https://api.jina.ai/v1/rerank"
        
        # Headers por defecto
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "JinaReranker-Python/1.0"
        }
        
        # Configurar sesión HTTP con retry logic
        self.session = self._create_session()
        
        # Métricas
        self.reset_metrics()
        
        logger.info(f"Initialized Jina Reranker m0: model={model}, timeout={timeout}s, "
                   f"max_chunk_size={max_chunk_size}")
    
    def reset_metrics(self):
        """Reset métricas de rendimiento"""
        self.metrics = {
            "total_requests": 0,
            "total_documents_processed": 0,
            "total_latency_ms": 0.0,
            "error_count": 0,
            "fallback_count": 0,
            "multimodal_queries": 0,
            "multimodal_docs": 0,
            "chunks_processed": 0
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Obtener métricas actuales"""
        metrics = self.metrics.copy()
        if metrics["total_requests"] > 0:
            metrics["avg_latency_ms"] = metrics["total_latency_ms"] / metrics["total_requests"]
            metrics["error_rate"] = metrics["error_count"] / metrics["total_requests"]
        else:
            metrics["avg_latency_ms"] = 0.0
            metrics["error_rate"] = 0.0
        return metrics
    
    def _create_session(self) -> requests.Session:
        """Crear sesión HTTP con retry logic optimizado"""
        session = requests.Session()
        
        # Configurar retry strategy
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=self.backoff_base,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"],
            raise_on_status=False  # Manejar status codes manualmente
        )
        
        # Crear adapter
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        
        return session
    
    def _is_url(self, text: str) -> bool:
        """Verificar si texto es una URL válida"""
        try:
            result = urlparse(text)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def _looks_like_image_url(self, url: str) -> bool:
        """Verificar si URL parece ser una imagen"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg'}
        parsed = urlparse(url)
        path = parsed.path.lower()
        return any(path.endswith(ext) for ext in image_extensions)
    
    def _looks_like_base64_image(self, text: str) -> bool:
        """Verificar si texto parece ser imagen en base64"""
        if text.startswith('data:image/'):
            return True
        # Verificar si es base64 válido y largo (típico de imágenes)
        try:
            if len(text) > 100 and not ' ' in text:
                base64.b64decode(text)
                return True
        except:
            pass
        return False
    
    def _normalize_query(self, query: Union[str, Dict]) -> Union[str, Dict]:
        """
        Normalizar query para formato multimodal
        
        Returns:
            str: para queries de texto
            Dict: para queries de imagen {"image": "..."}
        """
        if isinstance(query, dict):
            return query
        
        if isinstance(query, str):
            # Detectar si es imagen
            if self._is_url(query) and self._looks_like_image_url(query):
                self.metrics["multimodal_queries"] += 1
                return {"image": query}
            elif self._looks_like_base64_image(query):
                self.metrics["multimodal_queries"] += 1
                return {"image": query}
        
        return query
    
    def _normalize_document(self, doc: Union[str, Dict, Any]) -> Dict[str, str]:
        """
        Normalizar documento para formato multimodal
        
        Returns:
            Dict con {"text": "..."} y opcionalmente {"image": "..."}
        """
        if isinstance(doc, str):
            return {"text": doc}
        
        if isinstance(doc, dict):
            # Si ya tiene formato correcto
            if "text" in doc or "image" in doc:
                # Contar documentos multimodales
                if "image" in doc:
                    self.metrics["multimodal_docs"] += 1
                return doc
            
            # Extraer texto de campos comunes
            text = doc.get("content") or doc.get("page_content") or str(doc)
            result = {"text": text}
            
            # Buscar imagen en metadata
            image_url = None
            if "metadata" in doc and isinstance(doc["metadata"], dict):
                metadata = doc["metadata"]
                image_url = (metadata.get("image") or 
                           metadata.get("image_url") or 
                           metadata.get("thumbnail"))
            
            # También buscar imagen directamente en el doc
            if not image_url:
                image_url = doc.get("image") or doc.get("image_url")
            
            if image_url:
                result["image"] = image_url
                self.metrics["multimodal_docs"] += 1
            
            return result
        
        # Para objetos con atributos (ej: LangChain Document)
        if hasattr(doc, 'page_content'):
            text = doc.page_content
            result = {"text": text}
            
            # Buscar imagen en metadata
            if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict):
                image_url = (doc.metadata.get("image") or 
                           doc.metadata.get("image_url") or 
                           doc.metadata.get("thumbnail"))
                if image_url:
                    result["image"] = image_url
                    self.metrics["multimodal_docs"] += 1
            
            return result
        
        # Fallback: convertir a string
        return {"text": str(doc)}
    
    def _make_request_with_retry(
        self, 
        query: Union[str, Dict], 
        documents: List[Dict], 
        top_n: int
    ) -> List[Dict]:
        """Hacer request al API con retry logic y backoff exponencial"""
        
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": False  # Solo necesitamos índices y scores
        }
        
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(f"Making rerank request (attempt {attempt + 1}/{self.max_retries + 1}): "
                           f"{len(documents)} docs, top_n={top_n}")
                
                response = self.session.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                # Manejar rate limiting específico de Jina
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 60))
                    logger.warning(f"Rate limited by API (attempt {attempt + 1}), waiting {retry_after}s")
                    if attempt < self.max_retries:
                        time.sleep(retry_after)
                        continue
                
                # Verificar respuesta exitosa
                response.raise_for_status()
                
                # Parsear respuesta
                data = response.json()
                if "results" not in data:
                    raise ValueError(f"Respuesta inválida del API: {data}")
                
                return data["results"]
                
            except requests.exceptions.RequestException as e:
                last_exception = e
                if attempt < self.max_retries:
                    # Calcular tiempo de espera con jitter
                    wait_time = (self.backoff_base * (2 ** attempt)) + random.uniform(0, 1)
                    logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {wait_time:.2f}s: {str(e)}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All retry attempts failed: {str(e)}")
                    if hasattr(e, 'response') and e.response is not None:
                        logger.error(f"Response: {e.response.text}")
            except Exception as e:
                last_exception = e
                logger.error(f"Error procesando respuesta de Jina API (attempt {attempt + 1}): {str(e)}")
                if attempt < self.max_retries:
                    wait_time = (self.backoff_base * (2 ** attempt)) + random.uniform(0, 1)
                    time.sleep(wait_time)
        
        # Si llegamos aquí, todos los reintentos fallaron
        self.metrics["error_count"] += 1
        raise Exception(f"Jina Reranker request failed after {self.max_retries + 1} attempts: {last_exception}")
    
    def _chunk_documents(
        self, 
        documents: List[Dict], 
        chunk_size: Optional[int] = None
    ) -> List[List[Dict]]:
        """Dividir documentos en chunks para respetar límites del API"""
        if chunk_size is None:
            chunk_size = self.max_chunk_size
        
        chunks = []
        for i in range(0, len(documents), chunk_size):
            chunk = documents[i:i + chunk_size]
            chunks.append(chunk)
        
        return chunks
    
    def _combine_chunk_results(
        self, 
        chunk_results: List[List[Dict]], 
        chunk_offsets: List[int],
        top_n: int
    ) -> List[Dict]:
        """Combinar resultados de múltiples chunks y ajustar índices globales"""
        all_results = []
        
        # Combinar resultados ajustando índices globales
        for chunk_idx, (results, offset) in enumerate(zip(chunk_results, chunk_offsets)):
            for result in results:
                # Ajustar índice global
                result_copy = result.copy()
                result_copy["index"] = result["index"] + offset
                result_copy["chunk_id"] = chunk_idx
                all_results.append(result_copy)
        
        # Ordenar por score descendente
        score_key = "relevance_score" if "relevance_score" in all_results[0] else "score"
        all_results.sort(key=lambda x: x.get(score_key, 0), reverse=True)
        
        # Retornar top_n
        return all_results[:top_n]
    
    def rerank(
        self, 
        query: Union[str, Dict], 
        documents: List[Union[str, Dict, Any]], 
        top_n: Optional[int] = None,
        return_documents: bool = False
    ) -> Tuple[List[Dict], float]:
        """
        Rerank documentos usando Jina Reranker m0
        
        Args:
            query: Query de texto o imagen
            documents: Lista de documentos a rerank
            top_n: Número de documentos a devolver (None = todos)
            return_documents: Si incluir contenido de documentos en resultado
            
        Returns:
            Tuple de (resultados, latencia_ms)
            resultados: Lista de {index, relevance_score, [text?, image?]}
        """
        start_time = time.time()
        
        if not documents:
            return [], 0.0
        
        if top_n is None:
            top_n = len(documents)
        
        top_n = min(top_n, len(documents))
        
        try:
            self.metrics["total_requests"] += 1
            self.metrics["total_documents_processed"] += len(documents)
            
            # Normalizar query
            normalized_query = self._normalize_query(query)
            
            # Normalizar documentos
            normalized_docs = [self._normalize_document(doc) for doc in documents]
            
            # Si hay pocos documentos o menos que top_n, procesar directamente
            if len(normalized_docs) <= self.max_chunk_size:
                results = self._make_request_with_retry(normalized_query, normalized_docs, top_n)
            else:
                # Chunking para documentos grandes
                chunks = self._chunk_documents(normalized_docs)
                chunk_offsets = [i * self.max_chunk_size for i in range(len(chunks))]
                self.metrics["chunks_processed"] += len(chunks)
                
                logger.info(f"Processing {len(documents)} documents in {len(chunks)} chunks")
                
                # Procesar cada chunk
                chunk_results = []
                for i, chunk in enumerate(chunks):
                    # Para chunks, pedimos todos los resultados del chunk
                    chunk_top_n = min(top_n, len(chunk))
                    chunk_result = self._make_request_with_retry(normalized_query, chunk, chunk_top_n)
                    chunk_results.append(chunk_result)
                
                # Combinar resultados
                results = self._combine_chunk_results(chunk_results, chunk_offsets, top_n)
            
            # Enriquecer resultados con contenido si se requiere
            if return_documents:
                for result in results:
                    doc_idx = result["index"]
                    if doc_idx < len(normalized_docs):
                        doc = normalized_docs[doc_idx]
                        if "text" in doc:
                            result["text"] = doc["text"]
                        if "image" in doc:
                            result["image"] = doc["image"]
            
            latency_ms = (time.time() - start_time) * 1000
            self.metrics["total_latency_ms"] += latency_ms
            
            # Asegurar que relevance_score esté presente
            for result in results:
                if "relevance_score" not in result and "score" in result:
                    result["relevance_score"] = result["score"]
            
            logger.debug(f"Reranking completed: {len(documents)} → {len(results)} docs, "
                        f"latency: {latency_ms:.2f}ms")
            
            return results, latency_ms
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.metrics["total_latency_ms"] += latency_ms
            logger.error(f"Reranking failed: {str(e)}")
            raise
    
    def rerank_doc_objects(
        self, 
        query: Union[str, Dict], 
        documents: List[Any], 
        top_n: Optional[int] = None,
        to_doc_obj: Optional[callable] = None
    ) -> Tuple[List[Any], float]:
        """
        Rerank objetos de documento (ej: LangChain Documents) y devolver reordenados
        
        Args:
            query: Query de texto o imagen
            documents: Lista de objetos documento
            top_n: Número de documentos a devolver
            to_doc_obj: Función para convertir doc → dict para API (opcional)
            
        Returns:
            Tuple de (documentos_reordenados, latencia_ms)
            Los documentos incluyen rerank_score en metadata
        """
        if not documents:
            return [], 0.0
        
        if top_n is None:
            top_n = len(documents)
        
        try:
            # Si se proporciona función de conversión personalizada
            if to_doc_obj:
                normalized_docs = [to_doc_obj(doc) for doc in documents]
            else:
                normalized_docs = [self._normalize_document(doc) for doc in documents]
            
            # Hacer reranking
            results, latency_ms = self.rerank(query, normalized_docs, top_n, return_documents=False)
            
            # Reordenar documentos originales
            reordered_docs = []
            for result in results:
                doc_idx = result["index"]
                if doc_idx < len(documents):
                    doc = documents[doc_idx]
                    
                    # Añadir rerank_score a metadata si es posible
                    score = result.get("relevance_score", result.get("score", 0))
                    if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict):
                        doc.metadata["rerank_score"] = score
                    elif isinstance(doc, dict):
                        if "metadata" not in doc:
                            doc["metadata"] = {}
                        doc["metadata"]["rerank_score"] = score
                    
                    reordered_docs.append(doc)
            
            return reordered_docs, latency_ms
            
        except Exception as e:
            # Fallback: devolver documentos originales en orden truncado
            self.metrics["fallback_count"] += 1
            logger.error(f"Reranking failed, using fallback: {str(e)}")
            return documents[:top_n], 0.0
    
    def __repr__(self) -> str:
        return (f"JinaReranker(model='{self.model}', timeout={self.timeout}s, "
                f"max_chunk_size={self.max_chunk_size})")


# Función de conveniencia
def create_jina_reranker(
    enabled: bool = True,
    top_k: int = 15,
    **kwargs
) -> Optional[JinaReranker]:
    """
    Crear instancia de JinaReranker basada en configuración de entorno
    
    Args:
        enabled: Si habilitar el reranker
        top_k: Número de documentos por defecto
        **kwargs: Parámetros adicionales
        
    Returns:
        Instancia de JinaReranker o None si está deshabilitado
    """
    if not enabled or not os.getenv("JINA_RERANKER_ENABLED", "true").lower() == "true":
        return None
    
    return JinaReranker(**kwargs)