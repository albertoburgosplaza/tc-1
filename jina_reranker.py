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
        multimodal_timeout: Optional[int] = None,
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
            timeout: Timeout en segundos para requests HTTP estándar
            multimodal_timeout: Timeout específico para requests multimodales (si es None, usa timeout * 2)
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
        # Configurar timeout multimodal desde variable de entorno o parámetro
        self.multimodal_timeout = (
            multimodal_timeout or 
            int(os.getenv("MULTIMODAL_REQUEST_TIMEOUT", timeout * 2))
        )
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
                   f"multimodal_timeout={self.multimodal_timeout}s, max_chunk_size={max_chunk_size}")
    
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
            "chunks_processed": 0,
            "text_only_docs": 0,
            "image_only_docs": 0,
            "mixed_docs": 0,
            "modality_detection_calls": 0
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
    
    def _detect_modality(self, doc_data: Dict[str, Any]) -> str:
        """
        Detectar la modalidad de un documento basándose en su contenido y metadata
        
        Args:
            doc_data: Datos del documento (dict o metadata)
            
        Returns:
            "text", "image", o "mixed"
        """
        self.metrics["modality_detection_calls"] += 1
        
        has_text = False
        has_image = False
        
        # Verificar si tiene contenido de texto
        text_content = (
            doc_data.get("content") or 
            doc_data.get("page_content") or 
            doc_data.get("text")
        )
        if text_content and isinstance(text_content, str) and text_content.strip():
            has_text = True
        
        # Verificar modalidad explícita en metadata
        if "metadata" in doc_data and isinstance(doc_data["metadata"], dict):
            metadata = doc_data["metadata"]
            explicit_modality = metadata.get("modality")
            if explicit_modality == "image":
                has_image = True
            elif explicit_modality == "text":
                has_text = True
            elif explicit_modality == "image_description":
                # image_description is primarily text but may have associated image
                has_text = True
                # Check if it also has image reference for mixed modality
                if metadata.get("thumbnail_uri"):
                    has_image = True
            
            # Buscar campos de imagen en metadata del esquema multimodal
            image_fields = [
                "image", "image_url", "thumbnail", "thumbnail_uri", 
                "source_uri"  # Para imágenes almacenadas localmente
            ]
            for field in image_fields:
                if metadata.get(field):
                    image_value = metadata.get(field)
                    if self._is_image_reference(image_value):
                        has_image = True
                        break
        
        # Buscar campos de imagen directamente en el documento
        image_fields = ["image", "image_url", "thumbnail_uri"]
        for field in image_fields:
            if doc_data.get(field):
                image_value = doc_data.get(field)
                if self._is_image_reference(image_value):
                    has_image = True
                    break
        
        # Determinar modalidad final y actualizar métricas
        if has_image and has_text:
            self.metrics["mixed_docs"] += 1
            return "mixed"
        elif has_image:
            self.metrics["image_only_docs"] += 1
            return "image" 
        else:
            self.metrics["text_only_docs"] += 1
            return "text"
    
    def _is_image_reference(self, value: str) -> bool:
        """
        Verificar si un valor es una referencia a imagen (URL, URI local, o base64)
        
        Args:
            value: Valor a verificar
            
        Returns:
            bool: True si es referencia a imagen
        """
        if not isinstance(value, str) or not value.strip():
            return False
        
        value = value.strip()
        
        # Data URI de imagen
        if value.startswith('data:image/'):
            return True
        
        # URI local de imagen (típico del esquema multimodal)
        if value.startswith('file://') or value.startswith('/var/data/rag/images/'):
            return True
        
        # URL HTTP/HTTPS de imagen
        if self._is_url(value) and self._looks_like_image_url(value):
            return True
        
        # Base64 sin prefijo data:
        if self._looks_like_base64_image(value):
            return True
        
        return False
    
    def _process_image_for_api(self, image_ref: str) -> str:
        """
        Procesar referencia de imagen para el API de Jina Reranker m0
        
        Args:
            image_ref: Referencia a imagen (URI, URL, base64)
            
        Returns:
            str: Imagen formateada para el API
        """
        if not image_ref:
            return image_ref
            
        # Si ya es data URI, usar directamente
        if image_ref.startswith('data:image/'):
            return image_ref
        
        # Si es URI local (file://) - convertir a file:// si no lo tiene
        if image_ref.startswith('/var/data/rag/images/') or image_ref.startswith('/'):
            if not image_ref.startswith('file://'):
                return f"file://{image_ref}"
            return image_ref
        
        # Si es base64 puro, añadir prefijo data:image
        if self._looks_like_base64_image(image_ref) and not image_ref.startswith('data:'):
            # Asumir JPEG por defecto si no se puede determinar
            return f"data:image/jpeg;base64,{image_ref}"
        
        # Para URLs HTTP/HTTPS, usar directamente
        return image_ref
    
    def _is_multimodal_request(self, query: Union[str, Dict], documents: List[Dict]) -> bool:
        """
        Determinar si una request es multimodal (contiene imágenes)
        
        Args:
            query: Query de la request
            documents: Documentos de la request
            
        Returns:
            bool: True si la request es multimodal
        """
        # Verificar query
        if isinstance(query, dict) and "image" in query:
            return True
        
        # Verificar documentos
        for doc in documents:
            if isinstance(doc, dict) and "image" in doc:
                return True
                
        return False
    
    def _extract_original_metadata(self, doc: Union[str, Dict, Any]) -> Dict[str, Any]:
        """
        Extraer metadata original completa de un documento
        
        Args:
            doc: Documento original
            
        Returns:
            Dict con metadata preservada
        """
        if isinstance(doc, str):
            return {}
        
        if isinstance(doc, dict):
            # Para diccionarios, preservar toda la metadata
            metadata = doc.get("metadata", {})
            if isinstance(metadata, dict):
                # Crear copia para evitar modificar original
                preserved = metadata.copy()
                
                # Añadir campos de imagen directos si existen
                for field in ["image", "image_url", "thumbnail_uri"]:
                    if field in doc and field not in preserved:
                        preserved[field] = doc[field]
                
                return preserved
            return {}
        
        # Para objetos con atributos (ej: LangChain Document)
        if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict):
            return doc.metadata.copy()
        
        return {}
    
    def _handle_multimodal_fallback(
        self,
        query: Union[str, Dict], 
        documents: List[Union[str, Dict, Any]], 
        top_n: int,
        normalized_docs: List[Dict],
        latency_ms: float,
        error: Exception
    ) -> Tuple[List[Dict], float]:
        """
        Manejar fallback cuando reranking multimodal falla
        
        Args:
            query: Query original
            documents: Documentos originales
            top_n: Número de documentos requeridos
            normalized_docs: Documentos normalizados
            latency_ms: Latencia hasta el momento del error
            error: Error que causó el fallback
            
        Returns:
            Tuple de (resultados_fallback, latencia_ms)
        """
        self.metrics["fallback_count"] += 1
        
        # Detectar si hay contenido multimodal
        has_multimodal = any(
            isinstance(doc, dict) and "image" in doc 
            for doc in normalized_docs
        ) or (isinstance(query, dict) and "image" in query)
        
        if has_multimodal:
            logger.warning(f"Multimodal reranking failed ({str(error)}), attempting text-only fallback")
            
            # Intentar fallback solo con texto
            try:
                # Filtrar solo contenido de texto
                text_only_docs = []
                for doc in normalized_docs:
                    if isinstance(doc, dict) and "text" in doc:
                        text_only_docs.append({"text": doc["text"]})
                    elif isinstance(doc, str):
                        text_only_docs.append({"text": doc})
                
                # Query solo texto
                text_only_query = query
                if isinstance(query, dict) and "text" in query:
                    text_only_query = query["text"]
                elif isinstance(query, dict) and "image" in query:
                    # Si solo hay query de imagen, usar búsqueda básica
                    logger.warning("Image-only query in fallback, using basic similarity")
                    return self._basic_similarity_fallback(documents, top_n, latency_ms)
                
                if text_only_docs:
                    logger.info(f"Attempting text-only reranking with {len(text_only_docs)} documents")
                    results = self._make_request_with_retry(text_only_query, text_only_docs, top_n)
                    
                    logger.info(f"Text-only fallback successful: {len(results)} results")
                    return results, latency_ms
                    
            except Exception as fallback_error:
                logger.error(f"Text-only fallback also failed: {str(fallback_error)}")
        
        # Fallback final: orden original
        logger.warning(f"All reranking failed, returning documents in original order")
        return self._basic_similarity_fallback(documents, top_n, latency_ms)
    
    def _basic_similarity_fallback(
        self,
        documents: List[Union[str, Dict, Any]], 
        top_n: int, 
        latency_ms: float
    ) -> Tuple[List[Dict], float]:
        """
        Fallback básico que devuelve documentos en orden original con scores simulados
        
        Args:
            documents: Documentos originales
            top_n: Número de documentos requeridos
            latency_ms: Latencia acumulada
            
        Returns:
            Tuple de (resultados_básicos, latencia_ms)
        """
        results = []
        for i in range(min(top_n, len(documents))):
            # Score descendente simulado basado en posición original
            score = 1.0 - (i * 0.01)  # 1.0, 0.99, 0.98, etc.
            results.append({
                "index": i,
                "relevance_score": score,
                "score": score,
                "fallback": True
            })
        
        logger.info(f"Basic fallback returned {len(results)} documents with simulated scores")
        return results, latency_ms

    def _normalize_document(self, doc: Union[str, Dict, Any]) -> Dict[str, str]:
        """
        Normalizar documento para formato multimodal con detección mejorada
        
        Returns:
            Dict con {"text": "..."} y opcionalmente {"image": "..."}
        """
        if isinstance(doc, str):
            # Si es string puro, verificar si es imagen
            if self._is_image_reference(doc):
                self.metrics["multimodal_docs"] += 1
                return {"image": self._process_image_for_api(doc)}
            return {"text": doc}
        
        if isinstance(doc, dict):
            # Si ya tiene formato correcto de Jina
            if "text" in doc or "image" in doc:
                if "image" in doc:
                    self.metrics["multimodal_docs"] += 1
                return doc
            
            # Detectar modalidad del documento
            modality = self._detect_modality(doc)
            
            result = {}
            
            # Extraer contenido de texto
            text_content = (doc.get("content") or 
                          doc.get("page_content") or 
                          doc.get("text"))
            if text_content and isinstance(text_content, str) and text_content.strip():
                result["text"] = text_content
            
            # Extraer referencia de imagen
            image_ref = None
            
            # Buscar en metadata primero (esquema multimodal)
            if "metadata" in doc and isinstance(doc["metadata"], dict):
                metadata = doc["metadata"]
                # Prioridad: thumbnail_uri > image_url > source_uri > image
                for field in ["thumbnail_uri", "image_url", "source_uri", "image"]:
                    candidate = metadata.get(field)
                    if candidate and self._is_image_reference(candidate):
                        image_ref = candidate
                        break
            
            # Buscar directamente en el documento si no se encontró en metadata
            if not image_ref:
                for field in ["image", "image_url", "thumbnail_uri"]:
                    candidate = doc.get(field)
                    if candidate and self._is_image_reference(candidate):
                        image_ref = candidate
                        break
            
            # Añadir imagen si se encontró, procesándola para el API
            if image_ref:
                result["image"] = self._process_image_for_api(image_ref)
                self.metrics["multimodal_docs"] += 1
            
            # Si no hay contenido, usar fallback
            if not result:
                result["text"] = str(doc)
            
            return result
        
        # Para objetos con atributos (ej: LangChain Document)
        if hasattr(doc, 'page_content'):
            result = {}
            
            # Texto del documento
            if doc.page_content and doc.page_content.strip():
                result["text"] = doc.page_content
            
            # Buscar imagen en metadata
            if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict):
                # Usar mismo orden de prioridad
                for field in ["thumbnail_uri", "image_url", "source_uri", "image"]:
                    candidate = doc.metadata.get(field)
                    if candidate and self._is_image_reference(candidate):
                        result["image"] = self._process_image_for_api(candidate)
                        self.metrics["multimodal_docs"] += 1
                        break
            
            return result if result else {"text": str(doc)}
        
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
        
        # Determinar timeout apropiado basado en si es multimodal
        is_multimodal = self._is_multimodal_request(query, documents)
        request_timeout = self.multimodal_timeout if is_multimodal else self.timeout
        
        if is_multimodal:
            logger.debug(f"Using multimodal timeout: {request_timeout}s for request with images")
        
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(f"Making rerank request (attempt {attempt + 1}/{self.max_retries + 1}): "
                           f"{len(documents)} docs, top_n={top_n}")
                
                response = self.session.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=request_timeout
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
                if not isinstance(result, dict):
                    logger.warning(f"Invalid result type in chunk {chunk_idx}: {type(result)}")
                    continue
                    
                # Verificar que el índice sea válido
                if "index" not in result or result["index"] is None:
                    logger.warning(f"Missing or None index in result: {result}")
                    continue
                    
                try:
                    # Ajustar índice global
                    result_copy = result.copy()
                    result_copy["index"] = result["index"] + offset
                    result_copy["chunk_id"] = chunk_idx
                    all_results.append(result_copy)
                except (TypeError, ValueError) as e:
                    logger.warning(f"Error adjusting index for result {result}: {e}")
                    continue
        
        # Verificar que tenemos resultados antes de ordenar
        if not all_results:
            logger.warning("No valid results after combining chunks")
            return []
        
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
                    if doc_idx < len(normalized_docs) and doc_idx < len(documents):
                        normalized_doc = normalized_docs[doc_idx]
                        original_doc = documents[doc_idx]
                        
                        # Añadir contenido normalizado
                        if "text" in normalized_doc:
                            result["text"] = normalized_doc["text"]
                        if "image" in normalized_doc:
                            result["image"] = normalized_doc["image"]
                        
                        # Preservar metadata original completa
                        result["original_metadata"] = self._extract_original_metadata(original_doc)
            
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
            self.metrics["error_count"] += 1
            
            # Intentar fallback para errores multimodales específicos
            return self._handle_multimodal_fallback(
                query, documents, top_n, normalized_docs, latency_ms, e
            )
    
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
            
            # Verificar que results es válido
            if results is None or not isinstance(results, list):
                logger.warning(f"Rerank returned invalid results: {type(results)}, using original order")
                return documents[:top_n], latency_ms
            
            # Reordenar documentos originales
            reordered_docs = []
            for result in results:
                if not isinstance(result, dict) or "index" not in result:
                    logger.warning(f"Invalid result format: {result}, skipping")
                    continue
                
                doc_idx = result["index"]
                if doc_idx is None:
                    logger.warning(f"Result has None index: {result}, skipping")
                    continue
                    
                try:
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
                    else:
                        logger.warning(f"Document index {doc_idx} out of range (max: {len(documents)-1})")
                except (TypeError, ValueError) as e:
                    logger.warning(f"Error processing result index {doc_idx}: {e}")
            
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