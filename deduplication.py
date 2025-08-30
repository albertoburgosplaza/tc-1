"""
Módulo de deduplicación para el sistema RAG multimodal.

Este módulo proporciona funcionalidades para:
- Calcular hashes SHA-256 de contenido
- Verificar duplicados existentes en Qdrant
- Integrar deduplicación en el flujo de ingesta
- Estadísticas de deduplicación
"""

import hashlib
import logging
from typing import Dict, Any, Set, Optional, Tuple, List
import time

from qdrant_client import QdrantClient
from qdrant_client.http import models

from multimodal_schema import MultimodalPayload, create_deduplication_key, Modality

logger = logging.getLogger(__name__)

class DeduplicationManager:
    """Gestor de deduplicación para contenido multimodal"""
    
    def __init__(self, client: QdrantClient, collection_name: str):
        """
        Inicializar gestor de deduplicación.
        
        Args:
            client: Cliente Qdrant
            collection_name: Nombre de la colección
        """
        self.client = client
        self.collection_name = collection_name
        self.stats = {
            "total_processed": 0,
            "duplicates_found": 0,
            "unique_content": 0,
            "hash_cache_hits": 0,
            "qdrant_queries": 0,
            "start_time": None,
            "end_time": None
        }
        self.hash_cache: Set[str] = set()
        self.dedup_key_cache: Set[str] = set()
        
    def start_session(self):
        """Iniciar sesión de deduplicación"""
        self.stats["start_time"] = time.time()
        self.hash_cache.clear()
        self.dedup_key_cache.clear()
        logger.info("Started deduplication session")
    
    def end_session(self) -> Dict[str, Any]:
        """
        Finalizar sesión y retornar estadísticas.
        
        Returns:
            Dict: Estadísticas de la sesión
        """
        self.stats["end_time"] = time.time()
        duration = self.stats["end_time"] - self.stats["start_time"]
        
        summary = {
            **self.stats,
            "duration_seconds": duration,
            "deduplication_rate": (
                self.stats["duplicates_found"] / max(1, self.stats["total_processed"])
            ) * 100,
            "processing_rate": self.stats["total_processed"] / max(1, duration)
        }
        
        logger.info(f"Deduplication session completed: {summary}")
        return summary

    def compute_content_hash(self, content: str | bytes) -> str:
        """
        Generar hash SHA-256 del contenido.
        
        Args:
            content: Contenido de texto o binario
            
        Returns:
            str: Hash SHA-256 en formato hexadecimal
        """
        if isinstance(content, str):
            content_bytes = content.encode('utf-8')
        else:
            content_bytes = content
            
        return hashlib.sha256(content_bytes).hexdigest()
    
    def check_duplicate_exists(
        self, 
        hash_value: str, 
        doc_id: Optional[str] = None,
        modality: Optional[str] = None,
        use_cache: bool = True
    ) -> bool:
        """
        Verificar si existe contenido duplicado en Qdrant.
        
        Args:
            hash_value: Hash SHA-256 del contenido
            doc_id: ID del documento (opcional, para filtrado)
            modality: Modalidad del contenido (opcional, para filtrado)
            use_cache: Si usar cache local para optimización
            
        Returns:
            bool: True si existe duplicado
        """
        self.stats["total_processed"] += 1
        
        # Verificar cache local primero
        if use_cache and hash_value in self.hash_cache:
            self.stats["hash_cache_hits"] += 1
            self.stats["duplicates_found"] += 1
            logger.debug(f"Cache hit for hash: {hash_value[:16]}...")
            return True
        
        # Construir filtros para la búsqueda
        must_conditions = [
            models.FieldCondition(
                key="hash",
                match=models.MatchValue(value=hash_value)
            )
        ]
        
        if doc_id:
            must_conditions.append(
                models.FieldCondition(
                    key="doc_id",
                    match=models.MatchValue(value=doc_id)
                )
            )
            
        if modality:
            must_conditions.append(
                models.FieldCondition(
                    key="modality", 
                    match=models.MatchValue(value=modality)
                )
            )
        
        try:
            # Consultar Qdrant
            search_filter = models.Filter(
                must=must_conditions
            )
            
            self.stats["qdrant_queries"] += 1
            result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=search_filter,
                limit=1,  # Solo necesitamos saber si existe
                with_payload=False,
                with_vectors=False
            )
            
            points, _ = result
            exists = len(points) > 0
            
            if exists:
                self.stats["duplicates_found"] += 1
                if use_cache:
                    self.hash_cache.add(hash_value)
                logger.debug(f"Duplicate found in Qdrant for hash: {hash_value[:16]}...")
            else:
                self.stats["unique_content"] += 1
                logger.debug(f"Unique content for hash: {hash_value[:16]}...")
                
            return exists
            
        except Exception as e:
            logger.error(f"Error checking duplicate for hash {hash_value[:16]}: {e}")
            # En caso de error, asumir que no es duplicado para no bloquear ingesta
            return False
    
    def check_payload_duplicate(self, payload: MultimodalPayload, use_cache: bool = True) -> bool:
        """
        Verificar si un payload multimodal es duplicado usando clave de deduplicación.
        
        Args:
            payload: Payload multimodal
            use_cache: Si usar cache local
            
        Returns:
            bool: True si es duplicado
        """
        dedup_key = create_deduplication_key(payload)
        
        # Verificar cache local
        if use_cache and dedup_key in self.dedup_key_cache:
            self.stats["duplicates_found"] += 1
            self.stats["hash_cache_hits"] += 1
            return True
        
        # Verificar en Qdrant usando hash
        is_duplicate = self.check_duplicate_exists(
            hash_value=payload.hash,
            doc_id=payload.doc_id,
            modality=payload.modality.value,
            use_cache=False  # Ya manejamos cache aquí
        )
        
        if is_duplicate and use_cache:
            self.dedup_key_cache.add(dedup_key)
        elif not is_duplicate and use_cache:
            self.dedup_key_cache.add(dedup_key)  # Recordar que no es duplicado también
            
        return is_duplicate
    
    def filter_unique_payloads(
        self, 
        payloads: List[MultimodalPayload],
        force_reindex: bool = False
    ) -> Tuple[List[MultimodalPayload], List[MultimodalPayload]]:
        """
        Filtrar payloads únicos de una lista.
        
        Args:
            payloads: Lista de payloads a filtrar
            force_reindex: Si ignorar duplicados y forzar re-indexación
            
        Returns:
            Tuple: (payloads únicos, payloads duplicados)
        """
        unique_payloads = []
        duplicate_payloads = []
        
        for payload in payloads:
            if force_reindex or not self.check_payload_duplicate(payload):
                unique_payloads.append(payload)
            else:
                duplicate_payloads.append(payload)
        
        logger.info(f"Filtered {len(payloads)} payloads: {len(unique_payloads)} unique, {len(duplicate_payloads)} duplicates")
        return unique_payloads, duplicate_payloads
    
    def get_duplicate_statistics(self) -> Dict[str, Any]:
        """
        Obtener estadísticas detalladas de duplicados por modalidad y documento.
        
        Returns:
            Dict: Estadísticas detalladas
        """
        try:
            # Consultar estadísticas por modalidad
            modality_stats = {}
            for modality in ["text", "image"]:
                result = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="modality",
                                match=models.MatchValue(value=modality)
                            )
                        ]
                    ),
                    limit=10000,  # Límite alto para contar
                    with_payload=True,
                    with_vectors=False
                )
                
                points, _ = result
                modality_stats[modality] = {
                    "total_vectors": len(points),
                    "unique_docs": len(set(p.payload.get("doc_id", "unknown") for p in points)),
                    "unique_hashes": len(set(p.payload.get("hash", "") for p in points))
                }
            
            return {
                "collection_name": self.collection_name,
                "modality_statistics": modality_stats,
                "session_statistics": self.stats,
                "cache_size": {
                    "hash_cache": len(self.hash_cache),
                    "dedup_key_cache": len(self.dedup_key_cache)
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting duplicate statistics: {e}")
            return {"error": str(e)}
    
    def cleanup_duplicates(
        self, 
        dry_run: bool = True,
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Limpiar duplicados existentes en la colección.
        
        Args:
            dry_run: Si solo simular la limpieza
            batch_size: Tamaño de lote para procesamiento
            
        Returns:
            Dict: Resultado de la limpieza
        """
        logger.warning(f"Starting duplicate cleanup (dry_run={dry_run})")
        
        cleanup_stats = {
            "duplicates_found": 0,
            "duplicates_removed": 0,
            "errors": [],
            "dry_run": dry_run
        }
        
        try:
            # Obtener todos los puntos agrupados por hash
            hash_groups = {}
            
            offset = None
            while True:
                result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                
                points, next_offset = result
                if not points:
                    break
                
                for point in points:
                    hash_value = point.payload.get("hash")
                    if hash_value:
                        if hash_value not in hash_groups:
                            hash_groups[hash_value] = []
                        hash_groups[hash_value].append(point.id)
                
                offset = next_offset
                if offset is None:
                    break
            
            # Identificar y procesar duplicados
            for hash_value, point_ids in hash_groups.items():
                if len(point_ids) > 1:
                    cleanup_stats["duplicates_found"] += len(point_ids) - 1
                    
                    # Mantener el primer punto, eliminar el resto
                    points_to_remove = point_ids[1:]
                    
                    if not dry_run:
                        try:
                            self.client.delete(
                                collection_name=self.collection_name,
                                points_selector=models.PointIdsList(
                                    points=points_to_remove
                                )
                            )
                            cleanup_stats["duplicates_removed"] += len(points_to_remove)
                        except Exception as e:
                            error_msg = f"Failed to remove duplicates for hash {hash_value}: {e}"
                            cleanup_stats["errors"].append(error_msg)
                            logger.error(error_msg)
            
            logger.info(f"Duplicate cleanup completed: {cleanup_stats}")
            return cleanup_stats
            
        except Exception as e:
            error_msg = f"Duplicate cleanup failed: {e}"
            cleanup_stats["errors"].append(error_msg)
            logger.error(error_msg)
            return cleanup_stats


def create_deduplication_manager(client: QdrantClient, collection_name: str) -> DeduplicationManager:
    """
    Crear instancia de gestor de deduplicación.
    
    Args:
        client: Cliente Qdrant
        collection_name: Nombre de la colección
        
    Returns:
        DeduplicationManager: Instancia configurada
    """
    return DeduplicationManager(client, collection_name)


def compute_text_hash(text: str) -> str:
    """
    Función de conveniencia para calcular hash de texto.
    
    Args:
        text: Texto a hashear
        
    Returns:
        str: Hash SHA-256
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def compute_image_hash(image_data: bytes) -> str:
    """
    Función de conveniencia para calcular hash de imagen.
    
    Args:
        image_data: Datos binarios de imagen
        
    Returns:
        str: Hash SHA-256
    """
    return hashlib.sha256(image_data).hexdigest()


def batch_deduplicate_texts(
    texts: List[str], 
    dedup_manager: DeduplicationManager,
    doc_ids: Optional[List[str]] = None
) -> Tuple[List[str], List[int]]:
    """
    Deduplicar lote de textos.
    
    Args:
        texts: Lista de textos
        dedup_manager: Gestor de deduplicación
        doc_ids: IDs de documento correspondientes (opcional)
        
    Returns:
        Tuple: (textos únicos, índices de textos únicos)
    """
    unique_texts = []
    unique_indices = []
    
    for i, text in enumerate(texts):
        text_hash = compute_text_hash(text)
        doc_id = doc_ids[i] if doc_ids and i < len(doc_ids) else None
        
        if not dedup_manager.check_duplicate_exists(text_hash, doc_id, "text"):
            unique_texts.append(text)
            unique_indices.append(i)
    
    return unique_texts, unique_indices