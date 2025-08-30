"""
Definición del esquema multimodal para Qdrant que soporta tanto texto como imágenes.

Este módulo define la estructura de payload unificada que permite indexar
vectores de texto e imagen en una sola colección con deduplicación SHA-256.
"""

import hashlib
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Literal, Union
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path

class Modality(str, Enum):
    """Enum para tipos de modalidad soportados"""
    TEXT = "text"
    IMAGE = "image"

@dataclass
class MultimodalPayload:
    """
    Estructura unificada de payload para vectores de texto e imagen en Qdrant.
    
    Campos comunes requeridos:
    - id: UUID único para cada vector
    - modality: Tipo de contenido (text|image)
    - doc_id: Identificador del documento padre
    - page_number: Número de página en el documento
    - source_uri: Ruta al archivo original
    - hash: SHA-256 para deduplicación
    - embedding_model: Modelo usado para generar embeddings
    - created_at: Timestamp de creación
    
    Campos específicos por modalidad:
    - Para texto: page_content, content_preview
    - Para imagen: thumbnail_uri, width, height, image_index, bbox
    
    Campos opcionales:
    - title: Título del documento
    - author: Autor del documento 
    - creation_date: Fecha original del documento
    - access_control: Permisos de acceso
    """
    
    # Campos requeridos comunes
    id: str
    modality: Modality
    doc_id: str
    page_number: int
    source_uri: str
    hash: str
    embedding_model: str
    created_at: str
    
    # Campos específicos de texto (requeridos para modality=text)
    page_content: Optional[str] = None
    content_preview: Optional[str] = None
    
    # Campos específicos de imagen (requeridos para modality=image)  
    thumbnail_uri: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    image_index: Optional[int] = None  # Índice de imagen en la página
    bbox: Optional[Dict[str, float]] = None  # {"x0": float, "y0": float, "x1": float, "y1": float}
    
    # Campos opcionales comunes
    title: Optional[str] = None
    author: Optional[str] = None
    creation_date: Optional[str] = None
    access_control: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para inserción en Qdrant"""
        return asdict(self)
    
    @classmethod
    def from_text_chunk(
        cls,
        page_content: str,
        doc_id: str, 
        page_number: int,
        source_uri: str,
        embedding_model: str,
        title: Optional[str] = None,
        author: Optional[str] = None,
        creation_date: Optional[str] = None,
        **kwargs
    ) -> 'MultimodalPayload':
        """
        Crea payload para chunk de texto.
        
        Args:
            page_content: Contenido del chunk de texto
            doc_id: Identificador del documento
            page_number: Número de página
            source_uri: Ruta del archivo fuente
            embedding_model: Modelo usado para embeddings
            title: Título del documento (opcional)
            author: Autor del documento (opcional)
            creation_date: Fecha de creación original (opcional)
            **kwargs: Otros campos opcionales
        """
        # Generar hash SHA-256 del contenido
        content_hash = hashlib.sha256(page_content.encode('utf-8')).hexdigest()
        
        # Generar UUID único
        vector_id = str(uuid.uuid4())
        
        # Crear preview del contenido (primeros 200 chars)
        content_preview = page_content[:200] + "..." if len(page_content) > 200 else page_content
        
        return cls(
            id=vector_id,
            modality=Modality.TEXT,
            doc_id=doc_id,
            page_number=page_number,
            source_uri=source_uri,
            hash=content_hash,
            embedding_model=embedding_model,
            created_at=datetime.now().isoformat(),
            page_content=page_content,
            content_preview=content_preview,
            title=title,
            author=author,
            creation_date=creation_date,
            **kwargs
        )
    
    @classmethod
    def from_image_data(
        cls,
        image_data: bytes,
        doc_id: str,
        page_number: int, 
        image_index: int,
        source_uri: str,
        thumbnail_uri: str,
        width: int,
        height: int,
        embedding_model: str,
        bbox: Optional[Dict[str, float]] = None,
        title: Optional[str] = None,
        author: Optional[str] = None,
        creation_date: Optional[str] = None,
        **kwargs
    ) -> 'MultimodalPayload':
        """
        Crea payload para imagen.
        
        Args:
            image_data: Datos binarios de la imagen
            doc_id: Identificador del documento
            page_number: Número de página
            image_index: Índice de imagen en la página
            source_uri: Ruta del archivo fuente
            thumbnail_uri: Ruta del thumbnail
            width: Ancho en píxeles
            height: Alto en píxeles
            embedding_model: Modelo usado para embeddings
            bbox: Bounding box en la página (opcional)
            title: Título del documento (opcional)
            author: Autor del documento (opcional)
            creation_date: Fecha de creación original (opcional)
            **kwargs: Otros campos opcionales
        """
        # Generar hash SHA-256 de los datos de imagen
        image_hash = hashlib.sha256(image_data).hexdigest()
        
        # Generar UUID único
        vector_id = str(uuid.uuid4())
        
        return cls(
            id=vector_id,
            modality=Modality.IMAGE,
            doc_id=doc_id,
            page_number=page_number,
            source_uri=source_uri,
            hash=image_hash,
            embedding_model=embedding_model,
            created_at=datetime.now().isoformat(),
            thumbnail_uri=thumbnail_uri,
            width=width,
            height=height,
            image_index=image_index,
            bbox=bbox,
            title=title,
            author=author,
            creation_date=creation_date,
            **kwargs
        )
    
    def validate(self) -> bool:
        """
        Valida que el payload tenga todos los campos requeridos según modalidad.
        
        Returns:
            bool: True si es válido, False en caso contrario
        """
        # Campos requeridos comunes
        required_common = [
            self.id, self.modality, self.doc_id, 
            self.source_uri, self.hash, self.embedding_model, self.created_at
        ]
        
        if not all(field is not None for field in required_common):
            return False
        
        if self.page_number is None or self.page_number < 1:
            return False
            
        # Validaciones específicas por modalidad
        if self.modality == Modality.TEXT:
            return self.page_content is not None and len(self.page_content.strip()) > 0
            
        elif self.modality == Modality.IMAGE:
            required_image = [
                self.thumbnail_uri, self.width, self.height, self.image_index
            ]
            return all(field is not None for field in required_image) and \
                   self.width > 0 and self.height > 0 and self.image_index >= 0
        
        return False


class SchemaValidator:
    """Validador para asegurar compatibilidad con datos existentes"""
    
    @staticmethod
    def migrate_legacy_payload(legacy_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Migra payload legacy al nuevo formato multimodal.
        
        Args:
            legacy_payload: Payload en formato anterior
            
        Returns:
            Dict: Payload migrado al nuevo formato
        """
        # Detectar si es payload legacy por ausencia de campo 'modality'
        if 'modality' in legacy_payload:
            return legacy_payload  # Ya está en nuevo formato
            
        # Migrar payload de texto legacy
        migrated = {
            'id': str(uuid.uuid4()),  # Generar nuevo UUID
            'modality': Modality.TEXT.value,
            'doc_id': legacy_payload.get('doc_id', 'unknown'),
            'page_number': legacy_payload.get('page', 1),
            'source_uri': legacy_payload.get('source', 'unknown'),
            'hash': hashlib.sha256(
                legacy_payload.get('page_content', '').encode('utf-8')
            ).hexdigest(),
            'embedding_model': 'legacy-model',  # Valor por defecto
            'created_at': datetime.now().isoformat(),
            'page_content': legacy_payload.get('page_content'),
            'content_preview': legacy_payload.get('page_content', '')[:200],
            'title': legacy_payload.get('title'),
            'author': legacy_payload.get('author'),
            'creation_date': legacy_payload.get('creation_date')
        }
        
        return migrated
    
    @staticmethod 
    def is_valid_multimodal_payload(payload: Dict[str, Any]) -> bool:
        """
        Valida si un payload cumple con el esquema multimodal.
        
        Args:
            payload: Payload a validar
            
        Returns:
            bool: True si es válido
        """
        try:
            # Intentar crear objeto MultimodalPayload
            if payload.get('modality') == Modality.TEXT.value:
                MultimodalPayload(**{k: v for k, v in payload.items() 
                                   if k in MultimodalPayload.__dataclass_fields__})
            elif payload.get('modality') == Modality.IMAGE.value:
                MultimodalPayload(**{k: v for k, v in payload.items()
                                   if k in MultimodalPayload.__dataclass_fields__})
            else:
                return False
            return True
        except (TypeError, ValueError):
            return False


def create_deduplication_key(payload: MultimodalPayload) -> str:
    """
    Crea clave para deduplicación basada en hash, doc_id y modalidad.
    
    Args:
        payload: Payload multimodal
        
    Returns:
        str: Clave única para deduplicación
    """
    return f"{payload.modality.value}:{payload.doc_id}:{payload.hash}"


# Configuración de colección Qdrant
import os
MULTIMODAL_COLLECTION_CONFIG = {
    "collection_name": os.getenv("QDRANT_COLLECTION", "rag_multimodal"),  # Usar variable de entorno
    "vector_size": None,  # Se configurará dinámicamente según modelo
    "distance": None,  # Se configurará según proveedor (DOT para Jina, COSINE para otros)
    "payload_schema": {
        # Definir índices para campos comunes de búsqueda
        "modality": {"type": "keyword", "index": True},
        "doc_id": {"type": "keyword", "index": True}, 
        "page_number": {"type": "integer", "index": True},
        "source_uri": {"type": "keyword", "index": True},
        "hash": {"type": "keyword", "index": True},  # Para deduplicación
        "embedding_model": {"type": "keyword", "index": True},
        "created_at": {"type": "datetime", "index": True},
        "title": {"type": "text", "index": True},
        "author": {"type": "text", "index": True}
    }
}