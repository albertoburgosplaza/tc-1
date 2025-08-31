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
    IMAGE_DESCRIPTION = "image_description"

@dataclass
class MultimodalPayload:
    """
    Estructura unificada de payload para vectores de texto e imagen en Qdrant.
    
    Campos comunes requeridos:
    - id: UUID único para cada vector
    - modality: Tipo de contenido (text|image|image_description)
    - doc_id: Identificador del documento padre
    - page_number: Número de página en el documento
    - source_uri: Ruta al archivo original
    - hash: SHA-256 para deduplicación
    - embedding_model: Modelo usado para generar embeddings
    - created_at: Timestamp de creación
    
    Campos específicos por modalidad:
    - Para texto: page_content, content_preview
    - Para imagen: thumbnail_uri, width, height, image_index, bbox
    - Para descripción de imagen: page_content, content_preview, thumbnail_uri (opcional)
    
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
    
    # Campos específicos de descripción de imagen (requeridos para modality=image_description)
    image_description: Optional[str] = None  # Descripción generada por IA
    description_model: Optional[str] = None  # Modelo que generó la descripción
    description_generated_at: Optional[str] = None  # Timestamp de generación
    
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
    
    @classmethod
    def from_image_description(
        cls,
        page_content: str,
        doc_id: str,
        page_number: int,
        source_uri: str,
        image_hash: str,
        thumbnail_uri: str,
        embedding_model: str,
        image_description: str,
        description_model: str,
        description_generated_at: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        image_index: Optional[int] = None,
        bbox: Optional[Dict[str, float]] = None,
        title: Optional[str] = None,
        author: Optional[str] = None,
        creation_date: Optional[str] = None,
        **kwargs
    ) -> 'MultimodalPayload':
        """
        Crea payload para descripción de imagen.
        
        Args:
            page_content: Descripción textual de la imagen generada por AI
            doc_id: Identificador del documento
            page_number: Número de página
            source_uri: Ruta del archivo fuente
            image_hash: Hash SHA-256 de la imagen original
            thumbnail_uri: Ruta del thumbnail de la imagen
            embedding_model: Modelo usado para embeddings de texto
            image_description: Descripción de la imagen generada por IA (requerido)
            description_model: Modelo que generó la descripción (requerido)
            description_generated_at: Timestamp de generación (opcional, se genera automáticamente)
            width: Ancho en píxeles (opcional, para referencia)
            height: Alto en píxeles (opcional, para referencia)
            image_index: Índice de imagen en la página (opcional)
            bbox: Bounding box en la página (opcional)
            title: Título del documento (opcional)
            author: Autor del documento (opcional)
            creation_date: Fecha de creación original (opcional)
            **kwargs: Otros campos opcionales
        """
        # Generar UUID único
        vector_id = str(uuid.uuid4())
        
        # Generar timestamp automáticamente si no se proporciona
        if description_generated_at is None:
            description_generated_at = datetime.now().isoformat()
        
        # Crear preview del contenido (primeros 200 chars)
        content_preview = page_content[:200] + "..." if len(page_content) > 200 else page_content
        
        return cls(
            id=vector_id,
            modality=Modality.IMAGE_DESCRIPTION,
            doc_id=doc_id,
            page_number=page_number,
            source_uri=source_uri,
            hash=image_hash,  # Usar hash de la imagen original
            embedding_model=embedding_model,
            created_at=datetime.now().isoformat(),
            page_content=page_content,
            content_preview=content_preview,
            thumbnail_uri=thumbnail_uri,
            width=width,
            height=height,
            image_index=image_index,
            bbox=bbox,
            title=title,
            author=author,
            creation_date=creation_date,
            image_description=image_description,
            description_model=description_model,
            description_generated_at=description_generated_at,
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
                   
        elif self.modality == Modality.IMAGE_DESCRIPTION:
            # Validar campos básicos de contenido
            content_valid = self.page_content is not None and len(self.page_content.strip()) > 0
            
            # Validar campos específicos de descripción de imagen
            description_valid = (
                self.image_description is not None and 
                len(self.image_description.strip()) > 0 and
                self.description_model is not None and 
                len(self.description_model.strip()) > 0
            )
            
            # Validar formato de timestamp si está presente
            timestamp_valid = True
            if self.description_generated_at is not None:
                try:
                    datetime.fromisoformat(self.description_generated_at.replace('Z', '+00:00'))
                except ValueError:
                    timestamp_valid = False
            
            return content_valid and description_valid and timestamp_valid
        
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
            # Intentar crear objeto MultimodalPayload y validarlo
            filtered_payload = {k: v for k, v in payload.items() 
                              if k in MultimodalPayload.__dataclass_fields__}
            
            multimodal_obj = MultimodalPayload(**filtered_payload)
            
            # Usar el método validate() del objeto para validación específica por modalidad
            return multimodal_obj.validate()
            
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
        "author": {"type": "text", "index": True},
        # Índices para campos de descripción de imagen
        "image_description": {"type": "text", "index": True},  # Búsqueda textual en descripciones
        "description_model": {"type": "keyword", "index": True},  # Filtrado por modelo
        "description_generated_at": {"type": "datetime", "index": True}  # Filtrado por fecha de generación
    }
}