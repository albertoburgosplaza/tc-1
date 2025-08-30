"""
Módulo para gestión de almacenamiento local de imágenes del sistema RAG multimodal.

Este módulo proporciona funcionalidades para:
- Validación segura de rutas y prevención de directory traversal
- Almacenamiento organizado de imágenes y thumbnails
- Generación de URIs locales
- Control de acceso y permisos por documento
- Limpieza de archivos huérfanos

Estructura de almacenamiento:
/var/data/rag/images/{doc_id}/p{page_number}/{hash}.{ext}
"""

import os
import logging
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import quote
import re

logger = logging.getLogger(__name__)

class ImageStorageMetrics:
    """
    Clase para registrar y almacenar métricas de almacenamiento de imágenes
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reinicia todas las métricas"""
        self.images_stored = 0
        self.thumbnails_stored = 0
        self.storage_operations = 0
        self.storage_errors = 0
        self.total_storage_time = 0.0
        self.bytes_stored = 0
        self.duplicate_images_avoided = 0
        self.documents_processed = 0
        self.cleanup_operations = 0
        self.orphaned_files_cleaned = 0
        self.storage_times = []
        self.errors_by_type = {}
    
    def record_image_storage(self, success: bool, processing_time: float, bytes_stored: int = 0, was_duplicate: bool = False):
        """Registra métricas de almacenamiento de una imagen"""
        self.storage_operations += 1
        
        if success:
            if not was_duplicate:
                self.images_stored += 1
                self.bytes_stored += bytes_stored
            else:
                self.duplicate_images_avoided += 1
            
            self.total_storage_time += processing_time
            self.storage_times.append(processing_time)
        else:
            self.storage_errors += 1
    
    def record_thumbnail_storage(self, success: bool, bytes_stored: int = 0):
        """Registra métricas de almacenamiento de thumbnail"""
        if success:
            self.thumbnails_stored += 1
            self.bytes_stored += bytes_stored
    
    def record_document_processed(self):
        """Registra que se procesó un documento completo"""
        self.documents_processed += 1
    
    def record_cleanup_operation(self, orphaned_files: int = 0):
        """Registra operación de limpieza"""
        self.cleanup_operations += 1
        self.orphaned_files_cleaned += orphaned_files
    
    def record_storage_error(self, error_type: str):
        """Registra un error de almacenamiento"""
        self.errors_by_type[error_type] = self.errors_by_type.get(error_type, 0) + 1
    
    def get_metrics_summary(self) -> Dict:
        """Retorna resumen completo de métricas de almacenamiento"""
        total_operations = self.storage_operations
        success_rate = ((total_operations - self.storage_errors) / total_operations * 100) if total_operations > 0 else 100.0
        avg_storage_time = self.total_storage_time / len(self.storage_times) if self.storage_times else 0.0
        avg_bytes_per_image = self.bytes_stored / self.images_stored if self.images_stored > 0 else 0.0
        
        return {
            'storage_summary': {
                'images_stored': self.images_stored,
                'thumbnails_stored': self.thumbnails_stored,
                'total_storage_operations': self.storage_operations,
                'storage_errors': self.storage_errors,
                'duplicate_images_avoided': self.duplicate_images_avoided,
                'documents_processed': self.documents_processed,
                'success_rate_percent': round(success_rate, 2)
            },
            'performance_metrics': {
                'total_storage_time_seconds': round(self.total_storage_time, 2),
                'avg_storage_time_ms': round(avg_storage_time * 1000, 2),
                'total_bytes_stored': self.bytes_stored,
                'avg_bytes_per_image': round(avg_bytes_per_image, 2),
                'max_storage_time_ms': round(max(self.storage_times) * 1000 if self.storage_times else 0, 2),
                'min_storage_time_ms': round(min(self.storage_times) * 1000 if self.storage_times else 0, 2)
            },
            'maintenance_metrics': {
                'cleanup_operations': self.cleanup_operations,
                'orphaned_files_cleaned': self.orphaned_files_cleaned
            },
            'error_breakdown': self.errors_by_type
        }

# Instancia global de métricas de almacenamiento
image_storage_metrics = ImageStorageMetrics()

# Configuración de seguridad y validación
MAX_FILENAME_LENGTH = 255
MAX_PATH_COMPONENTS = 20
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
DANGEROUS_PATTERNS = [
    r'\.\./',  # Directory traversal
    r'\.\.',   # Parent directory
    r'^/',     # Absolute path
    r'^~',     # Home directory
    r'[<>:"|?*]',  # Windows forbidden chars
    r'\x00',   # Null bytes
]

class ImageStorageError(Exception):
    """Excepción personalizada para errores de almacenamiento de imágenes"""
    pass

class PathValidator:
    """Clase para validación y sanitización de rutas de archivos"""
    
    def __init__(self, base_dir: str):
        """
        Inicializa el validador de rutas
        
        Args:
            base_dir: Directorio base permitido para almacenamiento
        """
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Verificar que el directorio base es escribible
        if not os.access(self.base_dir, os.W_OK):
            raise ImageStorageError(f"Directorio base no escribible: {self.base_dir}")
    
    def validate_filename(self, filename: str) -> bool:
        """
        Valida que un nombre de archivo sea seguro
        
        Args:
            filename: Nombre del archivo a validar
            
        Returns:
            True si el filename es válido y seguro
            
        Raises:
            ImageStorageError: Si el filename es peligroso
        """
        if not filename or len(filename.strip()) == 0:
            raise ImageStorageError("Nombre de archivo vacío")
        
        if len(filename) > MAX_FILENAME_LENGTH:
            raise ImageStorageError(f"Nombre de archivo muy largo: {len(filename)} > {MAX_FILENAME_LENGTH}")
        
        # Verificar patrones peligrosos
        for pattern in DANGEROUS_PATTERNS:
            if re.search(pattern, filename):
                raise ImageStorageError(f"Nombre de archivo contiene patrón peligroso: {filename}")
        
        # Verificar caracteres de control
        if any(ord(c) < 32 for c in filename):
            raise ImageStorageError(f"Nombre de archivo contiene caracteres de control: {filename}")
        
        return True
    
    def validate_extension(self, filename: str) -> bool:
        """
        Valida que la extensión del archivo esté permitida
        
        Args:
            filename: Nombre del archivo con extensión
            
        Returns:
            True si la extensión está permitida
            
        Raises:
            ImageStorageError: Si la extensión no está permitida
        """
        ext = Path(filename).suffix.lower().lstrip('.')
        
        if not ext:
            raise ImageStorageError(f"Archivo sin extensión: {filename}")
        
        if ext not in ALLOWED_EXTENSIONS:
            raise ImageStorageError(f"Extensión no permitida: {ext}. Permitidas: {ALLOWED_EXTENSIONS}")
        
        return True
    
    def validate_doc_id(self, doc_id: str) -> bool:
        """
        Valida que un doc_id sea seguro para usar como nombre de directorio
        
        Args:
            doc_id: ID del documento a validar
            
        Returns:
            True si el doc_id es válido
            
        Raises:
            ImageStorageError: Si el doc_id es peligroso
        """
        if not doc_id or len(doc_id.strip()) == 0:
            raise ImageStorageError("doc_id vacío")
        
        if len(doc_id) > 100:  # Límite razonable para doc_id
            raise ImageStorageError(f"doc_id muy largo: {len(doc_id)} > 100")
        
        # Verificar patrones peligrosos
        for pattern in DANGEROUS_PATTERNS:
            if re.search(pattern, doc_id):
                raise ImageStorageError(f"doc_id contiene patrón peligroso: {doc_id}")
        
        # Verificar que solo contenga caracteres seguros
        safe_pattern = r'^[a-zA-Z0-9_\-\.]+$'
        if not re.match(safe_pattern, doc_id):
            raise ImageStorageError(f"doc_id contiene caracteres no seguros: {doc_id}")
        
        return True
    
    def sanitize_path_component(self, component: str) -> str:
        """
        Sanitiza un componente de ruta eliminando caracteres peligrosos
        
        Args:
            component: Componente de ruta a sanitizar
            
        Returns:
            Componente sanitizado
        """
        # Eliminar caracteres peligrosos
        sanitized = re.sub(r'[<>:"|?*\x00-\x1f]', '_', component)
        
        # Eliminar puntos iniciales y finales
        sanitized = sanitized.strip('.')
        
        # Truncar si es muy largo
        if len(sanitized) > MAX_FILENAME_LENGTH:
            sanitized = sanitized[:MAX_FILENAME_LENGTH]
        
        return sanitized
    
    def resolve_safe_path(self, *path_components: str) -> Path:
        """
        Resuelve una ruta segura dentro del directorio base
        
        Args:
            *path_components: Componentes de la ruta a resolver
            
        Returns:
            Path resuelto y validado
            
        Raises:
            ImageStorageError: Si la ruta resulta fuera del directorio base
        """
        if len(path_components) > MAX_PATH_COMPONENTS:
            raise ImageStorageError(f"Demasiados componentes de ruta: {len(path_components)}")
        
        # Sanitizar cada componente
        safe_components = []
        for component in path_components:
            if component:  # Saltar componentes vacíos
                safe_component = self.sanitize_path_component(str(component))
                if safe_component:  # Solo agregar si no quedó vacío
                    safe_components.append(safe_component)
        
        # Construir ruta dentro del directorio base
        target_path = self.base_dir
        for component in safe_components:
            target_path = target_path / component
        
        # Resolver la ruta para eliminar .. y .
        resolved_path = target_path.resolve()
        
        # Verificar que la ruta resuelta esté dentro del directorio base
        try:
            resolved_path.relative_to(self.base_dir)
        except ValueError:
            raise ImageStorageError(f"Ruta fuera del directorio base: {resolved_path}")
        
        return resolved_path

class ImageStorage:
    """
    Clase principal para gestión de almacenamiento de imágenes
    
    Maneja el almacenamiento seguro y organizado de imágenes y thumbnails
    siguiendo la estructura: /var/data/rag/images/{doc_id}/p{page_number}/{hash}.{ext}
    """
    
    def __init__(self, base_dir: str = "/var/data/rag/images"):
        """
        Inicializa el sistema de almacenamiento de imágenes
        
        Args:
            base_dir: Directorio base para almacenamiento
        """
        self.base_dir = Path(base_dir)
        self.validator = PathValidator(base_dir)
        
        logger.info(f"ImageStorage inicializado: {self.base_dir}")
    
    def get_document_path(self, doc_id: str) -> Path:
        """
        Obtiene la ruta del directorio de un documento
        
        Args:
            doc_id: ID del documento
            
        Returns:
            Path del directorio del documento
        """
        self.validator.validate_doc_id(doc_id)
        return self.validator.resolve_safe_path(doc_id)
    
    def get_page_path(self, doc_id: str, page_number: int) -> Path:
        """
        Obtiene la ruta del directorio de una página
        
        Args:
            doc_id: ID del documento
            page_number: Número de página (1-indexed)
            
        Returns:
            Path del directorio de la página
        """
        if page_number < 1 or page_number > 9999:
            raise ImageStorageError(f"Número de página inválido: {page_number}")
        
        return self.validator.resolve_safe_path(doc_id, f"p{page_number}")
    
    def get_image_path(self, doc_id: str, page_number: int, image_hash: str, extension: str) -> Path:
        """
        Obtiene la ruta completa de una imagen
        
        Args:
            doc_id: ID del documento
            page_number: Número de página
            image_hash: Hash de la imagen
            extension: Extensión del archivo
            
        Returns:
            Path completo de la imagen
        """
        filename = f"{image_hash}.{extension.lower()}"
        self.validator.validate_filename(filename)
        self.validator.validate_extension(filename)
        
        page_path = self.get_page_path(doc_id, page_number)
        return page_path / filename
    
    def get_thumbnail_path(self, doc_id: str, page_number: int, image_hash: str) -> Path:
        """
        Obtiene la ruta completa de un thumbnail
        
        Args:
            doc_id: ID del documento
            page_number: Número de página
            image_hash: Hash de la imagen
            
        Returns:
            Path completo del thumbnail
        """
        page_path = self.get_page_path(doc_id, page_number)
        thumbs_dir = page_path / "thumbs"
        filename = f"{image_hash}.jpg"
        
        return thumbs_dir / filename
    
    def generate_local_uri(self, doc_id: str, page_number: int, image_hash: str, 
                          extension: str, is_thumbnail: bool = False) -> str:
        """
        Genera URI local para acceder a una imagen
        
        Args:
            doc_id: ID del documento
            page_number: Número de página
            image_hash: Hash de la imagen
            extension: Extensión de la imagen
            is_thumbnail: Si es un thumbnail (usa .jpg y ruta thumbs/)
            
        Returns:
            URI local en formato local://images/{doc_id}/p{page}/...
        """
        # Validar parámetros
        self.validator.validate_doc_id(doc_id)
        
        if is_thumbnail:
            path_part = f"{doc_id}/p{page_number}/thumbs/{image_hash}.jpg"
        else:
            self.validator.validate_extension(f"dummy.{extension}")
            path_part = f"{doc_id}/p{page_number}/{image_hash}.{extension.lower()}"
        
        # URL encode para caracteres especiales
        encoded_path = quote(path_part, safe='/')
        
        return f"local://images/{encoded_path}"
    
    def create_directory_structure(self, doc_id: str, page_number: int) -> Tuple[Path, Path]:
        """
        Crea la estructura de directorios para almacenar imágenes
        
        Args:
            doc_id: ID del documento
            page_number: Número de página
            
        Returns:
            Tupla (page_path, thumbs_path) con los directorios creados
        """
        page_path = self.get_page_path(doc_id, page_number)
        thumbs_path = page_path / "thumbs"
        
        # Crear directorios con permisos apropiados
        page_path.mkdir(parents=True, exist_ok=True)
        thumbs_path.mkdir(exist_ok=True)
        
        # Establecer permisos (solo owner puede leer/escribir/ejecutar)
        os.chmod(page_path, 0o755)
        os.chmod(thumbs_path, 0o755)
        
        logger.debug(f"Estructura creada: {page_path}")
        
        return page_path, thumbs_path
    
    def image_exists(self, doc_id: str, page_number: int, image_hash: str, extension: str) -> bool:
        """
        Verifica si una imagen existe en el almacenamiento
        
        Args:
            doc_id: ID del documento
            page_number: Número de página
            image_hash: Hash de la imagen
            extension: Extensión del archivo
            
        Returns:
            True si la imagen existe
        """
        try:
            image_path = self.get_image_path(doc_id, page_number, image_hash, extension)
            return image_path.exists() and image_path.is_file()
        except ImageStorageError:
            return False
    
    def thumbnail_exists(self, doc_id: str, page_number: int, image_hash: str) -> bool:
        """
        Verifica si un thumbnail existe en el almacenamiento
        
        Args:
            doc_id: ID del documento
            page_number: Número de página
            image_hash: Hash de la imagen
            
        Returns:
            True si el thumbnail existe
        """
        try:
            thumb_path = self.get_thumbnail_path(doc_id, page_number, image_hash)
            return thumb_path.exists() and thumb_path.is_file()
        except ImageStorageError:
            return False
    
    def get_document_images(self, doc_id: str) -> List[Dict[str, Union[str, int]]]:
        """
        Obtiene información de todas las imágenes almacenadas de un documento
        
        Args:
            doc_id: ID del documento
            
        Returns:
            Lista de diccionarios con información de cada imagen
        """
        try:
            doc_path = self.get_document_path(doc_id)
        except ImageStorageError as e:
            logger.warning(f"doc_id inválido {doc_id}: {e}")
            return []
        
        if not doc_path.exists():
            return []
        
        images_info = []
        
        # Buscar todas las páginas (directorios p*)
        for page_dir in doc_path.glob("p*"):
            if not page_dir.is_dir():
                continue
            
            try:
                # Extraer número de página
                page_number = int(page_dir.name[1:])
            except ValueError:
                logger.warning(f"Directorio con formato inválido: {page_dir}")
                continue
            
            # Buscar imágenes en la página
            for image_file in page_dir.glob("*"):
                if not image_file.is_file():
                    continue
                
                # Verificar extensión permitida
                ext = image_file.suffix.lower().lstrip('.')
                if ext not in ALLOWED_EXTENSIONS:
                    continue
                
                # Extraer hash del nombre del archivo
                image_hash = image_file.stem
                
                # Verificar si existe thumbnail
                thumbnail_path = page_dir / "thumbs" / f"{image_hash}.jpg"
                
                image_info = {
                    'doc_id': doc_id,
                    'page_number': page_number,
                    'hash': image_hash,
                    'format': ext.upper(),
                    'image_path': str(image_file),
                    'thumbnail_path': str(thumbnail_path) if thumbnail_path.exists() else None,
                    'image_uri': self.generate_local_uri(doc_id, page_number, image_hash, ext),
                    'thumbnail_uri': self.generate_local_uri(doc_id, page_number, image_hash, ext, is_thumbnail=True) if thumbnail_path.exists() else None,
                    'file_size': image_file.stat().st_size if image_file.exists() else 0
                }
                
                images_info.append(image_info)
        
        # Ordenar por página y hash
        return sorted(images_info, key=lambda x: (x['page_number'], x['hash']))
    
    def cleanup_document(self, doc_id: str) -> bool:
        """
        Elimina todas las imágenes y directorios de un documento
        
        Args:
            doc_id: ID del documento a limpiar
            
        Returns:
            True si la limpieza fue exitosa
        """
        try:
            doc_path = self.get_document_path(doc_id)
            
            if not doc_path.exists():
                logger.debug(f"No hay imágenes que limpiar para documento {doc_id}")
                return True
            
            # Eliminar directorio completo del documento
            shutil.rmtree(doc_path)
            
            logger.info(f"Limpiadas imágenes del documento {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error limpiando imágenes del documento {doc_id}: {e}")
            return False
    
    def get_storage_stats(self) -> Dict[str, Union[int, str]]:
        """
        Obtiene estadísticas del almacenamiento
        
        Returns:
            Diccionario con estadísticas de uso
        """
        if not self.base_dir.exists():
            return {
                'total_documents': 0,
                'total_images': 0,
                'total_thumbnails': 0,
                'storage_size_bytes': 0,
                'storage_path': str(self.base_dir)
            }
        
        total_docs = 0
        total_images = 0
        total_thumbnails = 0
        total_size = 0
        
        for doc_dir in self.base_dir.iterdir():
            if not doc_dir.is_dir():
                continue
            
            total_docs += 1
            
            for page_dir in doc_dir.glob("p*"):
                if not page_dir.is_dir():
                    continue
                
                # Contar imágenes principales
                for img_file in page_dir.glob("*"):
                    if img_file.is_file() and img_file.suffix.lower().lstrip('.') in ALLOWED_EXTENSIONS:
                        total_images += 1
                        total_size += img_file.stat().st_size
                
                # Contar thumbnails
                thumbs_dir = page_dir / "thumbs"
                if thumbs_dir.exists():
                    for thumb_file in thumbs_dir.glob("*.jpg"):
                        if thumb_file.is_file():
                            total_thumbnails += 1
                            total_size += thumb_file.stat().st_size
        
        return {
            'total_documents': total_docs,
            'total_images': total_images,
            'total_thumbnails': total_thumbnails,
            'storage_size_bytes': total_size,
            'storage_path': str(self.base_dir)
        }
    
    def save_image(self, doc_id: str, page_number: int, image_hash: str, 
                   image_data: bytes, extension: str, 
                   thumbnail_data: Optional[bytes] = None) -> Dict[str, str]:
        """
        Guarda una imagen y su thumbnail en la estructura organizada
        
        Args:
            doc_id: ID del documento
            page_number: Número de página
            image_hash: Hash SHA-256 de la imagen
            image_data: Datos binarios de la imagen
            extension: Extensión del archivo (png, jpg, webp)
            thumbnail_data: Datos binarios del thumbnail (opcional)
            
        Returns:
            Diccionario con rutas e URIs generados
            
        Raises:
            ImageStorageError: Si hay error guardando los archivos
        """
        start_time = time.time()
        was_duplicate = False
        
        try:
            # Validar parámetros de entrada
            self.validator.validate_doc_id(doc_id)
            if not image_data or len(image_data) == 0:
                raise ImageStorageError("Datos de imagen vacíos")
            
            # Verificar si la imagen ya existe (deduplicación)
            image_path = self.get_image_path(doc_id, page_number, image_hash, extension)
            if image_path.exists():
                was_duplicate = True
                processing_time = time.time() - start_time
                
                # Registrar métrica de duplicado evitado
                image_storage_metrics.record_image_storage(
                    success=True,
                    processing_time=processing_time,
                    bytes_stored=len(image_data),
                    was_duplicate=True
                )
                
                logger.debug(f"Imagen duplicada evitada: {doc_id}/p{page_number}/{image_hash}.{extension}")
                
                # Generar URIs existentes
                image_uri = self.generate_local_uri(doc_id, page_number, image_hash, extension)
                thumbnail_path = self.get_thumbnail_path(doc_id, page_number, image_hash)
                thumbnail_uri = self.generate_local_uri(doc_id, page_number, image_hash, extension, is_thumbnail=True) if thumbnail_path.exists() else None
                
                return {
                    'image_path': str(image_path),
                    'image_uri': image_uri,
                    'thumbnail_path': str(thumbnail_path) if thumbnail_path.exists() else None,
                    'thumbnail_uri': thumbnail_uri,
                    'was_duplicate': True
                }
            
            # Crear estructura de directorios
            page_path, thumbs_path = self.create_directory_structure(doc_id, page_number)
            
            # Obtener rutas de archivos
            thumbnail_path = self.get_thumbnail_path(doc_id, page_number, image_hash)
            
            # Guardar imagen principal
            with open(image_path, 'wb') as f:
                f.write(image_data)
            
            # Establecer permisos de archivo (644 - owner read/write, group/other read)
            os.chmod(image_path, 0o644)
            
            logger.debug(f"Imagen guardada: {image_path}")
            
            # Generar URIs
            image_uri = self.generate_local_uri(doc_id, page_number, image_hash, extension)
            thumbnail_uri = None
            thumbnail_bytes = 0
            
            # Guardar thumbnail si se proporciona
            if thumbnail_data:
                with open(thumbnail_path, 'wb') as f:
                    f.write(thumbnail_data)
                
                os.chmod(thumbnail_path, 0o644)
                thumbnail_uri = self.generate_local_uri(doc_id, page_number, image_hash, extension, is_thumbnail=True)
                thumbnail_bytes = len(thumbnail_data)
                
                # Registrar métrica de thumbnail
                image_storage_metrics.record_thumbnail_storage(success=True, bytes_stored=thumbnail_bytes)
                
                logger.debug(f"Thumbnail guardado: {thumbnail_path}")
            
            processing_time = time.time() - start_time
            
            # Registrar métricas de almacenamiento exitoso
            image_storage_metrics.record_image_storage(
                success=True,
                processing_time=processing_time,
                bytes_stored=len(image_data),
                was_duplicate=False
            )
            
            # Retornar información de archivos guardados
            result = {
                'image_path': str(image_path),
                'image_uri': image_uri,
                'thumbnail_path': str(thumbnail_path) if thumbnail_data else None,
                'thumbnail_uri': thumbnail_uri,
                'was_duplicate': False
            }
            
            logger.info(f"Imagen guardada exitosamente: {doc_id}/p{page_number}/{image_hash}.{extension}")
            
            return result
            
        except PermissionError as e:
            processing_time = time.time() - start_time
            error_type = "PermissionError"
            
            # Registrar métricas de error
            image_storage_metrics.record_image_storage(success=False, processing_time=processing_time)
            image_storage_metrics.record_storage_error(error_type)
            
            logger.error(f"Error de permisos guardando imagen: {e}")
            raise ImageStorageError(f"Sin permisos para escribir: {e}")
        except OSError as e:
            processing_time = time.time() - start_time
            error_type = "OSError"
            
            # Registrar métricas de error
            image_storage_metrics.record_image_storage(success=False, processing_time=processing_time)
            image_storage_metrics.record_storage_error(error_type)
            
            logger.error(f"Error de filesystem: {e}")
            raise ImageStorageError(f"Error de filesystem: {e}")
        except Exception as e:
            processing_time = time.time() - start_time
            error_type = type(e).__name__
            
            # Registrar métricas de error
            image_storage_metrics.record_image_storage(success=False, processing_time=processing_time)
            image_storage_metrics.record_storage_error(error_type)
            
            logger.error(f"Error inesperado guardando imagen: {e}")
            raise ImageStorageError(f"Error guardando imagen: {e}")
    
    def save_images_batch(self, images: List[Dict]) -> List[Dict[str, str]]:
        """
        Guarda múltiples imágenes en lote
        
        Args:
            images: Lista de diccionarios con información de imágenes
                   Cada dict debe contener: doc_id, page_number, image_hash, 
                   image_data, extension, thumbnail_data (opcional)
        
        Returns:
            Lista de diccionarios con información de archivos guardados
        """
        results = []
        
        for i, image_info in enumerate(images):
            try:
                # Validar estructura del diccionario
                required_fields = ['doc_id', 'page_number', 'image_hash', 'image_data', 'extension']
                for field in required_fields:
                    if field not in image_info:
                        raise ImageStorageError(f"Campo requerido faltante en imagen {i}: {field}")
                
                # Guardar imagen individual
                result = self.save_image(
                    doc_id=image_info['doc_id'],
                    page_number=image_info['page_number'],
                    image_hash=image_info['image_hash'],
                    image_data=image_info['image_data'],
                    extension=image_info['extension'],
                    thumbnail_data=image_info.get('thumbnail_data')
                )
                
                # Agregar metadata adicional
                result.update({
                    'doc_id': image_info['doc_id'],
                    'page_number': image_info['page_number'],
                    'hash': image_info['image_hash'],
                    'format': image_info['extension'].upper(),
                    'batch_index': i
                })
                
                results.append(result)
                
            except ImageStorageError as e:
                logger.warning(f"Error guardando imagen {i}: {e}")
                # Continuar con las siguientes imágenes en caso de error
                results.append({
                    'error': str(e),
                    'batch_index': i,
                    'doc_id': image_info.get('doc_id', 'unknown'),
                    'page_number': image_info.get('page_number', 0),
                    'hash': image_info.get('image_hash', 'unknown')
                })
        
        successful_saves = len([r for r in results if 'error' not in r])
        
        # Registrar métricas de procesamiento de documento
        image_storage_metrics.record_document_processed()
        
        logger.info(f"Guardado en lote completado: {successful_saves}/{len(images)} imágenes guardadas")
        
        return results
    
    def ensure_permissions(self, doc_id: str) -> bool:
        """
        Asegura que los permisos estén correctamente establecidos para un documento
        
        Args:
            doc_id: ID del documento
            
        Returns:
            True si los permisos se establecieron correctamente
        """
        try:
            doc_path = self.get_document_path(doc_id)
            
            if not doc_path.exists():
                logger.debug(f"Documento {doc_id} no existe, no hay permisos que ajustar")
                return True
            
            # Establecer permisos para directorio del documento
            os.chmod(doc_path, 0o755)
            
            # Recorrer todas las páginas
            for page_dir in doc_path.glob("p*"):
                if page_dir.is_dir():
                    # Permisos para directorio de página
                    os.chmod(page_dir, 0o755)
                    
                    # Permisos para imágenes
                    for image_file in page_dir.glob("*"):
                        if image_file.is_file():
                            os.chmod(image_file, 0o644)
                    
                    # Permisos para directorio de thumbnails
                    thumbs_dir = page_dir / "thumbs"
                    if thumbs_dir.exists():
                        os.chmod(thumbs_dir, 0o755)
                        
                        # Permisos para thumbnails
                        for thumb_file in thumbs_dir.glob("*.jpg"):
                            if thumb_file.is_file():
                                os.chmod(thumb_file, 0o644)
            
            logger.debug(f"Permisos actualizados para documento {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error estableciendo permisos para documento {doc_id}: {e}")
            return False
    
    def cleanup_orphaned_files(self) -> Dict[str, int]:
        """
        Limpia archivos huérfanos que no siguen la estructura esperada
        
        Returns:
            Diccionario con estadísticas de limpieza
        """
        if not self.base_dir.exists():
            return {'orphaned_files': 0, 'orphaned_dirs': 0, 'errors': 0}
        
        orphaned_files = 0
        orphaned_dirs = 0
        errors = 0
        
        try:
            for item in self.base_dir.rglob("*"):
                try:
                    # Saltar el directorio base
                    if item == self.base_dir:
                        continue
                    
                    # Obtener componentes de la ruta relativa
                    relative_path = item.relative_to(self.base_dir)
                    parts = relative_path.parts
                    
                    is_orphaned = False
                    
                    if item.is_file():
                        # Validar estructura de archivo
                        if len(parts) < 2:
                            is_orphaned = True
                        elif len(parts) == 3:  # doc_id/p*/archivo
                            doc_id, page_dir, filename = parts
                            if not page_dir.startswith('p') or not page_dir[1:].isdigit():
                                is_orphaned = True
                            elif not self._is_valid_image_filename(filename):
                                is_orphaned = True
                        elif len(parts) == 4:  # doc_id/p*/thumbs/archivo.jpg
                            doc_id, page_dir, thumbs_dir, filename = parts
                            if (not page_dir.startswith('p') or not page_dir[1:].isdigit() or
                                thumbs_dir != 'thumbs' or not filename.endswith('.jpg')):
                                is_orphaned = True
                        else:
                            is_orphaned = True
                    
                    elif item.is_dir():
                        # Validar estructura de directorio
                        if len(parts) == 1:  # doc_id
                            doc_id = parts[0]
                            try:
                                self.validator.validate_doc_id(doc_id)
                            except ImageStorageError:
                                is_orphaned = True
                        elif len(parts) == 2:  # doc_id/p*
                            doc_id, page_dir = parts
                            if not page_dir.startswith('p') or not page_dir[1:].isdigit():
                                is_orphaned = True
                        elif len(parts) == 3:  # doc_id/p*/thumbs
                            doc_id, page_dir, thumbs_dir = parts
                            if thumbs_dir != 'thumbs':
                                is_orphaned = True
                        else:
                            is_orphaned = True
                    
                    # Eliminar si es huérfano
                    if is_orphaned:
                        if item.is_file():
                            item.unlink()
                            orphaned_files += 1
                            logger.debug(f"Archivo huérfano eliminado: {item}")
                        elif item.is_dir():
                            shutil.rmtree(item)
                            orphaned_dirs += 1
                            logger.debug(f"Directorio huérfano eliminado: {item}")
                
                except Exception as e:
                    logger.warning(f"Error procesando {item}: {e}")
                    errors += 1
        
        except Exception as e:
            logger.error(f"Error durante limpieza de huérfanos: {e}")
            errors += 1
        
        # Registrar métricas de limpieza
        total_orphaned = orphaned_files + orphaned_dirs
        image_storage_metrics.record_cleanup_operation(total_orphaned)
        
        result = {
            'orphaned_files': orphaned_files,
            'orphaned_dirs': orphaned_dirs,
            'errors': errors
        }
        
        if orphaned_files > 0 or orphaned_dirs > 0:
            logger.info(f"Limpieza de huérfanos completada: {orphaned_files} archivos, {orphaned_dirs} directorios eliminados")
        
        return result
    
    def _is_valid_image_filename(self, filename: str) -> bool:
        """
        Valida si un nombre de archivo sigue el patrón esperado para imágenes
        
        Args:
            filename: Nombre del archivo a validar
            
        Returns:
            True si el filename es válido
        """
        if not filename:
            return False
        
        # Verificar que tenga extensión permitida
        ext = Path(filename).suffix.lower().lstrip('.')
        if ext not in ALLOWED_EXTENSIONS:
            return False
        
        # Verificar que el nombre (sin extensión) sea un hash válido
        name_part = Path(filename).stem
        if not name_part or len(name_part) != 64:  # SHA-256 hex length
            return False
        
        # Verificar que solo contenga caracteres hex
        try:
            int(name_part, 16)
            return True
        except ValueError:
            return False
    
    def resolve_uri_to_path(self, uri: str) -> Optional[Path]:
        """
        Resuelve una URI local a una ruta de archivo en el filesystem
        
        Args:
            uri: URI local en formato local://images/...
            
        Returns:
            Path del archivo si existe, None si la URI es inválida o el archivo no existe
        """
        if not uri.startswith('local://images/'):
            logger.warning(f"URI inválida (no es local://images/): {uri}")
            return None
        
        try:
            # Extraer la parte de la ruta después de local://images/
            path_part = uri[len('local://images/'):]
            
            # Decodificar URL encoding
            from urllib.parse import unquote
            decoded_path = unquote(path_part)
            
            # Parsear componentes esperados: {doc_id}/p{page}/[thumbs/]{hash}.{ext}
            path_components = decoded_path.split('/')
            
            if len(path_components) < 3:
                logger.warning(f"URI con estructura inválida: {uri}")
                return None
            
            doc_id = path_components[0]
            page_dir = path_components[1]
            
            # Validar doc_id
            self.validator.validate_doc_id(doc_id)
            
            # Validar formato de página
            if not page_dir.startswith('p') or not page_dir[1:].isdigit():
                logger.warning(f"Directorio de página inválido: {page_dir}")
                return None
            
            page_number = int(page_dir[1:])
            
            # Determinar si es thumbnail o imagen principal
            if len(path_components) == 4 and path_components[2] == 'thumbs':
                # Es thumbnail
                filename = path_components[3]
                if not filename.endswith('.jpg'):
                    logger.warning(f"Thumbnail debe ser .jpg: {filename}")
                    return None
                
                image_hash = Path(filename).stem
                resolved_path = self.get_thumbnail_path(doc_id, page_number, image_hash)
                
            elif len(path_components) == 3:
                # Es imagen principal
                filename = path_components[2]
                
                # Extraer hash y extensión
                image_hash = Path(filename).stem
                extension = Path(filename).suffix.lstrip('.')
                
                resolved_path = self.get_image_path(doc_id, page_number, image_hash, extension)
            
            else:
                logger.warning(f"Estructura de URI no reconocida: {uri}")
                return None
            
            # Verificar que el archivo existe
            if resolved_path.exists() and resolved_path.is_file():
                return resolved_path
            else:
                logger.debug(f"Archivo no existe para URI {uri}: {resolved_path}")
                return None
                
        except Exception as e:
            logger.warning(f"Error resolviendo URI {uri}: {e}")
            return None
    
    def get_image_info(self, doc_id: str, page_number: int, image_hash: str, 
                      extension: Optional[str] = None) -> Optional[Dict]:
        """
        Obtiene información detallada de una imagen específica
        
        Args:
            doc_id: ID del documento
            page_number: Número de página
            image_hash: Hash de la imagen
            extension: Extensión (se detecta automáticamente si no se proporciona)
            
        Returns:
            Diccionario con metadata de la imagen o None si no existe
        """
        try:
            # Si no se proporciona extensión, buscar en todas las permitidas
            if not extension:
                for ext in ALLOWED_EXTENSIONS:
                    if self.image_exists(doc_id, page_number, image_hash, ext):
                        extension = ext
                        break
                
                if not extension:
                    logger.debug(f"Imagen no encontrada: {doc_id}/p{page_number}/{image_hash}")
                    return None
            
            # Obtener rutas de archivos
            image_path = self.get_image_path(doc_id, page_number, image_hash, extension)
            thumbnail_path = self.get_thumbnail_path(doc_id, page_number, image_hash)
            
            # Verificar existencia
            if not image_path.exists():
                return None
            
            # Obtener estadísticas del archivo
            image_stat = image_path.stat()
            
            # Información básica
            info = {
                'doc_id': doc_id,
                'page_number': page_number,
                'hash': image_hash,
                'format': extension.upper(),
                'image_path': str(image_path),
                'image_uri': self.generate_local_uri(doc_id, page_number, image_hash, extension),
                'file_size': image_stat.st_size,
                'created_time': image_stat.st_ctime,
                'modified_time': image_stat.st_mtime,
                'exists': True
            }
            
            # Información del thumbnail
            if thumbnail_path.exists():
                thumb_stat = thumbnail_path.stat()
                info.update({
                    'thumbnail_path': str(thumbnail_path),
                    'thumbnail_uri': self.generate_local_uri(doc_id, page_number, image_hash, extension, is_thumbnail=True),
                    'thumbnail_size': thumb_stat.st_size,
                    'has_thumbnail': True
                })
            else:
                info.update({
                    'thumbnail_path': None,
                    'thumbnail_uri': None,
                    'thumbnail_size': 0,
                    'has_thumbnail': False
                })
            
            # Intentar obtener dimensiones de la imagen usando PIL
            try:
                from PIL import Image
                with Image.open(image_path) as img:
                    info.update({
                        'width': img.width,
                        'height': img.height,
                        'mode': img.mode,
                        'has_dimensions': True
                    })
            except Exception as e:
                logger.debug(f"No se pudieron obtener dimensiones de {image_path}: {e}")
                info.update({
                    'width': None,
                    'height': None,
                    'mode': None,
                    'has_dimensions': False
                })
            
            return info
            
        except Exception as e:
            logger.error(f"Error obteniendo información de imagen {doc_id}/p{page_number}/{image_hash}: {e}")
            return None
    
    def file_exists(self, uri_or_path: Union[str, Path]) -> bool:
        """
        Verifica si un archivo existe dado una URI local o ruta de archivo
        
        Args:
            uri_or_path: URI local o ruta de archivo
            
        Returns:
            True si el archivo existe
        """
        if isinstance(uri_or_path, str) and uri_or_path.startswith('local://'):
            # Es una URI, resolverla a path
            resolved_path = self.resolve_uri_to_path(uri_or_path)
            return resolved_path is not None and resolved_path.exists()
        
        else:
            # Es una ruta de archivo
            try:
                path = Path(uri_or_path)
                return path.exists() and path.is_file()
            except Exception:
                return False
    
    def get_storage_metrics(self) -> Dict:
        """
        Retorna métricas actuales de almacenamiento de imágenes
        
        Returns:
            Diccionario con métricas de almacenamiento
        """
        return image_storage_metrics.get_metrics_summary()