"""
Módulo de caché de descripciones de imágenes con deduplicación por checksum SHA-256.

Este módulo proporciona funcionalidades para:
- Calcular checksums SHA-256 de imágenes (archivos y datos base64)
- Cachear descripciones de imágenes para evitar regeneración
- Persistencia de caché con almacenamiento JSON
- Métricas de rendimiento y thread-safety
- Limpieza y mantenimiento de caché
"""

import hashlib
import logging
import threading
import time
import base64
import os
from typing import Dict, Any, Optional, Union, Tuple, List
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Entrada de caché para una descripción de imagen"""
    checksum: str
    description: str
    image_path: Optional[str] = None
    image_metadata: Optional[Dict[str, Any]] = None
    created_at: str = None
    accessed_at: str = None
    access_count: int = 0
    generation_time: float = 0.0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.accessed_at is None:
            self.accessed_at = self.created_at


@dataclass
class CacheMetrics:
    """Métricas completas de rendimiento del caché"""
    # Métricas básicas
    hits: int = 0
    misses: int = 0
    cache_size: int = 0
    total_generation_time_saved: float = 0.0
    
    # Métricas de operaciones
    cache_gets: int = 0
    cache_puts: int = 0
    cache_invalidations: int = 0
    cache_saves: int = 0
    cache_loads: int = 0
    
    # Métricas temporales
    total_get_time: float = 0.0
    total_put_time: float = 0.0
    total_save_time: float = 0.0
    total_load_time: float = 0.0
    
    # Métricas de validación
    validation_successes: int = 0
    validation_failures: int = 0
    checksum_mismatches: int = 0
    expired_entries: int = 0
    collisions_detected: int = 0
    force_overrides: int = 0
    
    # Métricas de limpieza
    lru_cleanups: int = 0
    entries_removed_lru: int = 0
    integrity_checks: int = 0
    last_cleanup: Optional[str] = None
    
    # Timestamps
    created_at: str = None
    last_reset: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def hit_ratio(self) -> float:
        """Calcular ratio de aciertos del caché"""
        total = self.hits + self.misses
        return (self.hits / total) if total > 0 else 0.0
    
    def avg_get_time(self) -> float:
        """Tiempo promedio de operaciones get"""
        return (self.total_get_time / self.cache_gets) if self.cache_gets > 0 else 0.0
    
    def avg_put_time(self) -> float:
        """Tiempo promedio de operaciones put"""
        return (self.total_put_time / self.cache_puts) if self.cache_puts > 0 else 0.0
    
    def avg_save_time(self) -> float:
        """Tiempo promedio de operaciones save"""
        return (self.total_save_time / self.cache_saves) if self.cache_saves > 0 else 0.0
    
    def avg_load_time(self) -> float:
        """Tiempo promedio de operaciones load"""
        return (self.total_load_time / self.cache_loads) if self.cache_loads > 0 else 0.0
    
    def validation_success_rate(self) -> float:
        """Ratio de éxito en validaciones"""
        total_validations = self.validation_successes + self.validation_failures
        return (self.validation_successes / total_validations) if total_validations > 0 else 0.0
    
    def reset_metrics(self) -> None:
        """Reiniciar todas las métricas manteniendo timestamps"""
        created_at = self.created_at
        self.__init__()
        self.created_at = created_at
        self.last_reset = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir métricas a diccionario con calculados"""
        base_metrics = asdict(self)
        calculated_metrics = {
            'hit_ratio': self.hit_ratio(),
            'avg_get_time_ms': self.avg_get_time() * 1000,
            'avg_put_time_ms': self.avg_put_time() * 1000,
            'avg_save_time_ms': self.avg_save_time() * 1000,
            'avg_load_time_ms': self.avg_load_time() * 1000,
            'validation_success_rate': self.validation_success_rate(),
            'total_operations': self.cache_gets + self.cache_puts,
            'efficiency_score': self._calculate_efficiency_score()
        }
        
        return {**base_metrics, **calculated_metrics}
    
    def _calculate_efficiency_score(self) -> float:
        """Calcular puntuación de eficiencia del caché (0-100)"""
        # Factores que contribuyen a la eficiencia
        hit_ratio = self.hit_ratio()
        validation_success_rate = self.validation_success_rate()
        collision_rate = (self.collisions_detected / max(1, self.cache_puts))
        
        # Puntuación base del hit ratio (60% del peso)
        hit_score = hit_ratio * 60
        
        # Puntuación de validación (25% del peso)
        validation_score = validation_success_rate * 25
        
        # Penalización por colisiones (15% del peso)
        collision_penalty = min(collision_rate * 15, 15)
        
        return max(0, hit_score + validation_score - collision_penalty)


class DescriptionCache:
    """
    Sistema de caché para descripciones de imágenes con deduplicación por checksum SHA-256.
    
    Proporciona:
    - Cálculo de checksums SHA-256 para archivos e imágenes base64
    - Almacenamiento persistente en JSON
    - Thread-safety para acceso concurrente
    - Métricas de rendimiento
    - Limpieza automática de caché
    """
    
    DEFAULT_CACHE_PATH = "/var/data/rag/description_cache.json"
    DEFAULT_MAX_CACHE_SIZE = 10000
    DEFAULT_MAX_AGE_DAYS = 30
    SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
    BACKUP_SUFFIX = ".backup"
    TEMP_SUFFIX = ".tmp"
    
    def __init__(self, 
                 cache_path: Optional[str] = None,
                 max_cache_size: int = DEFAULT_MAX_CACHE_SIZE,
                 max_age_days: int = DEFAULT_MAX_AGE_DAYS,
                 auto_save: bool = True,
                 enabled: bool = True):
        """
        Inicializar caché de descripciones de imágenes.
        
        Args:
            cache_path: Ruta del archivo de caché JSON (default: /var/data/rag/description_cache.json)
            max_cache_size: Tamaño máximo del caché (entradas)
            max_age_days: Edad máxima de entradas en días
            auto_save: Si guardar automáticamente cambios al disco
            enabled: Si el caché está habilitado (default: True)
        """
        # Configuración desde variables de entorno
        env_cache_path = os.getenv('DESCRIPTION_CACHE_PATH')
        env_cache_enabled = os.getenv('DESCRIPTION_CACHE_ENABLED', 'true').lower() in ('true', '1', 'yes')
        
        self.cache_path = cache_path or env_cache_path or self.DEFAULT_CACHE_PATH
        self.backup_path = f"{self.cache_path}{self.BACKUP_SUFFIX}"
        self.temp_path = f"{self.cache_path}{self.TEMP_SUFFIX}"
        self.max_cache_size = max_cache_size
        self.max_age_days = max_age_days
        self.auto_save = auto_save
        self.enabled = enabled and env_cache_enabled
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Cache en memoria y métricas
        self._cache: Dict[str, CacheEntry] = {}
        self._metrics = CacheMetrics()
        
        # Inicializar directorio de caché si no existe
        self._ensure_cache_directory()
        
        # Cargar caché existente desde disco
        self._load_cache()
        
        logger.info(f"DescriptionCache initialized: path={self.cache_path}, "
                   f"max_size={self.max_cache_size}, max_age={self.max_age_days} days, "
                   f"loaded_entries={len(self._cache)}")
    
    def _ensure_cache_directory(self) -> None:
        """Crear directorio de caché si no existe"""
        cache_dir = Path(self.cache_path).parent
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Cache directory ensured: {cache_dir}")
    
    def compute_checksum(self, image_input: Union[str, bytes]) -> str:
        """
        Calcular checksum SHA-256 de una imagen.
        
        Args:
            image_input: Ruta del archivo de imagen, datos binarios, o datos base64 con data URI
            
        Returns:
            str: Checksum SHA-256 en formato hexadecimal
            
        Raises:
            ValueError: Si el input no es válido
        """
        try:
            if isinstance(image_input, str):
                # Determinar si es ruta de archivo o datos base64
                if image_input.startswith('data:image/'):
                    # Es datos base64 con data URI
                    return self._compute_checksum_from_base64(image_input)
                else:
                    # Es ruta de archivo
                    return self._compute_checksum_from_file(image_input)
            elif isinstance(image_input, bytes):
                # Datos binarios directos
                return hashlib.sha256(image_input).hexdigest()
            else:
                raise ValueError(f"Unsupported input type: {type(image_input)}")
        
        except Exception as e:
            logger.error(f"Error computing checksum: {str(e)}")
            raise ValueError(f"Failed to compute checksum: {str(e)}")
    
    def _compute_checksum_from_file(self, file_path: str) -> str:
        """
        Calcular checksum desde archivo de imagen.
        
        Args:
            file_path: Ruta del archivo de imagen
            
        Returns:
            str: Checksum SHA-256
            
        Raises:
            ValueError: Si el archivo no es válido
        """
        # Validar archivo usando patrón de ImageDescriptor
        path = Path(file_path)
        
        if not path.exists():
            raise ValueError(f"Image file does not exist: {file_path}")
        
        if not path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        
        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported image format: {path.suffix}. "
                           f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}")
        
        # Leer archivo y calcular hash
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            
            checksum = hashlib.sha256(content).hexdigest()
            logger.debug(f"Computed checksum for file {file_path}: {checksum[:16]}...")
            return checksum
            
        except Exception as e:
            raise ValueError(f"Failed to read image file {file_path}: {str(e)}")
    
    def _compute_checksum_from_base64(self, image_data: str) -> str:
        """
        Calcular checksum desde datos base64.
        
        Args:
            image_data: Imagen en formato base64 con data URI
            
        Returns:
            str: Checksum SHA-256
            
        Raises:
            ValueError: Si los datos base64 no son válidos
        """
        # Validar formato data URI
        if not image_data.startswith('data:image/'):
            raise ValueError("Image data must be in data URI format (data:image/...)")
        
        if ';base64,' not in image_data:
            raise ValueError("Image data must contain base64 encoding")
        
        try:
            # Extraer datos base64
            _, base64_data = image_data.split(';base64,')
            if not base64_data:
                raise ValueError("Base64 data is empty")
            
            # Decodificar y calcular hash
            image_bytes = base64.b64decode(base64_data)
            checksum = hashlib.sha256(image_bytes).hexdigest()
            logger.debug(f"Computed checksum for base64 data: {checksum[:16]}...")
            return checksum
            
        except Exception as e:
            raise ValueError(f"Failed to process base64 image data: {str(e)}")
    
    def get_description(self, checksum: str, 
                       validate_image: Optional[Union[str, bytes]] = None,
                       force_validation: bool = False) -> Optional[str]:
        """
        Obtener descripción desde el caché con validación opcional de imagen.
        
        Args:
            checksum: Checksum SHA-256 de la imagen
            validate_image: Imagen para validar (ruta de archivo, datos binarios o base64)
            force_validation: Forzar validación incluso si la entrada es reciente
            
        Returns:
            Descripción cacheada o None si no existe/es inválida
        """
        start_time = time.time()
        
        with self._lock:
            # Si el caché está deshabilitado, devolver None inmediatamente
            if not self.enabled:
                logger.debug(f"Cache disabled, skipping get for checksum {checksum[:16]}...")
                return None
                
            self._metrics.cache_gets += 1
            
            entry = self._cache.get(checksum)
            if not entry:
                self._metrics.misses += 1
                operation_time = time.time() - start_time
                self._metrics.total_get_time += operation_time
                logger.debug(f"Cache miss for checksum {checksum[:16]}...")
                return None
            
            # Validación de imagen si se proporciona
            if validate_image is not None:
                try:
                    # Calcular checksum actual de la imagen
                    current_checksum = self.compute_checksum(validate_image)
                    
                    # Verificar consistencia
                    if current_checksum != checksum:
                        logger.warning(f"Image checksum mismatch! Expected: {checksum[:16]}..., "
                                     f"Got: {current_checksum[:16]}... Invalidating cache entry.")
                        
                        # Actualizar métricas
                        self._metrics.checksum_mismatches += 1
                        self._metrics.validation_failures += 1
                        self._metrics.cache_invalidations += 1
                        
                        # Invalidar entrada corrupta
                        del self._cache[checksum]
                        self._metrics.cache_size = len(self._cache)
                        self._metrics.misses += 1
                        
                        operation_time = time.time() - start_time
                        self._metrics.total_get_time += operation_time
                        
                        # Auto-save si está habilitado
                        if self.auto_save:
                            self._save_cache()
                        
                        return None
                    else:
                        self._metrics.validation_successes += 1
                        
                except Exception as validation_error:
                    logger.error(f"Image validation failed for checksum {checksum[:16]}...: {validation_error}")
                    self._metrics.validation_failures += 1
                    # En caso de error de validación, devolver entrada cacheada pero registrar el problema
                    
            # Validación temporal si se fuerza
            if force_validation:
                # Verificar edad de la entrada
                try:
                    created_time = datetime.fromisoformat(entry.created_at.replace('Z', '+00:00'))
                    age_days = (datetime.now() - created_time).days
                    
                    if age_days > self.max_age_days:
                        logger.info(f"Cache entry expired (age: {age_days} days > {self.max_age_days}). "
                                  f"Checksum: {checksum[:16]}...")
                        
                        # Actualizar métricas
                        self._metrics.expired_entries += 1
                        self._metrics.cache_invalidations += 1
                        
                        # Eliminar entrada expirada
                        del self._cache[checksum]
                        self._metrics.cache_size = len(self._cache)
                        self._metrics.misses += 1
                        
                        operation_time = time.time() - start_time
                        self._metrics.total_get_time += operation_time
                        
                        # Auto-save si está habilitado
                        if self.auto_save:
                            self._save_cache()
                        
                        return None
                        
                except Exception as time_validation_error:
                    logger.warning(f"Time validation failed for checksum {checksum[:16]}...: {time_validation_error}")
                    # Continuar con la entrada si hay error de validación temporal
            
            # Actualizar estadísticas de acceso
            entry.accessed_at = datetime.now().isoformat()
            entry.access_count += 1
            self._metrics.hits += 1
            
            # Actualizar tiempo ahorrado en métricas
            if entry.generation_time > 0:
                self._metrics.total_generation_time_saved += entry.generation_time
            
            # Completar métricas de operación
            operation_time = time.time() - start_time
            self._metrics.total_get_time += operation_time
            
            logger.debug(f"Cache hit for checksum {checksum[:16]}... "
                       f"(accessed {entry.access_count} times)")
            return entry.description
    
    def put_description(self, 
                       checksum: str, 
                       description: str,
                       image_path: Optional[str] = None,
                       image_metadata: Optional[Dict[str, Any]] = None,
                       generation_time: float = 0.0,
                       force_override: bool = False,
                       validate_image: Optional[Union[str, bytes]] = None) -> bool:
        """
        Almacenar descripción en el caché con validación y control de colisiones.
        
        Args:
            checksum: Checksum SHA-256 de la imagen
            description: Descripción generada
            image_path: Ruta opcional del archivo de imagen
            image_metadata: Metadata opcional de la imagen
            generation_time: Tiempo que tomó generar la descripción
            force_override: Forzar override de entrada existente
            validate_image: Imagen para validar checksum (ruta, datos binarios o base64)
            
        Returns:
            bool: True si se almacenó correctamente, False si hubo conflicto sin override
        """
        start_time = time.time()
        
        with self._lock:
            # Si el caché está deshabilitado, simular éxito sin almacenar
            if not self.enabled:
                logger.debug(f"Cache disabled, skipping put for checksum {checksum[:16]}...")
                return True
                
            self._metrics.cache_puts += 1
            
            # Validación de imagen si se proporciona
            if validate_image is not None:
                try:
                    computed_checksum = self.compute_checksum(validate_image)
                    if computed_checksum != checksum:
                        logger.error(f"Checksum validation failed in put_description! "
                                   f"Expected: {checksum[:16]}..., "
                                   f"Got: {computed_checksum[:16]}...")
                        self._metrics.validation_failures += 1
                        operation_time = time.time() - start_time
                        self._metrics.total_put_time += operation_time
                        return False
                    else:
                        self._metrics.validation_successes += 1
                except Exception as validation_error:
                    logger.error(f"Image validation failed in put_description: {validation_error}")
                    self._metrics.validation_failures += 1
                    if not force_override:
                        operation_time = time.time() - start_time
                        self._metrics.total_put_time += operation_time
                        return False
            
            # Manejo de colisiones existentes
            existing_entry = self._cache.get(checksum)
            if existing_entry and not force_override:
                # Verificar si es el mismo contenido
                if existing_entry.description.strip() == description.strip():
                    # Mismo contenido, solo actualizar metadatos
                    existing_entry.accessed_at = datetime.now().isoformat()
                    existing_entry.access_count += 1
                    if image_path and not existing_entry.image_path:
                        existing_entry.image_path = image_path
                    if image_metadata and not existing_entry.image_metadata:
                        existing_entry.image_metadata = image_metadata
                    
                    logger.debug(f"Updated metadata for existing identical entry: {checksum[:16]}...")
                    
                    operation_time = time.time() - start_time
                    self._metrics.total_put_time += operation_time
                    
                    if self.auto_save:
                        self._save_cache()
                    return True
                else:
                    # Contenido diferente - colisión detectada
                    logger.warning(f"Cache collision detected for checksum {checksum[:16]}...! "
                                 f"Existing description: {len(existing_entry.description)} chars, "
                                 f"New description: {len(description)} chars. "
                                 f"Use force_override=True to replace.")
                    self._metrics.collisions_detected += 1
                    operation_time = time.time() - start_time
                    self._metrics.total_put_time += operation_time
                    return False
            
            # Crear nueva entrada de caché
            entry = CacheEntry(
                checksum=checksum,
                description=description,
                image_path=image_path,
                image_metadata=image_metadata,
                generation_time=generation_time
            )
            
            # Log apropiado según si es actualización o nueva entrada
            if existing_entry:
                if force_override:
                    logger.info(f"Force overriding cache entry for checksum {checksum[:16]}...")
                    self._metrics.force_overrides += 1
                else:
                    logger.debug(f"Updating cache entry for checksum {checksum[:16]}...")
            else:
                logger.debug(f"Adding new cache entry for checksum {checksum[:16]}...")
            
            self._cache[checksum] = entry
            self._metrics.cache_size = len(self._cache)
            
            # Limpieza automática si se excede el tamaño máximo
            if len(self._cache) > self.max_cache_size:
                self._cleanup_lru_entries()
            
            # Completar métricas de operación
            operation_time = time.time() - start_time
            self._metrics.total_put_time += operation_time
            
            # Auto-guardado si está habilitado
            if self.auto_save:
                self._save_cache()
            
            return True
    
    def _cleanup_lru_entries(self) -> None:
        """Limpiar entradas menos usadas (LRU) cuando se excede el tamaño máximo"""
        if len(self._cache) <= self.max_cache_size:
            return
        
        # Ordenar por último acceso (LRU)
        sorted_entries = sorted(
            self._cache.items(),
            key=lambda x: (x[1].accessed_at, x[1].access_count)
        )
        
        # Calcular cuántas entradas eliminar
        entries_to_remove = len(self._cache) - self.max_cache_size
        entries_to_remove = max(entries_to_remove, int(self.max_cache_size * 0.1))  # Al menos 10%
        
        # Eliminar entradas LRU
        for i in range(entries_to_remove):
            if i < len(sorted_entries):
                checksum_to_remove = sorted_entries[i][0]
                del self._cache[checksum_to_remove]
        
        self._metrics.cache_size = len(self._cache)
        self._metrics.lru_cleanups += 1
        self._metrics.entries_removed_lru += entries_to_remove
        self._metrics.last_cleanup = datetime.now().isoformat()
        
        logger.info(f"Cleaned up {entries_to_remove} LRU cache entries. "
                   f"Current size: {len(self._cache)}")
    
    def _load_cache(self) -> None:
        """Cargar caché desde archivo JSON con recuperación automática"""
        # Intentar cargar desde archivo principal
        loaded = self._try_load_cache_file(self.cache_path)
        
        # Si falla, intentar cargar desde backup
        if not loaded and Path(self.backup_path).exists():
            logger.warning(f"Primary cache corrupted, trying backup: {self.backup_path}")
            loaded = self._try_load_cache_file(self.backup_path)
            
            # Si backup funciona, restaurar archivo principal
            if loaded:
                try:
                    import shutil
                    shutil.copy2(self.backup_path, self.cache_path)
                    logger.info("Restored cache from backup successfully")
                except Exception as e:
                    logger.error(f"Failed to restore cache from backup: {e}")
        
        # Si no se pudo cargar nada, inicializar caché vacío
        if not loaded:
            logger.info("Starting with empty cache")
            self._cache = {}
            self._metrics = CacheMetrics()
    
    def _try_load_cache_file(self, file_path: str) -> bool:
        """Intentar cargar caché desde un archivo específico"""
        if not Path(file_path).exists():
            logger.debug(f"Cache file does not exist: {file_path}")
            return False
        
        try:
            # Validar integridad del archivo JSON
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if not content.strip():
                    raise ValueError("Empty cache file")
                
                data = json.loads(content)
            
            # Validar estructura del caché
            if not isinstance(data, dict) or 'cache' not in data:
                raise ValueError("Invalid cache file structure")
            
            # Cargar entradas de caché con validación
            cache_data = data.get('cache', {})
            loaded_entries = 0
            
            for checksum, entry_data in cache_data.items():
                try:
                    if not isinstance(entry_data, dict):
                        logger.warning(f"Skipping invalid entry for checksum {checksum[:16]}")
                        continue
                        
                    entry = CacheEntry(**entry_data)
                    self._cache[checksum] = entry
                    loaded_entries += 1
                except Exception as entry_error:
                    logger.warning(f"Skipping corrupted entry {checksum[:16]}: {entry_error}")
                    continue
            
            # Cargar métricas
            metrics_data = data.get('metrics', {})
            if metrics_data and isinstance(metrics_data, dict):
                try:
                    self._metrics = CacheMetrics(**metrics_data)
                except Exception:
                    logger.warning("Failed to load metrics, using defaults")
                    self._metrics = CacheMetrics()
            else:
                self._metrics = CacheMetrics()
            
            self._metrics.cache_size = len(self._cache)
            self._metrics.cache_loads += 1
            logger.info(f"Loaded cache from {file_path}: {loaded_entries} entries")
            return True
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error loading {file_path}: {e}")
        except Exception as e:
            logger.error(f"Failed to load cache from {file_path}: {str(e)}")
        
        return False
    
    def _save_cache(self) -> None:
        """Guardar caché a archivo JSON con backup automático"""
        start_time = time.time()
        
        try:
            # Crear backup del archivo existente si existe
            if Path(self.cache_path).exists():
                try:
                    import shutil
                    shutil.copy2(self.cache_path, self.backup_path)
                    logger.debug(f"Created backup: {self.backup_path}")
                except Exception as backup_error:
                    logger.warning(f"Failed to create backup: {backup_error}")
            
            # Preparar datos para serialización
            cache_data = {}
            serialization_errors = 0
            
            for checksum, entry in self._cache.items():
                try:
                    cache_data[checksum] = asdict(entry)
                except Exception as serialize_error:
                    logger.warning(f"Failed to serialize entry {checksum[:16]}: {serialize_error}")
                    serialization_errors += 1
                    continue
            
            if serialization_errors > 0:
                logger.warning(f"Skipped {serialization_errors} entries due to serialization errors")
            
            data = {
                'cache': cache_data,
                'metrics': asdict(self._metrics),
                'metadata': {
                    'version': '1.0',
                    'last_saved': datetime.now().isoformat(),
                    'max_cache_size': self.max_cache_size,
                    'max_age_days': self.max_age_days,
                    'serialization_errors': serialization_errors,
                    'entries_saved': len(cache_data)
                }
            }
            
            # Escribir archivo atómicamente
            with open(self.temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Verificar integridad del archivo temporal
            try:
                with open(self.temp_path, 'r', encoding='utf-8') as f:
                    json.load(f)  # Verificar que se puede leer
            except Exception as integrity_error:
                raise ValueError(f"Cache file integrity check failed: {integrity_error}")
            
            # Mover archivo temporal al definitivo
            if os.name == 'nt':  # Windows
                if Path(self.cache_path).exists():
                    os.remove(self.cache_path)
                os.rename(self.temp_path, self.cache_path)
            else:  # Unix/Linux
                os.rename(self.temp_path, self.cache_path)
            
            logger.debug(f"Saved cache to {self.cache_path}: {len(cache_data)} entries")
            
            # Actualizar métricas de save
            operation_time = time.time() - start_time
            self._metrics.cache_saves += 1
            self._metrics.total_save_time += operation_time
            
        except Exception as e:
            logger.error(f"Failed to save cache to {self.cache_path}: {str(e)}")
            
            # Actualizar métricas incluso en fallo
            operation_time = time.time() - start_time
            self._metrics.total_save_time += operation_time
            
            # Limpiar archivo temporal si existe
            try:
                if Path(self.temp_path).exists():
                    os.remove(self.temp_path)
            except Exception:
                pass
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Obtener información del estado del caché.
        
        Returns:
            Dict con información completa del caché
        """
        with self._lock:
            # Calcular uso de disco
            main_file_size = Path(self.cache_path).stat().st_size if Path(self.cache_path).exists() else 0
            backup_file_size = Path(self.backup_path).stat().st_size if Path(self.backup_path).exists() else 0
            total_disk_usage = main_file_size + backup_file_size
            
            # Calcular estadísticas de entradas
            entry_sizes = [len(entry.description) for entry in self._cache.values()]
            avg_entry_size = sum(entry_sizes) / len(entry_sizes) if entry_sizes else 0
            total_description_chars = sum(entry_sizes)
            
            return {
                'cache_path': self.cache_path,
                'backup_path': self.backup_path,
                'cache_size': len(self._cache),
                'max_cache_size': self.max_cache_size,
                'max_age_days': self.max_age_days,
                'auto_save': self.auto_save,
                'metrics': self._metrics.to_dict(),
                'oldest_entry': self._get_oldest_entry_date(),
                'newest_entry': self._get_newest_entry_date(),
                'file_exists': Path(self.cache_path).exists(),
                'backup_exists': Path(self.backup_path).exists(),
                'disk_usage': {
                    'main_file_bytes': main_file_size,
                    'backup_file_bytes': backup_file_size,
                    'total_bytes': total_disk_usage,
                    'main_file_mb': round(main_file_size / 1024 / 1024, 2),
                    'backup_file_mb': round(backup_file_size / 1024 / 1024, 2),
                    'total_mb': round(total_disk_usage / 1024 / 1024, 2)
                },
                'content_stats': {
                    'avg_description_length': round(avg_entry_size, 1),
                    'total_description_chars': total_description_chars,
                    'estimated_memory_usage_mb': round(total_description_chars / 1024 / 1024, 2)
                }
            }
    
    def _get_oldest_entry_date(self) -> Optional[str]:
        """Obtener fecha de la entrada más antigua"""
        if not self._cache:
            return None
        return min(entry.created_at for entry in self._cache.values())
    
    def _get_newest_entry_date(self) -> Optional[str]:
        """Obtener fecha de la entrada más reciente"""
        if not self._cache:
            return None
        return max(entry.created_at for entry in self._cache.values())
    
    def invalidate_entry(self, checksum: str, reason: str = "manual") -> bool:
        """
        Invalidar una entrada específica del caché.
        
        Args:
            checksum: Checksum SHA-256 de la entrada a invalidar
            reason: Razón de la invalidación para logging
            
        Returns:
            bool: True si se invalidó una entrada, False si no existía
        """
        with self._lock:
            if checksum in self._cache:
                del self._cache[checksum]
                self._metrics.cache_size = len(self._cache)
                
                logger.info(f"Invalidated cache entry {checksum[:16]}... (reason: {reason})")
                
                if self.auto_save:
                    self._save_cache()
                
                return True
            else:
                logger.debug(f"Attempted to invalidate non-existent entry: {checksum[:16]}...")
                return False
    
    def reset_metrics(self) -> None:
        """Reiniciar métricas de caché manteniendo estructura"""
        with self._lock:
            self._metrics.reset_metrics()
            logger.info("Cache metrics reset")
    
    def validate_cache_integrity(self, image_directory: Optional[str] = None) -> Dict[str, Any]:
        """
        Validar integridad del caché comparando con imágenes en disco.
        
        Args:
            image_directory: Directorio base de imágenes para validación
            
        Returns:
            Dict con resultados de validación
        """
        results = {
            'total_entries': len(self._cache),
            'valid_entries': 0,
            'invalid_entries': 0,
            'missing_files': 0,
            'checksum_mismatches': 0,
            'validation_errors': 0,
            'invalid_checksums': []
        }
        
        with self._lock:
            self._metrics.integrity_checks += 1
            
            for checksum, entry in list(self._cache.items()):
                try:
                    # Si hay path de imagen, validar contra archivo
                    if entry.image_path and image_directory:
                        # Construir path completo
                        if entry.image_path.startswith('/'):
                            full_path = entry.image_path
                        else:
                            full_path = os.path.join(image_directory, entry.image_path)
                        
                        # Verificar si archivo existe
                        if not os.path.exists(full_path):
                            results['missing_files'] += 1
                            results['invalid_entries'] += 1
                            results['invalid_checksums'].append({
                                'checksum': checksum[:16] + '...',
                                'reason': 'missing_file',
                                'path': full_path
                            })
                            continue
                        
                        # Verificar checksum
                        try:
                            current_checksum = self.compute_checksum(full_path)
                            if current_checksum != checksum:
                                results['checksum_mismatches'] += 1
                                results['invalid_entries'] += 1
                                results['invalid_checksums'].append({
                                    'checksum': checksum[:16] + '...',
                                    'reason': 'checksum_mismatch',
                                    'path': full_path,
                                    'expected': checksum[:16] + '...',
                                    'actual': current_checksum[:16] + '...'
                                })
                                continue
                        except Exception as checksum_error:
                            results['validation_errors'] += 1
                            results['invalid_entries'] += 1
                            results['invalid_checksums'].append({
                                'checksum': checksum[:16] + '...',
                                'reason': 'validation_error',
                                'error': str(checksum_error)
                            })
                            continue
                    
                    # Entrada válida
                    results['valid_entries'] += 1
                    
                except Exception as validation_error:
                    results['validation_errors'] += 1
                    results['invalid_entries'] += 1
                    logger.error(f"Validation error for checksum {checksum[:16]}...: {validation_error}")
        
        logger.info(f"Cache integrity validation completed: "
                   f"{results['valid_entries']} valid, {results['invalid_entries']} invalid")
        
        return results
    
    def cleanup_invalid_entries(self, validation_results: Optional[Dict[str, Any]] = None) -> int:
        """
        Limpiar entradas inválidas del caché.
        
        Args:
            validation_results: Resultados de validate_cache_integrity() (se ejecuta si no se proporciona)
            
        Returns:
            int: Número de entradas eliminadas
        """
        if validation_results is None:
            validation_results = self.validate_cache_integrity()
        
        removed_count = 0
        
        with self._lock:
            for invalid_entry in validation_results.get('invalid_checksums', []):
                # Extraer checksum completo (necesitamos buscar por prefijo)
                checksum_prefix = invalid_entry['checksum'].replace('...', '')
                
                for checksum in list(self._cache.keys()):
                    if checksum.startswith(checksum_prefix):
                        del self._cache[checksum]
                        removed_count += 1
                        logger.info(f"Removed invalid cache entry: {checksum[:16]}... "
                                  f"(reason: {invalid_entry['reason']})")
                        break
            
            self._metrics.cache_size = len(self._cache)
            
            if removed_count > 0 and self.auto_save:
                self._save_cache()
        
        logger.info(f"Cleanup completed: {removed_count} invalid entries removed")
        return removed_count
    
    def cleanup_expired_entries(self) -> int:
        """
        Limpiar entradas expiradas basadas en TTL.
        
        Returns:
            int: Número de entradas expiradas eliminadas
        """
        if self.max_age_days <= 0:
            return 0
        
        expired_count = 0
        cutoff_time = datetime.now() - timedelta(days=self.max_age_days)
        
        with self._lock:
            for checksum, entry in list(self._cache.items()):
                try:
                    created_time = datetime.fromisoformat(entry.created_at.replace('Z', '+00:00'))
                    if created_time < cutoff_time:
                        del self._cache[checksum]
                        expired_count += 1
                        logger.debug(f"Removed expired entry: {checksum[:16]}... "
                                   f"(age: {(datetime.now() - created_time).days} days)")
                except Exception as parse_error:
                    logger.warning(f"Failed to parse timestamp for entry {checksum[:16]}...: {parse_error}")
                    continue
            
            self._metrics.cache_size = len(self._cache)
            self._metrics.expired_entries += expired_count
            
            if expired_count > 0:
                self._metrics.last_cleanup = datetime.now().isoformat()
                logger.info(f"Cleaned up {expired_count} expired cache entries")
                
                if self.auto_save:
                    self._save_cache()
        
        return expired_count
    
    def cleanup_cache(self, force_full_cleanup: bool = False) -> Dict[str, int]:
        """
        Ejecutar limpieza completa del caché según políticas configuradas.
        
        Args:
            force_full_cleanup: Forzar limpieza completa independientemente de políticas
            
        Returns:
            Dict con contadores de limpieza por tipo
        """
        cleanup_stats = {
            'expired_entries': 0,
            'lru_entries': 0,
            'invalid_entries': 0,
            'total_removed': 0
        }
        
        with self._lock:
            initial_size = len(self._cache)
            
            logger.info(f"Starting cache cleanup (force={force_full_cleanup}). "
                       f"Current size: {initial_size} entries")
            
            # 1. Limpiar entradas expiradas
            if self.max_age_days > 0 or force_full_cleanup:
                cleanup_stats['expired_entries'] = self.cleanup_expired_entries()
            
            # 2. Limpiar por tamaño (LRU)
            if len(self._cache) > self.max_cache_size or force_full_cleanup:
                initial_lru_size = len(self._cache)
                self._cleanup_lru_entries()
                cleanup_stats['lru_entries'] = initial_lru_size - len(self._cache)
            
            # 3. Validar integridad si se fuerza
            if force_full_cleanup:
                validation_results = self.validate_cache_integrity()
                cleanup_stats['invalid_entries'] = self.cleanup_invalid_entries(validation_results)
            
            cleanup_stats['total_removed'] = initial_size - len(self._cache)
            
            # Actualizar métricas
            self._metrics.last_cleanup = datetime.now().isoformat()
            
            logger.info(f"Cache cleanup completed. Removed: {cleanup_stats['total_removed']} entries "
                       f"(expired: {cleanup_stats['expired_entries']}, "
                       f"lru: {cleanup_stats['lru_entries']}, "
                       f"invalid: {cleanup_stats['invalid_entries']})")
        
        return cleanup_stats
    
    def clear_cache(self) -> int:
        """
        Limpiar todo el caché.
        
        Returns:
            Número de entradas eliminadas
        """
        with self._lock:
            entries_count = len(self._cache)
            self._cache.clear()
            self._metrics = CacheMetrics()
            
            if self.auto_save:
                self._save_cache()
            
            logger.info(f"Cleared cache: {entries_count} entries removed")
            return entries_count
    
    def force_save(self) -> None:
        """Forzar guardado del caché a disco"""
        with self._lock:
            self._save_cache()
    
    def health_check(self) -> bool:
        """
        Verificar salud del sistema de caché.
        
        Returns:
            True si el caché está funcionando correctamente
        """
        try:
            # Verificar acceso de lectura/escritura al directorio de caché
            cache_dir = Path(self.cache_path).parent
            if not cache_dir.exists():
                return False
            
            # Test de escritura básico
            test_file = cache_dir / ".cache_health_check"
            test_file.write_text("health_check")
            test_file.unlink()
            
            # Verificar lock funcional
            with self._lock:
                pass
            
            logger.debug("Cache health check passed")
            return True
            
        except Exception as e:
            logger.error(f"Cache health check failed: {str(e)}")
            return False


class CachedImageDescriptor:
    """
    Wrapper para ImageDescriptor que integra DescriptionCache de manera transparente.
    
    Proporciona interfaz compatible con ImageDescriptor pero con capacidades de caché
    para evitar regenerar descripciones de imágenes idénticas.
    """
    
    def __init__(self, 
                 image_descriptor: Optional['ImageDescriptor'] = None,
                 cache: Optional[DescriptionCache] = None,
                 **kwargs):
        """
        Inicializar descriptor cacheado.
        
        Args:
            image_descriptor: Instancia de ImageDescriptor (se crea una nueva si no se proporciona)
            cache: Instancia de DescriptionCache (se crea una nueva si no se proporciona)
            **kwargs: Argumentos adicionales para ImageDescriptor si se crea nueva instancia
        """
        # Inicializar ImageDescriptor
        if image_descriptor is not None:
            self.descriptor = image_descriptor
        else:
            # Importar dinámicamente para evitar dependencias circulares
            try:
                from image_descriptor import ImageDescriptor
                self.descriptor = ImageDescriptor(**kwargs)
            except ImportError as e:
                raise ImportError(f"Could not import ImageDescriptor: {e}")
        
        # Inicializar caché
        self.cache = cache or create_description_cache()
        
        logger.info(f"CachedImageDescriptor initialized: "
                   f"cache_enabled={self.cache.enabled}, "
                   f"cache_size={len(self.cache._cache)}")
    
    def describe_image_robust(self, 
                            image_path: str,
                            image_metadata: Optional[Dict] = None,
                            image_type: Optional['ImageContentType'] = None,
                            custom_instructions: Optional[str] = None,
                            force_regenerate: bool = False) -> Optional[str]:
        """
        Generar descripción de imagen con caché robusto.
        
        Args:
            image_path: Ruta del archivo de imagen
            image_metadata: Metadata opcional de imagen
            image_type: Tipo específico de imagen
            custom_instructions: Instrucciones adicionales específicas
            force_regenerate: Forzar regeneración ignorando caché
            
        Returns:
            Descripción generada de la imagen o None si falla
        """
        start_time = time.time()
        
        try:
            # Calcular checksum de la imagen
            checksum = self.cache.compute_checksum(image_path)
            
            # Intentar obtener desde caché si no se fuerza regeneración
            if not force_regenerate:
                cached_description = self.cache.get_description(
                    checksum,
                    validate_image=image_path,
                    force_validation=False
                )
                
                if cached_description:
                    logger.debug(f"Cache hit for image {image_path}: {len(cached_description)} chars")
                    return cached_description
            
            # Generar nueva descripción usando ImageDescriptor
            logger.debug(f"Generating new description for image {image_path}")
            
            description = self.descriptor.describe_image_robust(
                image_path=image_path,
                image_metadata=image_metadata,
                image_type=image_type,
                custom_instructions=custom_instructions
            )
            
            if description:
                # Almacenar en caché
                generation_time = time.time() - start_time
                success = self.cache.put_description(
                    checksum=checksum,
                    description=description,
                    image_path=image_path,
                    image_metadata=image_metadata,
                    generation_time=generation_time,
                    force_override=force_regenerate,
                    validate_image=image_path
                )
                
                if success:
                    logger.debug(f"Cached description for {image_path}: {len(description)} chars")
                else:
                    logger.warning(f"Failed to cache description for {image_path}")
                
                return description
            else:
                logger.warning(f"Failed to generate description for {image_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error in describe_image_robust for {image_path}: {str(e)}")
            # Fallback a descriptor original sin caché
            try:
                return self.descriptor.describe_image_robust(
                    image_path=image_path,
                    image_metadata=image_metadata,
                    image_type=image_type,
                    custom_instructions=custom_instructions
                )
            except Exception as fallback_error:
                logger.error(f"Fallback descriptor also failed: {fallback_error}")
                return None
    
    def describe_image_from_data(self,
                               image_data: str,
                               image_metadata: Optional[Dict] = None,
                               image_type: Optional['ImageContentType'] = None,
                               custom_instructions: Optional[str] = None,
                               force_regenerate: bool = False) -> Optional[str]:
        """
        Generar descripción de imagen a partir de datos base64 con caché.
        
        Args:
            image_data: Imagen en formato base64 con data URI
            image_metadata: Metadata opcional de imagen
            image_type: Tipo específico de imagen
            custom_instructions: Instrucciones adicionales específicas
            force_regenerate: Forzar regeneración ignorando caché
            
        Returns:
            Descripción generada de la imagen o None si falla
        """
        start_time = time.time()
        
        try:
            # Calcular checksum de los datos de imagen
            checksum = self.cache.compute_checksum(image_data)
            
            # Intentar obtener desde caché si no se fuerza regeneración
            if not force_regenerate:
                cached_description = self.cache.get_description(
                    checksum,
                    validate_image=image_data,
                    force_validation=False
                )
                
                if cached_description:
                    logger.debug(f"Cache hit for base64 data: {len(cached_description)} chars")
                    return cached_description
            
            # Generar nueva descripción usando ImageDescriptor
            logger.debug("Generating new description for base64 image data")
            
            description = self.descriptor.describe_image_from_data(
                image_data=image_data,
                image_metadata=image_metadata,
                image_type=image_type,
                custom_instructions=custom_instructions
            )
            
            if description:
                # Almacenar en caché
                generation_time = time.time() - start_time
                success = self.cache.put_description(
                    checksum=checksum,
                    description=description,
                    image_metadata=image_metadata,
                    generation_time=generation_time,
                    force_override=force_regenerate,
                    validate_image=image_data
                )
                
                if success:
                    logger.debug(f"Cached description for base64 data: {len(description)} chars")
                else:
                    logger.warning("Failed to cache description for base64 data")
                
                return description
            else:
                logger.warning("Failed to generate description for base64 data")
                return None
                
        except Exception as e:
            logger.error(f"Error in describe_image_from_data: {str(e)}")
            # Fallback a descriptor original sin caché
            try:
                return self.descriptor.describe_image_from_data(
                    image_data=image_data,
                    image_metadata=image_metadata,
                    image_type=image_type,
                    custom_instructions=custom_instructions
                )
            except Exception as fallback_error:
                logger.error(f"Fallback descriptor also failed: {fallback_error}")
                return None
    
    def get_cache_metrics(self) -> Dict[str, Any]:
        """
        Obtener métricas combinadas de caché y descriptor.
        
        Returns:
            Dict con métricas completas
        """
        cache_info = self.cache.get_cache_info()
        descriptor_metrics = self.descriptor.get_metrics() if hasattr(self.descriptor, 'get_metrics') else {}
        
        return {
            'cache': cache_info,
            'descriptor': descriptor_metrics,
            'integration': {
                'cache_enabled': self.cache.enabled,
                'total_operations': cache_info['metrics']['total_operations'],
                'cache_efficiency': cache_info['metrics']['efficiency_score']
            }
        }
    
    def cleanup_cache(self, force_full_cleanup: bool = False) -> Dict[str, int]:
        """Ejecutar limpieza de caché"""
        return self.cache.cleanup_cache(force_full_cleanup)
    
    def health_check(self) -> bool:
        """Verificar salud del sistema completo"""
        descriptor_healthy = self.descriptor.health_check() if hasattr(self.descriptor, 'health_check') else True
        cache_healthy = self.cache.health_check()
        
        return descriptor_healthy and cache_healthy
    
    # Delegación de métodos para compatibilidad completa
    def __getattr__(self, name):
        """Delegar métodos no implementados al ImageDescriptor original"""
        return getattr(self.descriptor, name)


def create_description_cache(cache_path: Optional[str] = None,
                           max_cache_size: int = 10000,
                           max_age_days: int = 30) -> DescriptionCache:
    """
    Función de conveniencia para crear instancia de DescriptionCache.
    
    Args:
        cache_path: Ruta del archivo de caché JSON
        max_cache_size: Tamaño máximo del caché
        max_age_days: Edad máxima de entradas
        
    Returns:
        DescriptionCache: Instancia configurada
    """
    # Obtener configuración desde variables de entorno
    env_max_size = os.getenv('DESCRIPTION_CACHE_MAX_SIZE')
    if env_max_size:
        try:
            max_cache_size = int(env_max_size)
        except ValueError:
            logger.warning(f"Invalid DESCRIPTION_CACHE_MAX_SIZE: {env_max_size}, using default")
    
    env_max_age = os.getenv('DESCRIPTION_CACHE_MAX_AGE_DAYS')
    if env_max_age:
        try:
            max_age_days = int(env_max_age)
        except ValueError:
            logger.warning(f"Invalid DESCRIPTION_CACHE_MAX_AGE_DAYS: {env_max_age}, using default")
    
    return DescriptionCache(
        cache_path=cache_path,
        max_cache_size=max_cache_size,
        max_age_days=max_age_days,
        auto_save=True
    )


def create_cached_image_descriptor(**kwargs) -> CachedImageDescriptor:
    """
    Función de conveniencia para crear CachedImageDescriptor con configuración por defecto.
    
    Args:
        **kwargs: Argumentos para ImageDescriptor y DescriptionCache
        
    Returns:
        CachedImageDescriptor: Instancia configurada y lista para uso
    """
    return CachedImageDescriptor(**kwargs)