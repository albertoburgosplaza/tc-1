"""
Módulo para extraer imágenes de PDFs usando PyMuPDF y preprocesarlas para el sistema RAG multimodal.

Este módulo proporciona funcionalidades para:
- Extraer imágenes embebidas de PDFs usando PyMuPDF
- Validar formatos soportados (PNG, JPEG, WebP)  
- Preparar metadata de imágenes para indexación
"""

import os
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, NamedTuple
import hashlib
from io import BytesIO

import fitz  # PyMuPDF
from PIL import Image, ImageOps

# Configurar logging
logger = logging.getLogger(__name__)

class ImageExtractionMetrics:
    """
    Clase para registrar y almacenar métricas de extracción de imágenes
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reinicia todas las métricas"""
        self.pdfs_processed = 0
        self.total_images_found = 0
        self.total_images_saved = 0
        self.total_extraction_errors = 0
        self.total_processing_time = 0.0
        self.filter_stats = {
            'too_small': 0,
            'extreme_aspect_ratio': 0,
            'too_small_file': 0,
            'header_footer': 0,
            'low_complexity': 0,
            'duplicate_hash': 0,
            'total_filtered': 0,
            'total_kept': 0
        }
        self.extraction_times_by_pdf = []
        self.images_per_pdf = []
        self.errors_by_type = {}
    
    def record_pdf_processing(self, images_found: int, images_saved: int, processing_time: float, filter_stats: dict = None):
        """Registra métricas de procesamiento de un PDF"""
        self.pdfs_processed += 1
        self.total_images_found += images_found
        self.total_images_saved += images_saved
        self.total_processing_time += processing_time
        self.extraction_times_by_pdf.append(processing_time)
        self.images_per_pdf.append(images_found)
        
        if filter_stats:
            for key, value in filter_stats.items():
                if key in self.filter_stats:
                    self.filter_stats[key] += value
    
    def record_extraction_error(self, error_type: str):
        """Registra un error de extracción"""
        self.total_extraction_errors += 1
        self.errors_by_type[error_type] = self.errors_by_type.get(error_type, 0) + 1
    
    def get_metrics_summary(self) -> Dict:
        """Retorna resumen completo de métricas"""
        avg_processing_time = self.total_processing_time / self.pdfs_processed if self.pdfs_processed > 0 else 0.0
        avg_images_per_pdf = self.total_images_found / self.pdfs_processed if self.pdfs_processed > 0 else 0.0
        success_rate = ((self.pdfs_processed - self.total_extraction_errors) / self.pdfs_processed * 100) if self.pdfs_processed > 0 else 100.0
        
        return {
            'extraction_summary': {
                'pdfs_processed': self.pdfs_processed,
                'total_images_found': self.total_images_found,
                'total_images_saved': self.total_images_saved,
                'total_extraction_errors': self.total_extraction_errors,
                'success_rate_percent': round(success_rate, 2)
            },
            'performance_metrics': {
                'total_processing_time_seconds': round(self.total_processing_time, 2),
                'avg_processing_time_seconds': round(avg_processing_time, 2),
                'avg_images_per_pdf': round(avg_images_per_pdf, 2),
                'max_processing_time_seconds': round(max(self.extraction_times_by_pdf) if self.extraction_times_by_pdf else 0, 2),
                'min_processing_time_seconds': round(min(self.extraction_times_by_pdf) if self.extraction_times_by_pdf else 0, 2)
            },
            'filtering_metrics': {
                **self.filter_stats,
                'filter_efficiency_percent': round((self.filter_stats['total_filtered'] / (self.filter_stats['total_filtered'] + self.filter_stats['total_kept']) * 100) if (self.filter_stats['total_filtered'] + self.filter_stats['total_kept']) > 0 else 0, 2)
            },
            'error_breakdown': self.errors_by_type
        }

# Instancia global de métricas
image_extraction_metrics = ImageExtractionMetrics()

# Configuración de formatos soportados
SUPPORTED_FORMATS = {'PNG', 'JPEG', 'WEBP'}
MAX_IMAGE_DIMENSION = 1024  # Máximo tamaño para lado mayor
THUMBNAIL_SIZE = 256  # Tamaño máximo para thumbnails

# Configuración de filtros de imágenes
MIN_IMAGE_WIDTH = 50  # Mínimo ancho para considerar imagen significativa
MIN_IMAGE_HEIGHT = 50  # Mínimo alto para considerar imagen significativa
MAX_ASPECT_RATIO = 10.0  # Máximo aspect ratio (ancho/alto o alto/ancho)
MIN_FILE_SIZE = 500  # Mínimo tamaño de archivo en bytes
HEADER_FOOTER_THRESHOLD = 0.15  # 15% desde arriba/abajo se considera header/footer
MIN_VISUAL_COMPLEXITY_THRESHOLD = 100  # Mínima varianza de píxeles para complejidad visual

class ExtractedImage(NamedTuple):
    """Estructura de datos para imágenes extraídas"""
    image_data: bytes
    format: str
    width: int 
    height: int
    page_number: int
    image_index: int  # Índice de imagen en la página
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    hash: str
    
class ImageExtractionError(Exception):
    """Excepción personalizada para errores de extracción de imágenes"""
    pass

def _validate_image_format(image_ext: str) -> bool:
    """Valida si el formato de imagen es soportado"""
    return image_ext.upper() in SUPPORTED_FORMATS

def _calculate_image_hash(image_data: bytes) -> str:
    """Calcula hash SHA-256 del binario de imagen"""
    return hashlib.sha256(image_data).hexdigest()

def _normalize_image_orientation(image_data: bytes) -> bytes:
    """
    Normaliza la orientación de imagen usando datos EXIF si están disponibles
    """
    try:
        image = Image.open(BytesIO(image_data))
        
        # Aplicar rotación basada en EXIF usando ImageOps.exif_transpose
        # Esta función maneja automáticamente la orientación EXIF
        image = ImageOps.exif_transpose(image)
        
        # Convertir de vuelta a bytes
        output = BytesIO()
        # Mantener el formato original si es posible
        format_name = image.format or 'JPEG'
        image.save(output, format=format_name, optimize=True, quality=95)
        return output.getvalue()
        
    except Exception as e:
        logger.warning(f"Error al normalizar orientación de imagen: {e}")
        return image_data  # Retornar imagen original si falla

def _resize_image(image_data: bytes, max_dimension: int = MAX_IMAGE_DIMENSION) -> bytes:
    """
    Redimensiona imagen manteniendo aspect ratio, con lado mayor <= max_dimension
    """
    try:
        image = Image.open(BytesIO(image_data))
        
        # Verificar si necesita redimensionamiento
        if max(image.size) <= max_dimension:
            return image_data
            
        # Calcular nuevas dimensiones manteniendo aspect ratio
        width, height = image.size
        if width > height:
            new_width = max_dimension
            new_height = int((height * max_dimension) / width)
        else:
            new_height = max_dimension
            new_width = int((width * max_dimension) / height)
            
        # Redimensionar con alta calidad
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Guardar en bytes con calidad optimizada
        output = BytesIO()
        format_name = image.format or 'JPEG'
        image.save(output, format=format_name, optimize=True, quality=95)
        return output.getvalue()
        
    except Exception as e:
        logger.error(f"Error al redimensionar imagen: {e}")
        raise ImageExtractionError(f"Error al redimensionar imagen: {e}")

def _calculate_visual_complexity(image_data: bytes) -> float:
    """
    Calcula la complejidad visual de una imagen basada en la varianza de píxeles
    
    Args:
        image_data: Datos binarios de la imagen
        
    Returns:
        Valor de complejidad (varianza de píxeles en escala de grises)
    """
    try:
        image = Image.open(BytesIO(image_data))
        
        # Convertir a escala de grises para análisis de complejidad
        if image.mode != 'L':
            image = image.convert('L')
        
        # Calcular varianza de píxeles como medida de complejidad
        import numpy as np
        pixels = np.array(image)
        variance = np.var(pixels)
        
        return float(variance)
        
    except Exception as e:
        logger.debug(f"Error calculando complejidad visual: {e}")
        return MIN_VISUAL_COMPLEXITY_THRESHOLD + 1  # Asumir complejo si hay error

def _is_in_header_footer_region(bbox: Tuple[float, float, float, float], 
                                page_height: float) -> bool:
    """
    Determina si una imagen está en región de header o footer
    
    Args:
        bbox: Bounding box de la imagen (x0, y0, x1, y1)
        page_height: Altura total de la página
        
    Returns:
        True si la imagen está en header o footer
    """
    _, y0, _, y1 = bbox
    
    # Calcular posición vertical relativa
    header_threshold = page_height * HEADER_FOOTER_THRESHOLD
    footer_threshold = page_height * (1 - HEADER_FOOTER_THRESHOLD)
    
    # Verificar si está en header (parte superior)
    if y1 <= header_threshold:
        return True
    
    # Verificar si está en footer (parte inferior)  
    if y0 >= footer_threshold:
        return True
    
    return False

def _filter_meaningful_images(extracted_images: List[ExtractedImage], 
                             page_height: float) -> Tuple[List[ExtractedImage], Dict[str, int]]:
    """
    Filtra imágenes significativas eliminando elementos decorativos
    
    Args:
        extracted_images: Lista de imágenes extraídas
        page_height: Altura de la página para detectar header/footer
        
    Returns:
        Tupla con (imágenes_filtradas, estadísticas_filtros)
    """
    meaningful_images = []
    filter_stats = {
        'too_small': 0,
        'extreme_aspect_ratio': 0,
        'too_small_file': 0,
        'header_footer': 0,
        'low_complexity': 0,
        'duplicate_hash': 0,
        'total_filtered': 0,
        'total_kept': 0
    }
    
    # Tracking para duplicados por hash
    seen_hashes = set()
    
    for img in extracted_images:
        filter_reason = None
        
        # Filtro 1: Tamaño mínimo
        if img.width < MIN_IMAGE_WIDTH or img.height < MIN_IMAGE_HEIGHT:
            filter_reason = 'too_small'
        
        # Filtro 2: Aspect ratio extremo
        elif img.width > 0 and img.height > 0:
            aspect_ratio = max(img.width / img.height, img.height / img.width)
            if aspect_ratio > MAX_ASPECT_RATIO:
                filter_reason = 'extreme_aspect_ratio'
        
        # Filtro 3: Tamaño de archivo muy pequeño
        elif len(img.image_data) < MIN_FILE_SIZE:
            filter_reason = 'too_small_file'
        
        # Filtro 4: Header/Footer
        elif _is_in_header_footer_region(img.bbox, page_height):
            filter_reason = 'header_footer'
        
        # Filtro 5: Baja complejidad visual
        elif _calculate_visual_complexity(img.image_data) < MIN_VISUAL_COMPLEXITY_THRESHOLD:
            filter_reason = 'low_complexity'
        
        # Filtro 6: Duplicados por hash
        elif img.hash in seen_hashes:
            filter_reason = 'duplicate_hash'
        
        # Aplicar filtro o mantener imagen
        if filter_reason:
            filter_stats[filter_reason] += 1
            filter_stats['total_filtered'] += 1
            logger.debug(f"Imagen filtrada ({filter_reason}): "
                        f"página {img.page_number}, {img.width}x{img.height}, "
                        f"hash: {img.hash[:8]}...")
        else:
            meaningful_images.append(img)
            seen_hashes.add(img.hash)
            filter_stats['total_kept'] += 1
    
    return meaningful_images, filter_stats

def extract_images_from_pdf(pdf_path: str, enable_filtering: bool = True) -> Tuple[List[ExtractedImage], Dict[str, int]]:
    """
    Extrae todas las imágenes de un PDF usando PyMuPDF con filtrado opcional
    
    Args:
        pdf_path: Ruta al archivo PDF
        enable_filtering: Si habilitar filtrado de imágenes no significativas
        
    Returns:
        Tupla con (lista de ExtractedImage, estadísticas de filtrado)
        
    Raises:
        ImageExtractionError: Si hay errores durante la extracción
    """
    if not os.path.exists(pdf_path):
        raise ImageExtractionError(f"Archivo PDF no encontrado: {pdf_path}")
    
    extracted_images = []
    total_page_height = 0
    filter_stats = {
        'too_small': 0,
        'extreme_aspect_ratio': 0,
        'too_small_file': 0,
        'header_footer': 0,
        'low_complexity': 0,
        'duplicate_hash': 0,
        'total_filtered': 0,
        'total_kept': 0
    }
    
    try:
        # Abrir PDF con PyMuPDF
        pdf_document = fitz.open(pdf_path)
        logger.info(f"Extrayendo imágenes de PDF: {Path(pdf_path).name} ({pdf_document.page_count} páginas)")
        
        # Calcular altura promedio de páginas para filtrado header/footer
        for page_number in range(pdf_document.page_count):
            page = pdf_document.load_page(page_number)
            total_page_height += page.rect.height
        avg_page_height = total_page_height / pdf_document.page_count if pdf_document.page_count > 0 else 792
        
        # Iterar sobre todas las páginas
        for page_number in range(pdf_document.page_count):
            page = pdf_document.load_page(page_number)
            
            # Obtener lista de imágenes en la página
            image_list = page.get_images(full=True)
            
            if not image_list:
                continue
                
            logger.debug(f"Página {page_number + 1}: encontradas {len(image_list)} imágenes")
            
            # Extraer cada imagen
            for image_index, img in enumerate(image_list):
                try:
                    # Obtener referencia de imagen
                    xref = img[0]  # xref es el primer elemento
                    
                    # Extraer imagen como diccionario
                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    width = base_image["width"] 
                    height = base_image["height"]
                    
                    # Validar formato soportado
                    if not _validate_image_format(image_ext):
                        logger.debug(f"Formato no soportado: {image_ext} en página {page_number + 1}")
                        continue
                    
                    # Obtener bounding box de la imagen en la página
                    # img contiene: [xref, smask, width, height, bpc, colorspace, alt_colorspace, name, filter, bbox]
                    bbox = img[8] if len(img) > 8 else (0, 0, page.rect.width, page.rect.height)
                    
                    # Normalizar orientación 
                    normalized_image = _normalize_image_orientation(image_bytes)
                    
                    # Redimensionar si es necesario
                    resized_image = _resize_image(normalized_image)
                    
                    # Calcular hash del binario preprocesado
                    image_hash = _calculate_image_hash(resized_image)
                    
                    # Crear objeto ExtractedImage
                    extracted_image = ExtractedImage(
                        image_data=resized_image,
                        format=image_ext.upper(),
                        width=width,
                        height=height,
                        page_number=page_number + 1,  # 1-indexed
                        image_index=image_index,
                        bbox=bbox,
                        hash=image_hash
                    )
                    
                    extracted_images.append(extracted_image)
                    logger.debug(f"Imagen extraída: página {page_number + 1}, índice {image_index}, "
                               f"{width}x{height}, {image_ext.upper()}, hash: {image_hash[:8]}...")
                    
                except Exception as e:
                    logger.warning(f"Error extrayendo imagen {image_index} de página {page_number + 1}: {e}")
                    continue
        
        pdf_document.close()
        
        # Aplicar filtrado si está habilitado
        if enable_filtering and extracted_images:
            logger.info(f"Aplicando filtros a {len(extracted_images)} imágenes extraídas...")
            meaningful_images, filter_results = _filter_meaningful_images(extracted_images, avg_page_height)
            
            # Actualizar filter_stats con resultados del filtrado
            filter_stats.update(filter_results)
            
            # Log de estadísticas de filtrado
            if filter_stats['total_filtered'] > 0:
                logger.info(f"Filtrado completado: {filter_stats['total_kept']} imágenes mantenidas, "
                           f"{filter_stats['total_filtered']} filtradas")
                for reason, count in filter_stats.items():
                    if count > 0 and reason not in ['total_filtered', 'total_kept']:
                        logger.debug(f"  - {reason}: {count} imágenes")
            else:
                logger.info("Todas las imágenes pasaron el filtrado")
            
            extracted_images = meaningful_images
        else:
            # Si no hay filtrado, todas las imágenes se mantienen
            filter_stats['total_kept'] = len(extracted_images)
        
        logger.info(f"Extracción completada: {len(extracted_images)} imágenes significativas de {Path(pdf_path).name}")
        
    except Exception as e:
        logger.error(f"Error abriendo PDF {pdf_path}: {e}")
        raise ImageExtractionError(f"Error procesando PDF: {e}")
    
    return extracted_images, filter_stats

def generate_thumbnail(image_data: bytes, max_size: int = THUMBNAIL_SIZE) -> bytes:
    """
    Genera thumbnail JPEG de una imagen
    
    Args:
        image_data: Datos binarios de imagen original
        max_size: Tamaño máximo para lado mayor del thumbnail
        
    Returns:
        Datos binarios del thumbnail en formato JPEG
        
    Raises:
        ImageExtractionError: Si hay error generando thumbnail
    """
    try:
        image = Image.open(BytesIO(image_data))
        
        # Convertir a RGB si es necesario (para garantizar JPEG)
        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')
        
        # Calcular nuevas dimensiones para thumbnail
        width, height = image.size
        if max(width, height) <= max_size:
            # Ya es lo suficientemente pequeña
            new_width, new_height = width, height
        else:
            if width > height:
                new_width = max_size
                new_height = int((height * max_size) / width)
            else:
                new_height = max_size
                new_width = int((width * max_size) / height)
        
        # Crear thumbnail
        thumbnail = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Guardar como JPEG optimizado
        output = BytesIO()
        thumbnail.save(output, format='JPEG', optimize=True, quality=85)
        return output.getvalue()
        
    except Exception as e:
        logger.error(f"Error generando thumbnail: {e}")
        raise ImageExtractionError(f"Error generando thumbnail: {e}")

def save_processed_image(extracted_image: ExtractedImage, doc_id: str, 
                        base_images_dir: str = "/var/data/rag/images") -> Dict[str, str]:
    """
    Guarda imagen procesada y thumbnail en estructura de directorios organizada
    
    Args:
        extracted_image: Objeto ExtractedImage
        doc_id: ID del documento fuente
        base_images_dir: Directorio base para almacenar imágenes
        
    Returns:
        Diccionario con rutas de archivos guardados
        
    Raises:
        ImageExtractionError: Si hay error guardando archivos
    """
    try:
        # Crear estructura de directorios
        # /var/data/rag/images/{doc_id}/p{page_number}/
        page_dir = Path(base_images_dir) / doc_id / f"p{extracted_image.page_number}"
        page_dir.mkdir(parents=True, exist_ok=True)
        
        # Crear subcarpeta thumbs
        thumbs_dir = page_dir / "thumbs"
        thumbs_dir.mkdir(exist_ok=True)
        
        # Determinar extensión basada en formato
        ext_map = {'PNG': 'png', 'JPEG': 'jpg', 'WEBP': 'webp'}
        ext = ext_map.get(extracted_image.format, 'jpg')
        
        # Guardar imagen procesada
        image_filename = f"{extracted_image.hash}.{ext}"
        image_path = page_dir / image_filename
        
        with open(image_path, 'wb') as f:
            f.write(extracted_image.image_data)
            
        logger.debug(f"Imagen guardada: {image_path}")
        
        # Generar y guardar thumbnail
        thumbnail_data = generate_thumbnail(extracted_image.image_data)
        thumbnail_filename = f"{extracted_image.hash}.jpg"  # Thumbnails siempre JPEG
        thumbnail_path = thumbs_dir / thumbnail_filename
        
        with open(thumbnail_path, 'wb') as f:
            f.write(thumbnail_data)
            
        logger.debug(f"Thumbnail guardado: {thumbnail_path}")
        
        return {
            'image_path': str(image_path),
            'thumbnail_path': str(thumbnail_path),
            'image_uri': f"local://images/{doc_id}/p{extracted_image.page_number}/{image_filename}",
            'thumbnail_uri': f"local://images/{doc_id}/p{extracted_image.page_number}/thumbs/{thumbnail_filename}"
        }
        
    except PermissionError as e:
        logger.error(f"Error de permisos guardando imagen: {e}")
        raise ImageExtractionError(f"Sin permisos para escribir en {base_images_dir}: {e}")
    except OSError as e:
        logger.error(f"Error de filesystem guardando imagen: {e}")
        raise ImageExtractionError(f"Error de filesystem: {e}")
    except Exception as e:
        logger.error(f"Error inesperado guardando imagen: {e}")
        raise ImageExtractionError(f"Error guardando imagen: {e}")

def save_all_images(extracted_images: List[ExtractedImage], doc_id: str,
                   base_images_dir: str = "/var/data/rag/images") -> List[Dict]:
    """
    Guarda todas las imágenes extraídas de un documento
    
    Args:
        extracted_images: Lista de ExtractedImage
        doc_id: ID del documento fuente
        base_images_dir: Directorio base para almacenar imágenes
        
    Returns:
        Lista de diccionarios con información de archivos guardados
    """
    saved_images = []
    
    for img in extracted_images:
        try:
            # Guardar imagen individual
            paths = save_processed_image(img, doc_id, base_images_dir)
            
            # Crear metadata completa
            image_info = {
                **get_image_metadata(img, doc_id),
                **paths
            }
            
            saved_images.append(image_info)
            
        except ImageExtractionError as e:
            logger.warning(f"Error guardando imagen {img.hash[:8]}: {e}")
            continue
    
    logger.info(f"Guardadas {len(saved_images)} imágenes para documento {doc_id}")
    return saved_images

def cleanup_document_images(doc_id: str, base_images_dir: str = "/var/data/rag/images") -> bool:
    """
    Limpia todas las imágenes de un documento (para re-ingesta)
    
    Args:
        doc_id: ID del documento
        base_images_dir: Directorio base de imágenes
        
    Returns:
        True si la limpieza fue exitosa
    """
    try:
        doc_dir = Path(base_images_dir) / doc_id
        
        if not doc_dir.exists():
            logger.debug(f"No hay imágenes que limpiar para documento {doc_id}")
            return True
        
        # Eliminar directorio completo del documento
        import shutil
        shutil.rmtree(doc_dir)
        
        logger.info(f"Limpiadas imágenes del documento {doc_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error limpiando imágenes del documento {doc_id}: {e}")
        return False

def get_image_metadata(extracted_image: ExtractedImage, doc_id: str) -> Dict:
    """
    Genera diccionario de metadata para una imagen extraída
    
    Args:
        extracted_image: Objeto ExtractedImage
        doc_id: ID del documento fuente
        
    Returns:
        Diccionario con metadata de la imagen
    """
    return {
        'doc_id': doc_id,
        'page_number': extracted_image.page_number,
        'image_index': extracted_image.image_index,
        'format': extracted_image.format,
        'width': extracted_image.width,
        'height': extracted_image.height,
        'hash': extracted_image.hash,
        'bbox': {
            'x0': extracted_image.bbox[0],
            'y0': extracted_image.bbox[1], 
            'x1': extracted_image.bbox[2],
            'y1': extracted_image.bbox[3]
        },
        'processed_size': len(extracted_image.image_data)
    }

class ImageExtractor:
    """
    Clase principal para extraer, procesar y almacenar imágenes de PDFs
    
    Coordina todo el flujo de extracción multimodal:
    1. Extracción de imágenes del PDF
    2. Preprocesamiento (orientación, redimensionamiento)  
    3. Generación de thumbnails
    4. Almacenamiento en estructura organizada
    5. Generación de metadata completa
    """
    
    def __init__(self, base_images_dir: str = "/var/data/rag/images", 
                 max_pdf_size_mb: int = 100, enable_filtering: bool = True):
        """
        Inicializa el extractor de imágenes
        
        Args:
            base_images_dir: Directorio base para almacenar imágenes
            max_pdf_size_mb: Tamaño máximo de PDF en MB
            enable_filtering: Habilitar filtrado de imágenes no significativas
        """
        self.base_images_dir = Path(base_images_dir)
        self.max_pdf_size_bytes = max_pdf_size_mb * 1024 * 1024
        self.enable_filtering = enable_filtering
        
        logger.info(f"ImageExtractor inicializado: dir={base_images_dir}, "
                   f"max_size={max_pdf_size_mb}MB, filtrado={'habilitado' if enable_filtering else 'deshabilitado'}")
    
    def validate_pdf_file(self, pdf_path: str) -> tuple[bool, str]:
        """
        Valida que el archivo PDF sea procesable
        
        Args:
            pdf_path: Ruta al archivo PDF
            
        Returns:
            Tupla (es_válido, mensaje_error)
        """
        path = Path(pdf_path)
        
        # Verificar existencia
        if not path.exists():
            return False, f"Archivo no encontrado: {pdf_path}"
        
        # Verificar tamaño
        size_bytes = path.stat().st_size
        if size_bytes == 0:
            return False, f"Archivo vacío: {pdf_path}"
        
        if size_bytes > self.max_pdf_size_bytes:
            size_mb = size_bytes / (1024 * 1024)
            max_mb = self.max_pdf_size_bytes / (1024 * 1024)
            return False, f"Archivo demasiado grande: {size_mb:.1f}MB > {max_mb}MB"
        
        # Verificar extensión
        if path.suffix.lower() != '.pdf':
            return False, f"No es un archivo PDF: {path.suffix}"
        
        return True, "PDF válido"
    
    def ensure_storage_directory(self):
        """
        Asegura que el directorio de almacenamiento existe con permisos correctos
        
        Raises:
            ImageExtractionError: Si no puede crear o acceder al directorio
        """
        try:
            self.base_images_dir.mkdir(parents=True, exist_ok=True)
            
            # Verificar permisos de escritura
            test_file = self.base_images_dir / ".write_test"
            test_file.write_text("test")
            test_file.unlink()
            
            logger.debug(f"Directorio de almacenamiento válido: {self.base_images_dir}")
            
        except PermissionError as e:
            raise ImageExtractionError(f"Sin permisos de escritura en {self.base_images_dir}: {e}")
        except Exception as e:
            raise ImageExtractionError(f"Error accediendo a directorio de almacenamiento: {e}")
    
    def process_pdf(self, pdf_path: str, doc_id: Optional[str] = None,
                   save_images: bool = True) -> Dict:
        """
        Procesa un PDF completo extrayendo y almacenando todas sus imágenes
        
        Args:
            pdf_path: Ruta al archivo PDF
            doc_id: ID del documento (usa nombre del archivo si no se proporciona)
            save_images: Si guardar imágenes en disco (True por defecto)
            
        Returns:
            Diccionario con resultados del procesamiento
            
        Raises:
            ImageExtractionError: Si hay errores durante el procesamiento
        """
        start_time = time.time()
        
        # Validar archivo PDF
        is_valid, error_msg = self.validate_pdf_file(pdf_path)
        if not is_valid:
            raise ImageExtractionError(f"PDF inválido: {error_msg}")
        
        # Determinar doc_id
        if doc_id is None:
            doc_id = Path(pdf_path).stem
        
        logger.info(f"Iniciando procesamiento multimodal de {Path(pdf_path).name} (doc_id: {doc_id})")
        
        if save_images:
            # Asegurar directorio de almacenamiento
            self.ensure_storage_directory()
            
            # Limpiar imágenes anteriores del documento
            cleanup_document_images(doc_id, str(self.base_images_dir))
        
        try:
            # Extraer imágenes del PDF con configuración de filtrado
            extracted_images, filter_stats = extract_images_from_pdf(pdf_path, enable_filtering=self.enable_filtering)
            
            if not extracted_images:
                processing_time = time.time() - start_time
                
                # Registrar métricas incluso cuando no hay imágenes
                image_extraction_metrics.record_pdf_processing(
                    images_found=0,
                    images_saved=0,
                    processing_time=processing_time,
                    filter_stats=filter_stats
                )
                
                logger.info(f"No se encontraron imágenes en {Path(pdf_path).name}")
                return {
                    'success': True,
                    'doc_id': doc_id,
                    'images_found': 0,
                    'images_saved': 0,
                    'processing_time': processing_time,
                    'saved_images': []
                }
            
            # Guardar imágenes si está habilitado
            saved_images = []
            if save_images:
                saved_images = save_all_images(extracted_images, doc_id, str(self.base_images_dir))
            else:
                # Generar solo metadata sin guardar
                for img in extracted_images:
                    metadata = get_image_metadata(img, doc_id)
                    saved_images.append(metadata)
            
            processing_time = time.time() - start_time
            
            # Registrar métricas de procesamiento exitoso
            image_extraction_metrics.record_pdf_processing(
                images_found=len(extracted_images),
                images_saved=len(saved_images),
                processing_time=processing_time,
                filter_stats=filter_stats
            )
            
            logger.info(f"Procesamiento multimodal completado para {doc_id}: "
                       f"{len(extracted_images)} imágenes encontradas, "
                       f"{len(saved_images)} procesadas en {processing_time:.2f}s")
            
            return {
                'success': True,
                'doc_id': doc_id,
                'images_found': len(extracted_images),
                'images_saved': len(saved_images),
                'processing_time': processing_time,
                'saved_images': saved_images
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_type = type(e).__name__
            
            # Registrar error de procesamiento
            image_extraction_metrics.record_extraction_error(error_type)
            
            logger.error(f"Error procesando PDF {pdf_path}: {e}")
            raise ImageExtractionError(f"Error en procesamiento multimodal: {e}")
    
    def get_document_images(self, doc_id: str) -> List[Dict]:
        """
        Obtiene información de todas las imágenes almacenadas de un documento
        
        Args:
            doc_id: ID del documento
            
        Returns:
            Lista de diccionarios con información de imágenes
        """
        doc_dir = self.base_images_dir / doc_id
        
        if not doc_dir.exists():
            return []
        
        images_info = []
        
        # Buscar todas las páginas
        for page_dir in doc_dir.glob("p*"):
            if not page_dir.is_dir():
                continue
                
            page_number = int(page_dir.name[1:])  # Extraer número después de 'p'
            
            # Buscar imágenes en la página
            for image_file in page_dir.glob("*"):
                if image_file.is_file() and image_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']:
                    # Construir información de imagen
                    hash_name = image_file.stem
                    
                    # Buscar thumbnail correspondiente
                    thumbnail_path = page_dir / "thumbs" / f"{hash_name}.jpg"
                    
                    image_info = {
                        'doc_id': doc_id,
                        'page_number': page_number,
                        'hash': hash_name,
                        'format': image_file.suffix[1:].upper(),
                        'image_path': str(image_file),
                        'thumbnail_path': str(thumbnail_path) if thumbnail_path.exists() else None,
                        'image_uri': f"local://images/{doc_id}/p{page_number}/{image_file.name}",
                        'thumbnail_uri': f"local://images/{doc_id}/p{page_number}/thumbs/{hash_name}.jpg" if thumbnail_path.exists() else None
                    }
                    
                    images_info.append(image_info)
        
        return sorted(images_info, key=lambda x: (x['page_number'], x['hash']))
    
    def cleanup_document(self, doc_id: str) -> bool:
        """
        Limpia todas las imágenes de un documento
        
        Args:
            doc_id: ID del documento a limpiar
            
        Returns:
            True si la limpieza fue exitosa
        """
        return cleanup_document_images(doc_id, str(self.base_images_dir))
    
    def get_extraction_metrics(self) -> Dict:
        """
        Retorna métricas actuales de extracción de imágenes
        
        Returns:
            Diccionario con métricas de extracción
        """
        return image_extraction_metrics.get_metrics_summary()