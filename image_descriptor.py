"""
Módulo para generar descripciones automáticas de imágenes usando Google Gemini 2.5 Flash
"""

import os
import logging
import time
import base64
import uuid
import random
from typing import Optional, Dict, Any, Union, Tuple, List
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from enum import Enum

# Configurar logging estructurado
logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Tipos de errores para clasificación y manejo específico"""
    TRANSIENT_API_ERROR = "transient_api"        # Errores temporales de API (500, 502, etc.)
    RATE_LIMIT_ERROR = "rate_limit"              # Límites de velocidad de API
    AUTHENTICATION_ERROR = "auth"                # Errores de autenticación (permanente)
    INVALID_INPUT_ERROR = "invalid_input"        # Datos de entrada inválidos (permanente)
    FILE_NOT_FOUND_ERROR = "file_not_found"     # Archivo no encontrado (permanente)
    NETWORK_ERROR = "network"                    # Errores de conexión de red (transitorio)
    TIMEOUT_ERROR = "timeout"                    # Timeouts (transitorio)
    QUOTA_EXCEEDED_ERROR = "quota"               # Cuota excedida (temporal)
    UNKNOWN_ERROR = "unknown"                    # Errores desconocidos


class DescriptionMetrics:
    """Métricas de generación de descripciones"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reiniciar todas las métricas"""
        self.total_descriptions = 0
        self.successful_descriptions = 0
        self.failed_descriptions = 0
        self.total_processing_time = 0.0
        self.retry_attempts = 0
        self.errors_by_type = {}
        self.processing_times = []
        self.description_lengths = []
    
    def record_success(self, processing_time: float, description_length: int):
        """Registrar descripción exitosa"""
        self.total_descriptions += 1
        self.successful_descriptions += 1
        self.total_processing_time += processing_time
        self.processing_times.append(processing_time)
        self.description_lengths.append(description_length)
    
    def record_failure(self, error_type: ErrorType, processing_time: float):
        """Registrar descripción fallida"""
        self.total_descriptions += 1
        self.failed_descriptions += 1
        self.total_processing_time += processing_time
        self.errors_by_type[error_type.value] = self.errors_by_type.get(error_type.value, 0) + 1
    
    def record_retry(self):
        """Registrar intento de reintento"""
        self.retry_attempts += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Obtener resumen de métricas"""
        success_rate = (self.successful_descriptions / self.total_descriptions * 100) if self.total_descriptions > 0 else 0
        avg_processing_time = (self.total_processing_time / self.total_descriptions) if self.total_descriptions > 0 else 0
        avg_description_length = (sum(self.description_lengths) / len(self.description_lengths)) if self.description_lengths else 0
        
        return {
            'total_descriptions': self.total_descriptions,
            'successful_descriptions': self.successful_descriptions,
            'failed_descriptions': self.failed_descriptions,
            'success_rate_percent': round(success_rate, 2),
            'retry_attempts': self.retry_attempts,
            'avg_processing_time_seconds': round(avg_processing_time, 2),
            'avg_description_length': round(avg_description_length, 1),
            'errors_by_type': self.errors_by_type
        }


class RetryHandler:
    """Manejador de reintentos con backoff exponencial"""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
        """
        Inicializar manejador de reintentos
        
        Args:
            max_retries: Número máximo de reintentos
            base_delay: Delay base en segundos
            max_delay: Delay máximo en segundos
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
    
    def classify_error(self, error: Exception) -> ErrorType:
        """
        Clasificar error para determinar estrategia de reintento
        
        Args:
            error: Excepción a clasificar
            
        Returns:
            Tipo de error clasificado
        """
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # Errores de autenticación (permanentes)
        if 'auth' in error_str or 'api key' in error_str or 'unauthorized' in error_str:
            return ErrorType.AUTHENTICATION_ERROR
        
        # Errores de límite de velocidad
        if 'rate limit' in error_str or 'too many requests' in error_str or '429' in error_str:
            return ErrorType.RATE_LIMIT_ERROR
        
        # Errores de cuota
        if 'quota' in error_str or 'limit exceeded' in error_str:
            return ErrorType.QUOTA_EXCEEDED_ERROR
        
        # Errores de timeout
        if 'timeout' in error_str or 'timed out' in error_str:
            return ErrorType.TIMEOUT_ERROR
        
        # Errores de red (temporales)
        if 'connection' in error_str or 'network' in error_str or 'dns' in error_str:
            return ErrorType.NETWORK_ERROR
        
        # Errores de archivo
        if 'filenotfound' in error_type or 'no such file' in error_str:
            return ErrorType.FILE_NOT_FOUND_ERROR
        
        # Errores de entrada inválida
        if 'valueerror' in error_type or 'invalid' in error_str or 'bad request' in error_str:
            return ErrorType.INVALID_INPUT_ERROR
        
        # Errores transitorios de API
        if any(code in error_str for code in ['500', '502', '503', '504']):
            return ErrorType.TRANSIENT_API_ERROR
        
        return ErrorType.UNKNOWN_ERROR
    
    def should_retry(self, error_type: ErrorType, attempt: int) -> bool:
        """
        Determinar si se debe reintentar basado en el tipo de error
        
        Args:
            error_type: Tipo de error clasificado
            attempt: Número de intento actual
            
        Returns:
            True si se debe reintentar
        """
        if attempt >= self.max_retries:
            return False
        
        # Errores que NO se deben reintentar (permanentes)
        non_retryable = {
            ErrorType.AUTHENTICATION_ERROR,
            ErrorType.INVALID_INPUT_ERROR,
            ErrorType.FILE_NOT_FOUND_ERROR
        }
        
        return error_type not in non_retryable
    
    def calculate_delay(self, attempt: int, error_type: ErrorType) -> float:
        """
        Calcular delay para el siguiente intento
        
        Args:
            attempt: Número de intento actual
            error_type: Tipo de error
            
        Returns:
            Delay en segundos
        """
        # Delay especial para rate limiting
        if error_type == ErrorType.RATE_LIMIT_ERROR:
            return min(self.base_delay * (4 ** attempt), self.max_delay)
        
        # Delay especial para cuota excedida
        if error_type == ErrorType.QUOTA_EXCEEDED_ERROR:
            return min(self.base_delay * (8 ** attempt), self.max_delay)
        
        # Backoff exponencial estándar con jitter
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        jitter = delay * 0.1 * random.random()  # Añadir jitter del 10%
        
        return delay + jitter


class ImageDescriptor:
    """
    Clase para generar descripciones automáticas de imágenes usando Google Gemini 2.5 Flash
    """
    
    # Constantes de configuración
    DEFAULT_MODEL = "gemini-2.5-flash"
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
    DEFAULT_TIMEOUT = 30  # segundos
    API_TIMEOUT = int(os.getenv("GEMINI_API_TIMEOUT", str(DEFAULT_TIMEOUT)))
    MAX_TOKENS = 512  # Límite conservador para descripciones concisas
    TEMPERATURE = 0.1  # Temperatura baja para descripciones consistentes
    
    # Límites de validación
    MAX_IMAGE_SIZE_MB = 20  # Límite de Gemini API
    SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
    MAX_DESCRIPTION_LENGTH = CHUNK_SIZE - 200  # Margen para metadata
    
    def __init__(self, 
                 model_name: Optional[str] = None,
                 google_api_key: Optional[str] = None,
                 timeout: Optional[int] = None):
        """
        Inicializar ImageDescriptor con configuración del cliente Google Gemini
        
        Args:
            model_name: Nombre del modelo Gemini (default: gemini-2.5-flash)
            google_api_key: API key de Google (default: desde GOOGLE_API_KEY env var)
            timeout: Timeout para llamadas API en segundos (default: 30)
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.timeout = timeout or self.API_TIMEOUT
        
        # Configurar API key
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY", "")
        if not self.google_api_key:
            raise ValueError(
                "GOOGLE_API_KEY is required for ImageDescriptor. "
                "Please configure it in environment variables or pass it directly."
            )
        
        # Inicializar cliente Gemini
        self._initialize_client()
        
        # Inicializar generador de prompts
        self.prompt_generator = PromptGenerator(max_description_length=self.MAX_DESCRIPTION_LENGTH)
        
        # Inicializar sistema de reintentos y métricas
        self.retry_handler = RetryHandler(max_retries=3, base_delay=1.0, max_delay=60.0)
        self.metrics = DescriptionMetrics()
        
        # Logging de inicialización
        logger.info(f"ImageDescriptor initialized with model: {self.model_name}, "
                   f"timeout: {self.timeout}s, max_tokens: {self.MAX_TOKENS}, "
                   f"max_retries: {self.retry_handler.max_retries}")
    
    def _initialize_client(self) -> None:
        """
        Inicializar el cliente de Google Gemini con configuración optimizada
        """
        try:
            self.client = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.TEMPERATURE,
                google_api_key=self.google_api_key,
                max_tokens=self.MAX_TOKENS,
                convert_system_message_to_human=True,  # Gemini no soporta system messages
                timeout=self.timeout
            )
            logger.debug("Google Gemini client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Google Gemini client: {str(e)}")
            raise ValueError(f"Failed to initialize Gemini client: {str(e)}")
    
    def validate_image_path(self, image_path: str) -> bool:
        """
        Validar que la ruta de imagen existe y es válida
        
        Args:
            image_path: Ruta del archivo de imagen
            
        Returns:
            bool: True si la imagen es válida
            
        Raises:
            ValueError: Si la imagen no es válida
        """
        if not image_path:
            raise ValueError("Image path cannot be empty")
        
        # Convertir a Path para validación
        path = Path(image_path)
        
        # Verificar existencia
        if not path.exists():
            raise ValueError(f"Image file does not exist: {image_path}")
        
        # Verificar que es archivo
        if not path.is_file():
            raise ValueError(f"Path is not a file: {image_path}")
        
        # Verificar extensión
        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported image format: {path.suffix}. "
                           f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}")
        
        # Verificar tamaño
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.MAX_IMAGE_SIZE_MB:
            raise ValueError(f"Image file too large: {file_size_mb:.1f}MB. "
                           f"Maximum allowed: {self.MAX_IMAGE_SIZE_MB}MB")
        
        logger.debug(f"Image validation successful: {image_path} ({file_size_mb:.1f}MB)")
        return True
    
    def validate_image_data(self, image_data: str) -> bool:
        """
        Validar datos de imagen en base64
        
        Args:
            image_data: Imagen en formato base64 con data URI
            
        Returns:
            bool: True si los datos son válidos
            
        Raises:
            ValueError: Si los datos no son válidos
        """
        if not image_data:
            raise ValueError("Image data cannot be empty")
        
        # Verificar formato data URI
        if not image_data.startswith('data:image/'):
            raise ValueError("Image data must be in data URI format (data:image/...)")
        
        # Verificar que contiene base64
        if ';base64,' not in image_data:
            raise ValueError("Image data must contain base64 encoding")
        
        # Extraer y validar base64
        try:
            _, base64_data = image_data.split(';base64,')
            if not base64_data:
                raise ValueError("Base64 data is empty")
        except ValueError:
            raise ValueError("Invalid data URI format")
        
        logger.debug(f"Image data validation successful: {len(base64_data)} base64 chars")
        return True
    
    def get_configuration(self) -> Dict[str, Any]:
        """
        Obtener configuración actual del descriptor
        
        Returns:
            Dict con configuración actual
        """
        return {
            'model_name': self.model_name,
            'timeout': self.timeout,
            'chunk_size': self.CHUNK_SIZE,
            'max_tokens': self.MAX_TOKENS,
            'temperature': self.TEMPERATURE,
            'max_description_length': self.MAX_DESCRIPTION_LENGTH,
            'max_image_size_mb': self.MAX_IMAGE_SIZE_MB,
            'supported_formats': list(self.SUPPORTED_FORMATS)
        }
    
    def health_check(self) -> bool:
        """
        Verificar que el cliente está funcionando correctamente
        
        Returns:
            bool: True si el health check es exitoso
        """
        try:
            # Test básico con prompt simple
            test_message = HumanMessage(content="Responde solo 'OK'")
            response = self.client.invoke([test_message])
            
            if response and response.content:
                logger.debug("ImageDescriptor health check passed")
                return True
            else:
                logger.warning("ImageDescriptor health check failed: no response")
                return False
                
        except Exception as e:
            logger.error(f"ImageDescriptor health check failed: {str(e)}")
            return False
    
    def _load_image_as_base64(self, image_path: str) -> Optional[str]:
        """
        Cargar imagen del filesystem y convertirla a base64 con data URI
        (Reimplementación local de la función load_image_as_base64 de app.py)
        
        Args:
            image_path: Ruta del archivo de imagen
            
        Returns:
            Imagen codificada en base64 con data URI format o None si falla
        """
        try:
            # Manejar rutas absolutas y relativas
            if image_path.startswith('/var/data/rag/images/'):
                # Ya es una ruta absoluta
                full_path = image_path
            elif image_path.startswith('file://'):
                # Quitar prefijo file://
                full_path = image_path.replace('file://', '')
            else:
                # Asumir que es relativa al directorio de imágenes
                full_path = f"/var/data/rag/images/{image_path}"
            
            # Verificar existencia del archivo
            if not os.path.exists(full_path):
                logger.warning(f"Image file not found: {full_path}")
                return None
                
            # Leer y codificar la imagen
            with open(full_path, 'rb') as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Determinar mime type basado en extensión
            ext = Path(full_path).suffix.lower()
            mime_type = {
                '.png': 'image/png',
                '.jpg': 'image/jpeg', 
                '.jpeg': 'image/jpeg',
                '.gif': 'image/gif',
                '.bmp': 'image/bmp',
                '.webp': 'image/webp'
            }.get(ext, 'image/png')  # Default a PNG
            
            return f"data:{mime_type};base64,{encoded_image}"
            
        except Exception as e:
            logger.error(f"Error loading image as base64: {str(e)}")
            return None
    
    def describe_image(self,
                      image_path: str,
                      image_metadata: Optional[Dict] = None,
                      image_type: Optional['ImageContentType'] = None,
                      custom_instructions: Optional[str] = None) -> Optional[str]:
        """
        Generar descripción de una imagen usando Gemini 2.5 Flash con análisis multimodal
        
        Args:
            image_path: Ruta del archivo de imagen
            image_metadata: Metadata opcional de imagen (width, height, etc.)
            image_type: Tipo específico de imagen (auto-detectar si None)
            custom_instructions: Instrucciones adicionales específicas
            
        Returns:
            Descripción generada de la imagen o None si falla
        """
        start_time = time.time()
        
        try:
            # Validar ruta de imagen
            self.validate_image_path(image_path)
            
            # Cargar imagen como base64
            logger.debug(f"Loading image for description: {image_path}")
            image_base64 = self._load_image_as_base64(image_path)
            
            if not image_base64:
                logger.error(f"Failed to load image as base64: {image_path}")
                return None
            
            # Validar datos de imagen
            self.validate_image_data(image_base64)
            
            # Generar prompt especializado
            specialized_prompt = self.create_specialized_prompt(
                image_type=image_type,
                image_metadata=image_metadata,
                image_path=image_path,
                custom_instructions=custom_instructions
            )
            
            # Construir mensaje multimodal para Gemini
            message_content = [
                {
                    "type": "text",
                    "text": specialized_prompt
                },
                {
                    "type": "image_url", 
                    "image_url": image_base64
                }
            ]
            
            # Crear mensaje multimodal
            human_message = HumanMessage(content=message_content)
            
            # Realizar llamada a API de Gemini
            logger.info(f"Invoking Gemini API for image description: {image_path}")
            response = self.client.invoke([human_message])
            
            if not response or not response.content:
                logger.warning(f"Empty response from Gemini API for image: {image_path}")
                return None
            
            description = response.content.strip()
            
            # Validar longitud de descripción
            if len(description) > self.MAX_DESCRIPTION_LENGTH:
                logger.warning(f"Description too long ({len(description)} chars), truncating to {self.MAX_DESCRIPTION_LENGTH}")
                description = description[:self.MAX_DESCRIPTION_LENGTH].rsplit(' ', 1)[0] + "..."
            
            # Validar que no esté vacía
            if not description or len(description.strip()) < 10:
                logger.warning(f"Generated description is too short or empty: {description}")
                return None
            
            processing_time = time.time() - start_time
            logger.info(f"Successfully generated description for {image_path}: "
                       f"{len(description)} chars in {processing_time:.2f}s")
            
            return description
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Failed to generate description for {image_path} "
                        f"after {processing_time:.2f}s: {str(e)}")
            return None
    
    def describe_image_from_data(self,
                                image_data: str,
                                image_metadata: Optional[Dict] = None,
                                image_type: Optional['ImageContentType'] = None,
                                custom_instructions: Optional[str] = None) -> Optional[str]:
        """
        Generar descripción de imagen a partir de datos base64
        
        Args:
            image_data: Imagen en formato base64 con data URI
            image_metadata: Metadata opcional de imagen
            image_type: Tipo específico de imagen (auto-detectar si None)
            custom_instructions: Instrucciones adicionales específicas
            
        Returns:
            Descripción generada de la imagen o None si falla
        """
        start_time = time.time()
        
        try:
            # Validar datos de imagen
            self.validate_image_data(image_data)
            
            # Generar prompt especializado
            specialized_prompt = self.create_specialized_prompt(
                image_type=image_type,
                image_metadata=image_metadata,
                custom_instructions=custom_instructions
            )
            
            # Construir mensaje multimodal para Gemini
            message_content = [
                {
                    "type": "text",
                    "text": specialized_prompt
                },
                {
                    "type": "image_url",
                    "image_url": image_data
                }
            ]
            
            # Crear mensaje multimodal
            human_message = HumanMessage(content=message_content)
            
            # Realizar llamada a API de Gemini
            logger.info("Invoking Gemini API for image description from base64 data")
            response = self.client.invoke([human_message])
            
            if not response or not response.content:
                logger.warning("Empty response from Gemini API for image data")
                return None
            
            description = response.content.strip()
            
            # Validar longitud de descripción
            if len(description) > self.MAX_DESCRIPTION_LENGTH:
                logger.warning(f"Description too long ({len(description)} chars), truncating to {self.MAX_DESCRIPTION_LENGTH}")
                description = description[:self.MAX_DESCRIPTION_LENGTH].rsplit(' ', 1)[0] + "..."
            
            # Validar que no esté vacía
            if not description or len(description.strip()) < 10:
                logger.warning(f"Generated description is too short or empty: {description}")
                return None
            
            processing_time = time.time() - start_time
            logger.info(f"Successfully generated description from data: "
                       f"{len(description)} chars in {processing_time:.2f}s")
            
            return description
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Failed to generate description from image data "
                        f"after {processing_time:.2f}s: {str(e)}")
            return None
    
    def _generate_description_with_retry(self,
                                       message_content: List[Dict],
                                       correlation_id: str,
                                       context_info: str = "") -> Optional[str]:
        """
        Generar descripción con sistema de reintentos robusto
        
        Args:
            message_content: Contenido del mensaje multimodal
            correlation_id: ID de correlación para tracking
            context_info: Información contextual para logging
            
        Returns:
            Descripción generada o None si falla
        """
        attempt = 0
        last_error = None
        
        while attempt <= self.retry_handler.max_retries:
            try:
                # Crear mensaje multimodal
                human_message = HumanMessage(content=message_content)
                
                # Realizar llamada a API de Gemini
                logger.info(f"[{correlation_id}] Invoking Gemini API (attempt {attempt + 1}/{self.retry_handler.max_retries + 1}) {context_info}")
                response = self.client.invoke([human_message])
                
                if not response or not response.content:
                    raise ValueError("Empty response from Gemini API")
                
                description = response.content.strip()
                
                # Validar longitud de descripción
                if len(description) > self.MAX_DESCRIPTION_LENGTH:
                    logger.warning(f"[{correlation_id}] Description too long ({len(description)} chars), truncating to {self.MAX_DESCRIPTION_LENGTH}")
                    description = description[:self.MAX_DESCRIPTION_LENGTH].rsplit(' ', 1)[0] + "..."
                
                # Validar que no esté vacía
                if not description or len(description.strip()) < 10:
                    raise ValueError(f"Generated description is too short or empty: {description}")
                
                logger.info(f"[{correlation_id}] Successfully generated description: {len(description)} chars on attempt {attempt + 1}")
                return description
                
            except Exception as error:
                last_error = error
                error_type = self.retry_handler.classify_error(error)
                
                logger.error(f"[{correlation_id}] Attempt {attempt + 1} failed with {error_type.value}: {str(error)}")
                
                # Verificar si se debe reintentar
                if not self.retry_handler.should_retry(error_type, attempt):
                    logger.error(f"[{correlation_id}] Not retrying {error_type.value} error after {attempt + 1} attempts")
                    break
                
                # Calcular delay y esperar antes del siguiente intento
                if attempt < self.retry_handler.max_retries:
                    delay = self.retry_handler.calculate_delay(attempt, error_type)
                    logger.info(f"[{correlation_id}] Retrying in {delay:.2f} seconds (error: {error_type.value})")
                    time.sleep(delay)
                    self.metrics.record_retry()
                
                attempt += 1
        
        # Todos los intentos fallaron
        if last_error:
            final_error_type = self.retry_handler.classify_error(last_error)
            logger.error(f"[{correlation_id}] All retry attempts exhausted. Final error: {final_error_type.value}")
        
        return None
    
    def describe_image_robust(self,
                            image_path: str,
                            image_metadata: Optional[Dict] = None,
                            image_type: Optional['ImageContentType'] = None,
                            custom_instructions: Optional[str] = None) -> Optional[str]:
        """
        Generar descripción de imagen con sistema robusto de reintentos y manejo de errores
        
        Args:
            image_path: Ruta del archivo de imagen
            image_metadata: Metadata opcional de imagen (width, height, etc.)
            image_type: Tipo específico de imagen (auto-detectar si None)
            custom_instructions: Instrucciones adicionales específicas
            
        Returns:
            Descripción generada de la imagen o None si falla
        """
        correlation_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        
        try:
            # Validar ruta de imagen
            self.validate_image_path(image_path)
            
            # Cargar imagen como base64
            logger.debug(f"[{correlation_id}] Loading image for description: {image_path}")
            image_base64 = self._load_image_as_base64(image_path)
            
            if not image_base64:
                error_msg = f"Failed to load image as base64: {image_path}"
                logger.error(f"[{correlation_id}] {error_msg}")
                processing_time = time.time() - start_time
                self.metrics.record_failure(ErrorType.FILE_NOT_FOUND_ERROR, processing_time)
                return None
            
            # Validar datos de imagen
            self.validate_image_data(image_base64)
            
            # Generar prompt especializado
            specialized_prompt = self.create_specialized_prompt(
                image_type=image_type,
                image_metadata=image_metadata,
                image_path=image_path,
                custom_instructions=custom_instructions
            )
            
            # Construir mensaje multimodal para Gemini
            message_content = [
                {
                    "type": "text",
                    "text": specialized_prompt
                },
                {
                    "type": "image_url", 
                    "image_url": image_base64
                }
            ]
            
            # Generar descripción con reintentos
            description = self._generate_description_with_retry(
                message_content=message_content,
                correlation_id=correlation_id,
                context_info=f"for image: {image_path}"
            )
            
            processing_time = time.time() - start_time
            
            if description:
                self.metrics.record_success(processing_time, len(description))
                logger.info(f"[{correlation_id}] Successfully generated description for {image_path}: "
                           f"{len(description)} chars in {processing_time:.2f}s")
                return description
            else:
                self.metrics.record_failure(ErrorType.UNKNOWN_ERROR, processing_time)
                return None
                
        except Exception as e:
            processing_time = time.time() - start_time
            error_type = self.retry_handler.classify_error(e)
            self.metrics.record_failure(error_type, processing_time)
            logger.error(f"[{correlation_id}] Failed to generate description for {image_path} "
                        f"after {processing_time:.2f}s: {str(e)}")
            return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Obtener resumen de métricas actuales
        
        Returns:
            Dict con métricas de rendimiento y errores
        """
        return self.metrics.get_summary()
    
    def reset_metrics(self) -> None:
        """Reiniciar todas las métricas"""
        self.metrics.reset()
        logger.info("ImageDescriptor metrics reset")
    
    def detect_content_type(self, 
                           image_metadata: Optional[Dict] = None,
                           image_path: Optional[str] = None) -> 'ImageContentType':
        """
        Detectar automáticamente el tipo de contenido de una imagen
        
        Args:
            image_metadata: Metadata de imagen extraída (width, height, etc.)
            image_path: Ruta del archivo de imagen para contexto
            
        Returns:
            ImageContentType detectado
        """
        return self.prompt_generator.detect_image_type(image_metadata, image_path)
    
    def create_specialized_prompt(self,
                                 image_type: Optional['ImageContentType'] = None,
                                 image_metadata: Optional[Dict] = None,
                                 image_path: Optional[str] = None,
                                 custom_instructions: Optional[str] = None) -> str:
        """
        Crear prompt especializado para análisis de imagen
        
        Args:
            image_type: Tipo específico de imagen (auto-detectar si None)
            image_metadata: Metadata de imagen para contexto
            image_path: Ruta de imagen para contexto  
            custom_instructions: Instrucciones adicionales específicas
            
        Returns:
            Prompt optimizado para el tipo de imagen
        """
        return self.prompt_generator.generate_prompt(
            image_type=image_type,
            image_metadata=image_metadata,
            image_path=image_path,
            custom_instructions=custom_instructions
        )
    
    def get_supported_content_types(self) -> Dict[str, str]:
        """
        Obtener tipos de contenido soportados
        
        Returns:
            Dict con tipos soportados y sus descripciones
        """
        return self.prompt_generator.get_supported_types()


class ImageContentType(Enum):
    """Tipos de contenido visual soportados"""
    CHART = "chart"          # Gráficos, barras, líneas, sectores
    TABLE = "table"          # Tablas de datos
    DIAGRAM = "diagram"      # Diagramas de flujo, esquemas, organigramas  
    PHOTO = "photo"          # Fotografías reales
    GENERAL = "general"      # Imágenes generales o no clasificadas


class PromptGenerator:
    """
    Generador de prompts especializados para diferentes tipos de imágenes
    """
    
    # Plantillas de prompts base
    BASE_INSTRUCTION = (
        "Analiza esta imagen y proporciona una descripción factual, clara y concisa. "
        "NO inventes datos que no sean visibles en la imagen. "
        "Limita la descripción a un máximo de {max_length} caracteres."
    )
    
    PROMPT_TEMPLATES = {
        ImageContentType.CHART: (
            "{base_instruction} "
            "Esta imagen contiene un gráfico o chart. Describe:\n"
            "1. Tipo de gráfico (barras, líneas, sectores, etc.)\n"
            "2. Ejes y escalas visibles\n"
            "3. Tendencias principales observables\n"
            "4. Valores aproximados cuando sean claramente legibles\n"
            "5. Título y etiquetas si están presentes\n"
            "Enfócate en patrones y tendencias, no en valores exactos a menos que sean muy claros."
        ),
        
        ImageContentType.TABLE: (
            "{base_instruction} "
            "Esta imagen contiene una tabla de datos. Describe:\n"
            "1. Estructura de la tabla (número aproximado de filas/columnas)\n"
            "2. Encabezados de columnas si son legibles\n"
            "3. Tipo de datos contenidos (números, texto, fechas, etc.)\n"
            "4. Patrones notables en los datos\n"
            "5. Título de la tabla si está presente\n"
            "Si el texto es demasiado pequeño para leer, menciona que contiene datos tabulares pero no inventes contenido específico."
        ),
        
        ImageContentType.DIAGRAM: (
            "{base_instruction} "
            "Esta imagen contiene un diagrama o esquema. Describe:\n"
            "1. Tipo de diagrama (flujo, organigrama, esquema, etc.)\n"
            "2. Elementos principales y su organización\n"
            "3. Conexiones y flujos entre elementos\n"
            "4. Texto legible en cajas o nodos\n"
            "5. Propósito aparente del diagrama\n"
            "Enfócate en la estructura y relaciones, no en detalles de texto que no sean claramente legibles."
        ),
        
        ImageContentType.PHOTO: (
            "{base_instruction} "
            "Esta imagen es una fotografía. Describe:\n"
            "1. Sujetos principales y su ubicación\n"
            "2. Entorno o contexto visible\n"
            "3. Elementos notables o distintivos\n"
            "4. Colores y composición predominantes\n"
            "5. Texto visible si lo hay\n"
            "Proporciona una descripción objetiva de lo que se ve sin interpretaciones subjetivas."
        ),
        
        ImageContentType.GENERAL: (
            "{base_instruction} "
            "Describe el contenido de esta imagen de manera objetiva:\n"
            "1. Elementos principales visibles\n"
            "2. Organización y disposición\n"
            "3. Colores y características destacadas\n"
            "4. Texto legible si está presente\n"
            "5. Contexto o propósito aparente\n"
            "Mantén la descripción factual y concisa."
        )
    }
    
    def __init__(self, max_description_length: int = 1000):
        """
        Inicializar generador de prompts
        
        Args:
            max_description_length: Longitud máxima para descripciones
        """
        self.max_description_length = max_description_length
        logger.debug(f"PromptGenerator initialized with max_length: {max_description_length}")
    
    def detect_image_type(self, 
                         image_metadata: Optional[Dict] = None,
                         image_path: Optional[str] = None) -> ImageContentType:
        """
        Detectar automáticamente el tipo de contenido basado en metadata y contexto
        
        Args:
            image_metadata: Metadata de imagen extraída (width, height, etc.)
            image_path: Ruta del archivo de imagen para contexto
            
        Returns:
            ImageContentType detectado
        """
        # Palabras clave para detectar tipos en nombres de archivo
        filename_keywords = {
            ImageContentType.CHART: ['chart', 'graph', 'plot', 'grafico', 'grafica'],
            ImageContentType.TABLE: ['table', 'tabla', 'data', 'datos'],
            ImageContentType.DIAGRAM: ['diagram', 'flow', 'schema', 'diagrama', 'flujo', 'esquema', 'organi'],
            ImageContentType.PHOTO: ['photo', 'foto', 'picture', 'image', 'img']
        }
        
        detected_type = ImageContentType.GENERAL  # Default
        
        # Análisis por nombre de archivo si está disponible
        if image_path:
            filename_lower = Path(image_path).stem.lower()
            for content_type, keywords in filename_keywords.items():
                if any(keyword in filename_lower for keyword in keywords):
                    detected_type = content_type
                    break
        
        # Análisis por dimensiones y aspect ratio
        if image_metadata and 'width' in image_metadata and 'height' in image_metadata:
            width = image_metadata['width']
            height = image_metadata['height']
            aspect_ratio = max(width, height) / min(width, height)
            
            # Heurísticas basadas en dimensiones
            if aspect_ratio > 3.0:  # Imágenes muy alargadas suelen ser tablas o charts
                if detected_type == ImageContentType.GENERAL:
                    detected_type = ImageContentType.TABLE
            elif width > 800 and height > 600:  # Imágenes grandes suelen ser diagramas o fotos
                if detected_type == ImageContentType.GENERAL:
                    detected_type = ImageContentType.DIAGRAM
        
        logger.debug(f"Detected image type: {detected_type.value} for path: {image_path}")
        return detected_type
    
    def generate_prompt(self,
                       image_type: Optional[ImageContentType] = None,
                       image_metadata: Optional[Dict] = None,
                       image_path: Optional[str] = None,
                       custom_instructions: Optional[str] = None) -> str:
        """
        Generar prompt especializado para un tipo de imagen
        
        Args:
            image_type: Tipo específico de imagen (auto-detectar si None)
            image_metadata: Metadata de imagen para contexto
            image_path: Ruta de imagen para contexto
            custom_instructions: Instrucciones adicionales específicas
            
        Returns:
            Prompt optimizado para el tipo de imagen
        """
        # Auto-detectar tipo si no se especifica
        if image_type is None:
            image_type = self.detect_image_type(image_metadata, image_path)
        
        # Obtener template base
        template = self.PROMPT_TEMPLATES.get(image_type, self.PROMPT_TEMPLATES[ImageContentType.GENERAL])
        
        # Construir instrucción base con límite de longitud
        base_instruction = self.BASE_INSTRUCTION.format(max_length=self.max_description_length)
        
        # Formatear template
        prompt = template.format(base_instruction=base_instruction)
        
        # Agregar instrucciones personalizadas si están disponibles
        if custom_instructions:
            prompt += f"\n\nInstrucciones adicionales: {custom_instructions}"
        
        # Validar longitud del prompt (debe caber en el contexto disponible)
        if len(prompt) > (self.max_description_length * 2):
            logger.warning(f"Generated prompt is very long ({len(prompt)} chars), may affect performance")
        
        logger.debug(f"Generated prompt for type {image_type.value}: {len(prompt)} characters")
        return prompt
    
    def get_supported_types(self) -> Dict[str, str]:
        """
        Obtener tipos soportados y sus descripciones
        
        Returns:
            Dict con tipos soportados y descripciones
        """
        return {
            content_type.value: content_type.name.lower().replace('_', ' ').title()
            for content_type in ImageContentType
        }