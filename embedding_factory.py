"""
Factory para crear diferentes tipos de embeddings de forma configurable
"""

import os
import base64
import logging
import time
from typing import Any, Dict, Optional, List, Union
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from jina_embeddings import JinaEmbeddings

logger = logging.getLogger(__name__)

class EmbeddingMetrics:
    """
    Clase para registrar y almacenar métricas de generación de embeddings de imágenes
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reinicia todas las métricas"""
        self.image_embeddings_generated = 0
        self.text_embeddings_generated = 0
        self.total_embedding_time = 0.0
        self.image_embedding_errors = 0
        self.text_embedding_errors = 0
        self.embedding_times = []
        self.errors_by_type = {}
        self.images_processed_by_batch = []
        self.avg_latency_per_image = []
    
    def record_image_embedding_batch(self, batch_size: int, processing_time: float, success: bool = True):
        """Registra métricas de procesamiento de un batch de embeddings de imagen"""
        if success:
            self.image_embeddings_generated += batch_size
            self.total_embedding_time += processing_time
            self.embedding_times.append(processing_time)
            self.images_processed_by_batch.append(batch_size)
            
            if batch_size > 0:
                latency_per_image = processing_time / batch_size
                self.avg_latency_per_image.append(latency_per_image)
        else:
            self.image_embedding_errors += batch_size
            self.total_embedding_time += processing_time
    
    def record_text_embedding_batch(self, batch_size: int, processing_time: float, success: bool = True):
        """Registra métricas de procesamiento de un batch de embeddings de texto"""
        if success:
            self.text_embeddings_generated += batch_size
            self.total_embedding_time += processing_time
            self.embedding_times.append(processing_time)
        else:
            self.text_embedding_errors += batch_size
    
    def record_embedding_error(self, error_type: str, batch_size: int = 1):
        """Registra un error de generación de embedding"""
        self.errors_by_type[error_type] = self.errors_by_type.get(error_type, 0) + batch_size
    
    def get_metrics_summary(self) -> Dict:
        """Retorna resumen completo de métricas de embeddings"""
        total_embeddings = self.image_embeddings_generated + self.text_embeddings_generated
        total_errors = self.image_embedding_errors + self.text_embedding_errors
        
        success_rate = ((total_embeddings) / (total_embeddings + total_errors) * 100) if (total_embeddings + total_errors) > 0 else 100.0
        avg_processing_time = self.total_embedding_time / len(self.embedding_times) if self.embedding_times else 0.0
        avg_images_per_batch = sum(self.images_processed_by_batch) / len(self.images_processed_by_batch) if self.images_processed_by_batch else 0.0
        avg_latency_per_image = sum(self.avg_latency_per_image) / len(self.avg_latency_per_image) if self.avg_latency_per_image else 0.0
        
        return {
            'embedding_summary': {
                'total_image_embeddings': self.image_embeddings_generated,
                'total_text_embeddings': self.text_embeddings_generated,
                'total_embedding_errors': total_errors,
                'image_embedding_errors': self.image_embedding_errors,
                'text_embedding_errors': self.text_embedding_errors,
                'success_rate_percent': round(success_rate, 2)
            },
            'performance_metrics': {
                'total_embedding_time_seconds': round(self.total_embedding_time, 2),
                'avg_batch_processing_time_seconds': round(avg_processing_time, 2),
                'avg_images_per_batch': round(avg_images_per_batch, 2),
                'avg_latency_per_image_ms': round(avg_latency_per_image * 1000, 2),
                'max_batch_time_seconds': round(max(self.embedding_times) if self.embedding_times else 0, 2),
                'min_batch_time_seconds': round(min(self.embedding_times) if self.embedding_times else 0, 2)
            },
            'error_breakdown': self.errors_by_type
        }

# Instancia global de métricas de embeddings
embedding_metrics = EmbeddingMetrics()

class EmbeddingFactory:
    """Factory para crear instancias de embeddings configurables"""
    
    SUPPORTED_MODELS = {
        "google": {
            "models/embedding-001": {"dimensions": 768},
            "models/text-embedding-004": {"dimensions": 768},
        },
        "jina": {
            "jina-embeddings-v4": {"dimensions": 1024},  # Configurable 128-2048
        }
    }
    
    @classmethod
    def create_embedding(
        self,
        model_name: str = None,
        provider: str = None,
        **kwargs
    ) -> Any:
        """
        Crea una instancia de embedding basada en el modelo y proveedor especificados
        
        Args:
            model_name: Nombre del modelo (si no se especifica, usa variables de entorno)
            provider: Proveedor (google, jina) (si no se especifica, intenta detectar)
            **kwargs: Argumentos adicionales para el modelo
            
        Returns:
            Instancia de embeddings configurada
        """
        # Obtener configuración de variables de entorno si no se especifica
        if not model_name:
            model_name = os.getenv("EMBEDDING_MODEL", "jina-embeddings-v4")
        
        if not provider:
            provider = os.getenv("EMBEDDING_PROVIDER", "jina")
        
        logger.info(f"Creating {provider} embedding with model: {model_name}")
        
        if provider == "google":
            return self._create_google_embedding(model_name, **kwargs)
        elif provider == "jina":
            return self._create_jina_embedding(model_name, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider}. Supported: 'google', 'jina'")
    
    
    @classmethod
    def _create_google_embedding(self, model_name: str, **kwargs) -> GoogleGenerativeAIEmbeddings:
        """Crea embedding de Google Gemini"""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY es requerida para usar embeddings de Google. "
                "Configúrala en las variables de entorno."
            )
        
        # Configuración por defecto para Gemini
        default_config = {
            "model": model_name,
            "google_api_key": api_key,
            "dimensions": kwargs.get("dimensions", 768),  # Dimensión recomendada
            "task_type": "retrieval_document",  # Optimizado para RAG
        }
        
        # Mergear con kwargs personalizados
        config = {**default_config, **kwargs}
        
        try:
            return GoogleGenerativeAIEmbeddings(**config)
        except Exception as e:
            logger.error(f"Error creando Google embedding: {e}")
            logger.info("Verificar que GOOGLE_API_KEY esté configurada correctamente")
            raise
    
    @classmethod
    def _create_jina_embedding(self, model_name: str, **kwargs) -> JinaEmbeddings:
        """Crea embedding de Jina v4"""
        api_key = os.getenv("JINA_API_KEY")
        if not api_key:
            raise ValueError(
                "JINA_API_KEY es requerida para usar embeddings de Jina. "
                "Configúrala en las variables de entorno."
            )
        
        # Configuración por defecto para Jina v4
        default_config = {
            "model": model_name,
            "api_key": api_key,
            "dimensions": kwargs.get("dimensions", 1024),  # Balance óptimo
            "normalized": True,  # Para usar DOT distance en Qdrant
            "late_chunking": kwargs.get("late_chunking", False),  # Configurable
            "task_type": kwargs.get("task_type", "retrieval.passage"),  # Consistente con imágenes
        }
        
        # Mergear con kwargs personalizados
        config = {**default_config, **kwargs}
        
        try:
            return JinaEmbeddings(**config)
        except Exception as e:
            logger.error(f"Error creando Jina embedding: {e}")
            logger.info("Verificar que JINA_API_KEY esté configurada correctamente")
            raise
    
    @classmethod
    def get_model_dimensions(self, model_name: str, provider: str = None) -> int:
        """Obtiene las dimensiones del modelo especificado"""
        if not provider:
            provider = os.getenv("EMBEDDING_PROVIDER", "jina")
        
        if provider not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported provider: {provider}. Supported: {list(self.SUPPORTED_MODELS.keys())}")
        
        model_info = self.SUPPORTED_MODELS.get(provider, {}).get(model_name)
        if model_info:
            return model_info["dimensions"]
        
        # Valores por defecto según el proveedor
        if provider == "google":
            return 768  # Gemini embedding-001 default
        elif provider == "jina":
            return 1024  # Jina v4 default optimizado
        
        return 1024  # Fallback general (Jina default)
    
    @classmethod
    def create_image_embedding(
        self,
        model_name: str = None,
        provider: str = None,
        **kwargs
    ) -> Any:
        """
        Crea una instancia de embedding específicamente configurada para imágenes
        
        Args:
            model_name: Nombre del modelo (si no se especifica, usa variables de entorno)
            provider: Proveedor (por defecto jina para imágenes)
            **kwargs: Argumentos adicionales para el modelo
            
        Returns:
            Instancia de embeddings configurada para imágenes
        """
        # Para imágenes, usar Jina por defecto ya que soporta mejor multimodal
        if not provider:
            provider = os.getenv("IMAGE_EMBEDDING_PROVIDER", "jina")
        
        if not model_name:
            model_name = os.getenv("IMAGE_EMBEDDING_MODEL", "jina-embeddings-v4")
        
        logger.info(f"Creating {provider} image embedding with model: {model_name}")
        
        if provider == "jina":
            # Configurar específicamente para imágenes con task_type apropiado
            image_kwargs = {
                "task_type": "retrieval.passage",  # Optimizado para documento/imagen
                "dimensions": kwargs.get("dimensions", 1024),  # Consistente con texto
                "normalized": True,  # Para uso con DOT distance en Qdrant
                "late_chunking": False,  # No necesario para imágenes
                **kwargs
            }
            return self._create_jina_embedding(model_name, **image_kwargs)
        elif provider == "google":
            # Google no soporta imágenes nativamente en embedding-001
            raise ValueError(
                "Google embeddings no soporta imágenes directamente. "
                "Usa 'jina' como provider para embeddings de imagen."
            )
        else:
            raise ValueError(f"Unsupported provider for images: {provider}. Supported: 'jina'")
    
    @classmethod
    def embed_images_flexible(
        self,
        images: Union[List[str], List[bytes], str, bytes],
        model_name: str = None,
        provider: str = None,
        **kwargs
    ) -> List[List[float]]:
        """
        Método de conveniencia para generar embeddings de imágenes desde múltiples formatos
        
        Args:
            images: Rutas de archivos, datos en bytes, o strings base64 (individual o lista)
            model_name: Nombre del modelo de embedding a usar
            provider: Proveedor de embedding (por defecto jina)
            **kwargs: Argumentos adicionales
            
        Returns:
            Lista de embeddings de imágenes
        """
        # Normalizar input a lista
        if not isinstance(images, list):
            images = [images]
        
        # Crear instancia de embedding configurada para imágenes
        embedding_instance = self.create_image_embedding(model_name, provider, **kwargs)
        
        # Procesar según el tipo de datos
        processed_images = []
        image_data_list = []
        
        for img in images:
            if isinstance(img, str):
                # Verificar si es base64 o ruta de archivo
                if self._is_base64_image(img):
                    # Es base64, decodificar a bytes
                    try:
                        img_bytes = base64.b64decode(img)
                        image_data_list.append(img_bytes)
                    except Exception as e:
                        logger.error(f"Error decoding base64 image: {e}")
                        raise ValueError(f"Invalid base64 image data")
                else:
                    # Es ruta de archivo
                    processed_images.append(img)
            elif isinstance(img, bytes):
                # Es datos raw
                image_data_list.append(img)
            else:
                raise ValueError(f"Unsupported image type: {type(img)}. Expected str or bytes.")
        
        embeddings = []
        
        # Procesar imágenes de archivo
        if processed_images:
            start_time = time.time()
            try:
                file_embeddings = embedding_instance.embed_images(processed_images)
                embeddings.extend(file_embeddings)
                processing_time = time.time() - start_time
                
                # Registrar métricas de éxito
                embedding_metrics.record_image_embedding_batch(
                    batch_size=len(processed_images),
                    processing_time=processing_time,
                    success=True
                )
                
                logger.debug(f"Generated embeddings for {len(processed_images)} image files in {processing_time:.2f}s")
                
            except Exception as e:
                processing_time = time.time() - start_time
                error_type = type(e).__name__
                
                # Registrar métricas de error
                embedding_metrics.record_image_embedding_batch(
                    batch_size=len(processed_images),
                    processing_time=processing_time,
                    success=False
                )
                embedding_metrics.record_embedding_error(error_type, len(processed_images))
                
                logger.error(f"Error processing image files: {e}")
                raise
        
        # Procesar imágenes de datos
        if image_data_list:
            start_time = time.time()
            try:
                data_embeddings = embedding_instance.embed_images_data(image_data_list)
                embeddings.extend(data_embeddings)
                processing_time = time.time() - start_time
                
                # Registrar métricas de éxito
                embedding_metrics.record_image_embedding_batch(
                    batch_size=len(image_data_list),
                    processing_time=processing_time,
                    success=True
                )
                
                logger.debug(f"Generated embeddings for {len(image_data_list)} image data items in {processing_time:.2f}s")
                
            except Exception as e:
                processing_time = time.time() - start_time
                error_type = type(e).__name__
                
                # Registrar métricas de error
                embedding_metrics.record_image_embedding_batch(
                    batch_size=len(image_data_list),
                    processing_time=processing_time,
                    success=False
                )
                embedding_metrics.record_embedding_error(error_type, len(image_data_list))
                
                logger.error(f"Error processing image data: {e}")
                raise
        
        return embeddings
    
    @classmethod
    def _is_base64_image(self, s: str) -> bool:
        """
        Verifica si un string es una imagen codificada en base64
        
        Args:
            s: String a verificar
            
        Returns:
            True si es base64 válido, False si no
        """
        try:
            # Verificar longitud mínima y caracteres válidos
            if len(s) < 100:  # Imagen muy pequeña para ser válida
                return False
            
            # Verificar caracteres de base64
            if not all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=' for c in s):
                return False
            
            # Intentar decodificar
            decoded = base64.b64decode(s, validate=True)
            
            # Verificar headers de imagen comunes
            return (
                decoded.startswith(b'\xff\xd8\xff') or  # JPEG
                decoded.startswith(b'\x89PNG\r\n\x1a\n') or  # PNG
                decoded.startswith(b'GIF87a') or decoded.startswith(b'GIF89a') or  # GIF
                decoded.startswith(b'\x42\x4d') or  # BMP
                decoded.startswith(b'RIFF') and b'WEBP' in decoded[:12]  # WebP
            )
            
        except (ValueError, TypeError):
            return False
    
    @classmethod
    def verify_embedding_compatibility(
        self,
        text_model_name: str = None,
        image_model_name: str = None,
        provider: str = None
    ) -> Dict[str, Any]:
        """
        Verifica la compatibilidad entre embeddings de texto e imagen
        
        Args:
            text_model_name: Modelo para texto (opcional)
            image_model_name: Modelo para imagen (opcional) 
            provider: Proveedor a verificar (opcional)
            
        Returns:
            Diccionario con información de compatibilidad
        """
        provider = provider or os.getenv("EMBEDDING_PROVIDER", "jina")
        text_model = text_model_name or os.getenv("EMBEDDING_MODEL", "jina-embeddings-v4")
        image_model = image_model_name or os.getenv("IMAGE_EMBEDDING_MODEL", "jina-embeddings-v4")
        
        # Obtener dimensiones de ambos modelos
        text_dims = self.get_model_dimensions(text_model, provider)
        image_dims = self.get_model_dimensions(image_model, provider)
        
        # Crear instancias para verificar configuración
        try:
            text_embedding = self.create_embedding(text_model, provider)
            image_embedding = self.create_image_embedding(image_model, provider)
            
            compatibility = {
                "compatible": text_dims == image_dims,
                "text_model": text_model,
                "image_model": image_model,
                "provider": provider,
                "text_dimensions": text_dims,
                "image_dimensions": image_dims,
                "same_dimensions": text_dims == image_dims,
                "text_config": {
                    "model": text_embedding.model,
                    "dimensions": text_embedding.dimensions,
                    "normalized": text_embedding.normalized,
                    "task_type": text_embedding.task_type
                },
                "image_config": {
                    "model": image_embedding.model,
                    "dimensions": image_embedding.dimensions,
                    "normalized": image_embedding.normalized,
                    "task_type": image_embedding.task_type
                }
            }
            
            # Verificar configuraciones críticas
            critical_match = (
                compatibility["same_dimensions"] and
                compatibility["text_config"]["normalized"] == compatibility["image_config"]["normalized"] and
                compatibility["text_config"]["task_type"] == compatibility["image_config"]["task_type"]
            )
            
            compatibility["critical_settings_match"] = critical_match
            compatibility["ready_for_mixed_search"] = critical_match
            
            if critical_match:
                logger.info(f"✅ Text and image embeddings are fully compatible for mixed search")
            else:
                logger.warning(f"⚠️  Text and image embeddings have compatibility issues")
            
            return compatibility
            
        except Exception as e:
            logger.error(f"Error verifying embedding compatibility: {e}")
            return {
                "compatible": False,
                "error": str(e),
                "text_model": text_model,
                "image_model": image_model,
                "provider": provider
            }
    
    @classmethod
    def list_supported_models(self) -> Dict[str, Dict[str, Any]]:
        """Lista todos los modelos soportados"""
        return self.SUPPORTED_MODELS
    
    @classmethod
    def get_embedding_metrics(self) -> Dict:
        """
        Retorna métricas actuales de generación de embeddings
        
        Returns:
            Diccionario con métricas de embeddings
        """
        return embedding_metrics.get_metrics_summary()