"""
Factory para crear diferentes tipos de embeddings de forma configurable
"""

import os
import logging
from typing import Any, Dict, Optional
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from jina_embeddings import JinaEmbeddings

logger = logging.getLogger(__name__)

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
            model_name = os.getenv("EMBEDDING_MODEL", "models/embedding-001")
        
        if not provider:
            provider = os.getenv("EMBEDDING_PROVIDER", "google")
        
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
            "task_type": kwargs.get("task_type", "retrieval.document"),  # Default para ingesta
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
            provider = os.getenv("EMBEDDING_PROVIDER", "google")
        
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
        
        return 768  # Fallback general
    
    @classmethod
    def list_supported_models(self) -> Dict[str, Dict[str, Any]]:
        """Lista todos los modelos soportados"""
        return self.SUPPORTED_MODELS