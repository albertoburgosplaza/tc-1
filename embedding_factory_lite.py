"""
Factory para crear diferentes tipos de embeddings de forma configurable - Versión lite sin HuggingFace
"""

import os
import logging
from typing import Any, Dict, Optional
from langchain_google_genai import GoogleGenerativeAIEmbeddings

logger = logging.getLogger(__name__)

class EmbeddingFactory:
    """Factory para crear instancias de embeddings configurables - Solo Google"""
    
    SUPPORTED_MODELS = {
        "google": {
            "models/embedding-001": {"dimensions": 768},
            "models/text-embedding-004": {"dimensions": 768},
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
            provider: Proveedor (google) (si no se especifica, intenta detectar)
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
        else:
            raise ValueError(f"Proveedor no soportado en versión lite: {provider}")
    
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
    def get_model_dimensions(self, model_name: str, provider: str = None) -> int:
        """Obtiene las dimensiones del modelo especificado"""
        if not provider:
            provider = "google"
        
        model_info = self.SUPPORTED_MODELS.get(provider, {}).get(model_name)
        if model_info:
            return model_info["dimensions"]
        
        # Valor por defecto para Google
        return 768  # Gemini embedding-001 default
    
    @classmethod
    def list_supported_models(self) -> Dict[str, Dict[str, Any]]:
        """Lista todos los modelos soportados"""
        return self.SUPPORTED_MODELS