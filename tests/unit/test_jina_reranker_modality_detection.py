"""
Tests unitarios para funcionalidad de detección de modalidad en JinaReranker.

Tests exhaustivos que cubren:
- Detección de modalidad básica (texto, imagen, mixto)
- Casos edge (metadata corrupta, valores no estándar)
- Clasificación de contenido mixto
- Preservación de metadata original
- Rendimiento y escalabilidad
- Mocking de dependencias externas
"""

import pytest
import time
from unittest.mock import Mock, patch
import base64

# Import the JinaReranker class
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from jina_reranker import JinaReranker


class TestJinaRerankerModalityDetection:
    """Test suite para detección de modalidad en JinaReranker"""
    
    @pytest.fixture
    def mock_reranker(self):
        """Crear instancia de JinaReranker con API key mock"""
        with patch.dict(os.environ, {'JINA_API_KEY': 'test_key'}):
            reranker = JinaReranker()
            return reranker
    
    @pytest.fixture
    def text_document_samples(self):
        """Documentos de solo texto para testing"""
        return [
            # Texto simple string
            "Este es un documento de texto simple",
            
            # Dict con page_content
            {
                "page_content": "Contenido de página de texto",
                "metadata": {
                    "modality": "text",
                    "page_number": 1,
                    "source": "document.pdf"
                }
            },
            
            # Dict con content
            {
                "content": "Contenido alternativo de texto",
                "metadata": {
                    "doc_id": "doc123",
                    "title": "Documento de ejemplo"
                }
            },
            
            # Objeto simulando LangChain Document
            Mock(
                page_content="Contenido de LangChain Document",
                metadata={"modality": "text", "source_uri": "file:///path/doc.pdf"}
            )
        ]
    
    @pytest.fixture  
    def image_document_samples(self):
        """Documentos de solo imagen para testing"""
        return [
            # String URI de imagen
            "file:///var/data/rag/images/chart_001.png",
            
            # String data URI
            "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD//2Q==",
            
            # Dict con imagen en metadata
            {
                "page_content": "",
                "metadata": {
                    "modality": "image",
                    "thumbnail_uri": "/var/data/rag/images/thumb_001.jpg",
                    "width": 800,
                    "height": 600,
                    "image_index": 0
                }
            },
            
            # Dict con image_url
            {
                "content": "",
                "image_url": "https://example.com/image.png",
                "metadata": {
                    "source": "web_scraping"
                }
            },
            
            # Objeto con imagen en metadata
            Mock(
                page_content="",
                metadata={
                    "modality": "image",
                    "source_uri": "file:///var/data/rag/images/diagram.png",
                    "bbox": {"x0": 10, "y0": 20, "x1": 300, "y1": 200}
                }
            )
        ]
    
    @pytest.fixture
    def mixed_document_samples(self):
        """Documentos mixtos (texto + imagen) para testing"""
        return [
            # Dict con texto e imagen
            {
                "page_content": "Esta página contiene texto y una imagen",
                "metadata": {
                    "thumbnail_uri": "/var/data/rag/images/page_2_img_1.png",
                    "page_number": 2,
                    "doc_id": "mixed_doc"
                }
            },
            
            # Dict con content e image_url
            {
                "content": "Descripción del gráfico mostrado abajo",
                "image_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
                "metadata": {
                    "title": "Reporte con gráfico"
                }
            },
            
            # Objeto LangChain con texto e imagen
            Mock(
                page_content="Análisis de datos con visualización",
                metadata={
                    "image": "file:///tmp/analysis_chart.svg",
                    "modality": "mixed"  # Modalidad explícita
                }
            )
        ]
    
    @pytest.fixture
    def edge_case_samples(self):
        """Casos edge para testing robusto"""
        return [
            # Metadata corrupta
            {
                "page_content": "Texto con metadata corrupta",
                "metadata": None
            },
            
            # Metadata con valores no estándar
            {
                "content": "Contenido con modalidad desconocida",
                "metadata": {
                    "modality": "unknown",
                    "type": "weird_format"
                }
            },
            
            # Documento vacío
            {
                "page_content": "",
                "content": "",
                "metadata": {}
            },
            
            # Solo metadata sin contenido
            {
                "metadata": {
                    "modality": "text",
                    "title": "Solo metadata"
                }
            },
            
            # Imagen con base64 malformado
            {
                "page_content": "Texto con imagen corrupta",
                "metadata": {
                    "image": "data:image/jpeg;base64,INVALID_BASE64!!!"
                }
            },
            
            # URI de imagen no válida
            {
                "content": "Texto con URI inválida",
                "image_url": "not_a_valid_url"
            }
        ]
    
    def test_detect_text_only_modality(self, mock_reranker, text_document_samples):
        """Test detección de modalidad para documentos solo texto"""
        for doc in text_document_samples:
            modality = mock_reranker._detect_modality(
                doc if isinstance(doc, dict) else {"page_content": doc}
            )
            assert modality == "text", f"Failed for document: {doc}"
    
    def test_detect_image_only_modality(self, mock_reranker, image_document_samples):
        """Test detección de modalidad para documentos solo imagen"""
        for doc in image_document_samples:
            if isinstance(doc, str):
                # Para strings, verificar que es referencia de imagen
                assert mock_reranker._is_image_reference(doc)
            else:
                modality = mock_reranker._detect_modality(doc)
                assert modality == "image", f"Failed for document: {doc}"
    
    def test_detect_mixed_modality(self, mock_reranker, mixed_document_samples):
        """Test detección de modalidad para documentos mixtos"""
        for doc in mixed_document_samples:
            if hasattr(doc, 'page_content'):  # Mock object
                doc_dict = {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                }
            else:
                doc_dict = doc
            
            modality = mock_reranker._detect_modality(doc_dict)
            assert modality in ["mixed", "image"], f"Failed for document: {doc}"
    
    def test_edge_cases_robustness(self, mock_reranker, edge_case_samples):
        """Test manejo robusto de casos edge"""
        for doc in edge_case_samples:
            # No debería lanzar excepciones
            try:
                modality = mock_reranker._detect_modality(doc)
                assert modality in ["text", "image", "mixed"]
            except Exception as e:
                pytest.fail(f"Edge case failed: {doc} - Error: {e}")
    
    def test_is_image_reference_detection(self, mock_reranker):
        """Test detección de referencias de imagen"""
        # URLs válidas de imagen
        valid_image_refs = [
            "https://example.com/image.jpg",
            "http://site.com/photo.png",
            "file:///var/data/rag/images/chart.svg",
            "/var/data/rag/images/diagram.webp",
            "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD//2Q==",
            "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAY=",
        ]
        
        # Referencias no válidas
        invalid_refs = [
            "https://example.com/document.pdf",
            "texto normal sin imagen",
            "",
            None,
            "file:///path/to/document.txt",
            "not_a_base64_string",
            "data:text/plain;charset=utf-8,Hello World"
        ]
        
        for ref in valid_image_refs:
            assert mock_reranker._is_image_reference(ref), f"Should be valid image ref: {ref}"
        
        for ref in invalid_refs:
            if ref is not None:
                assert not mock_reranker._is_image_reference(ref), f"Should be invalid image ref: {ref}"
    
    def test_normalize_document_preservation(self, mock_reranker, mixed_document_samples):
        """Test preservación de metadata durante normalización"""
        for original_doc in mixed_document_samples:
            normalized = mock_reranker._normalize_document(original_doc)
            
            # Verificar estructura válida
            assert isinstance(normalized, dict)
            assert "text" in normalized or "image" in normalized
            
            # Si hay texto, verificar que se preserva
            if hasattr(original_doc, 'page_content'):
                if original_doc.page_content:
                    assert "text" in normalized
                    assert normalized["text"] == original_doc.page_content
            elif isinstance(original_doc, dict):
                text_content = (original_doc.get("page_content") or 
                              original_doc.get("content") or 
                              original_doc.get("text"))
                if text_content:
                    assert "text" in normalized
    
    def test_metrics_tracking(self, mock_reranker):
        """Test tracking de métricas durante detección de modalidad"""
        initial_metrics = mock_reranker.get_metrics()
        
        # Procesar varios documentos
        docs = [
            {"page_content": "texto simple"},  # text
            {"metadata": {"image": "file:///img.jpg"}},  # image  
            {"page_content": "texto", "metadata": {"thumbnail_uri": "/thumb.png"}},  # mixed
        ]
        
        for doc in docs:
            mock_reranker._detect_modality(doc)
        
        final_metrics = mock_reranker.get_metrics()
        
        # Verificar incremento de contadores
        assert final_metrics["modality_detection_calls"] > initial_metrics["modality_detection_calls"]
        assert final_metrics["text_only_docs"] >= 1
        assert final_metrics["image_only_docs"] >= 1  
        assert final_metrics["mixed_docs"] >= 1
    
    def test_performance_batch_processing(self, mock_reranker):
        """Test rendimiento con batches grandes de documentos"""
        # Crear batch de 100 documentos variados
        batch_docs = []
        for i in range(100):
            if i % 3 == 0:  # texto
                doc = {"page_content": f"Documento de texto {i}"}
            elif i % 3 == 1:  # imagen
                doc = {"metadata": {"image": f"file:///img_{i}.jpg"}}
            else:  # mixto
                doc = {
                    "page_content": f"Texto {i}",
                    "metadata": {"thumbnail_uri": f"/thumb_{i}.png"}
                }
            batch_docs.append(doc)
        
        # Medir tiempo de procesamiento
        start_time = time.time()
        
        for doc in batch_docs:
            mock_reranker._detect_modality(doc)
        
        processing_time = time.time() - start_time
        
        # Verificar rendimiento (debe ser < 100ms para 100 docs)
        assert processing_time < 0.1, f"Batch processing too slow: {processing_time:.3f}s"
        
        # Verificar métricas finales
        metrics = mock_reranker.get_metrics()
        assert metrics["modality_detection_calls"] >= 100
    
    def test_process_image_for_api_formatting(self, mock_reranker):
        """Test formateo de imágenes para API"""
        test_cases = [
            # Casos que no deben cambiar
            ("data:image/jpeg;base64,/9j/test", "data:image/jpeg;base64,/9j/test"),
            ("https://example.com/image.jpg", "https://example.com/image.jpg"),
            
            # Casos que deben transformarse
            ("/var/data/rag/images/chart.png", "file:///var/data/rag/images/chart.png"),
            ("file:///already/formatted.jpg", "file:///already/formatted.jpg"),
            
            # Base64 sin prefijo (debería añadir prefijo)
            ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAY=", "data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAY="),
        ]
        
        for input_ref, expected_output in test_cases:
            if mock_reranker._is_image_reference(input_ref):
                processed = mock_reranker._process_image_for_api(input_ref)
                assert processed == expected_output, f"Input: {input_ref}, Expected: {expected_output}, Got: {processed}"
    
    def test_multimodal_request_detection(self, mock_reranker):
        """Test detección de requests multimodales"""
        # Query con imagen
        query_with_image = {"image": "data:image/jpeg;base64,test"}
        text_docs = [{"text": "documento texto"}]
        assert mock_reranker._is_multimodal_request(query_with_image, text_docs)
        
        # Documentos con imagen
        text_query = "query de texto"
        image_docs = [{"text": "texto"}, {"image": "file:///img.jpg"}]
        assert mock_reranker._is_multimodal_request(text_query, image_docs)
        
        # Solo texto
        text_query = "query de texto"
        text_docs = [{"text": "documento texto"}]
        assert not mock_reranker._is_multimodal_request(text_query, text_docs)
    
    def test_extract_original_metadata(self, mock_reranker):
        """Test extracción completa de metadata original"""
        # Documento dict con metadata compleja
        doc_dict = {
            "page_content": "Contenido de prueba",
            "image_url": "direct_field.jpg",
            "metadata": {
                "modality": "mixed",
                "thumbnail_uri": "/thumb.png",
                "width": 800,
                "height": 600,
                "custom_field": "valor personalizado",
                "bbox": {"x0": 10, "y0": 20, "x1": 100, "y1": 200}
            }
        }
        
        extracted = mock_reranker._extract_original_metadata(doc_dict)
        
        # Verificar que se preserva toda la metadata
        assert "modality" in extracted
        assert "thumbnail_uri" in extracted
        assert "custom_field" in extracted
        assert "bbox" in extracted
        assert extracted["custom_field"] == "valor personalizado"
        
        # Verificar que se añaden campos directos
        assert "image_url" in extracted
        assert extracted["image_url"] == "direct_field.jpg"
        
        # Objeto mock
        mock_doc = Mock()
        mock_doc.metadata = {"source": "test", "page": 1}
        
        extracted_mock = mock_reranker._extract_original_metadata(mock_doc)
        assert "source" in extracted_mock
        assert extracted_mock["page"] == 1
    
    def test_modality_detection_consistency(self, mock_reranker):
        """Test consistencia en detección de modalidad múltiples llamadas"""
        test_doc = {
            "page_content": "Texto de ejemplo",
            "metadata": {
                "thumbnail_uri": "/var/data/rag/images/thumb.jpg",
                "modality": "mixed"
            }
        }
        
        # Múltiples llamadas deben dar mismo resultado
        results = []
        for _ in range(10):
            result = mock_reranker._detect_modality(test_doc)
            results.append(result)
        
        # Todos los resultados deben ser iguales
        assert all(r == results[0] for r in results), f"Inconsistent results: {results}"
        assert results[0] == "mixed"


if __name__ == "__main__":
    # Ejecutar tests con pytest si se llama directamente
    pytest.main([__file__, "-v"])