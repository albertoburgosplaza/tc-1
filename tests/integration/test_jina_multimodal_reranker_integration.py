"""
Tests de integración end-to-end para reranking multimodal con JinaReranker.

Tests exhaustivos que cubren:
- Reranking multimodal con jina-rerank-m0
- Manejo de URIs de imagen (locales, HTTP, base64)
- Procesamiento de candidatos mixtos texto/imagen
- Métricas de precisión de ranking (NDCG, MRR, top-k)
- Tests de rendimiento y escalabilidad
- Integración con pipeline RAG completo
- Casos edge y manejo de errores
"""

import pytest
import time
import json
import base64
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any
import tempfile
import os

# Import the JinaReranker class
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from jina_reranker import JinaReranker


class TestJinaMultimodalRerankerIntegration:
    """Test suite para integración de reranking multimodal"""
    
    @pytest.fixture
    def mock_api_response(self):
        """Response mock del API de Jina para tests"""
        return {
            "results": [
                {"index": 2, "relevance_score": 0.95, "score": 0.95},
                {"index": 0, "relevance_score": 0.88, "score": 0.88}, 
                {"index": 1, "relevance_score": 0.72, "score": 0.72},
                {"index": 3, "relevance_score": 0.61, "score": 0.61}
            ]
        }
    
    @pytest.fixture
    def mixed_candidate_set(self):
        """Conjunto realista de candidatos mixtos para testing"""
        return [
            # Documento solo texto
            {
                "page_content": "El análisis financiero muestra un crecimiento del 15% en ingresos durante el Q3 2023.",
                "metadata": {
                    "modality": "text",
                    "doc_id": "financial_report_2023",
                    "page_number": 5,
                    "source_uri": "file:///docs/financial_report.pdf"
                }
            },
            
            # Documento con imagen local
            {
                "page_content": "Gráfico de tendencias de ventas por región:",
                "metadata": {
                    "modality": "image", 
                    "thumbnail_uri": "/var/data/rag/images/sales_chart_q3.png",
                    "width": 800,
                    "height": 600,
                    "image_index": 1,
                    "doc_id": "sales_analysis",
                    "page_number": 12
                }
            },
            
            # Documento mixto con imagen base64
            {
                "page_content": "El siguiente diagrama ilustra el proceso de optimización:",
                "metadata": {
                    "modality": "mixed",
                    "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
                    "doc_id": "process_optimization",
                    "page_number": 8,
                    "bbox": {"x0": 50, "y0": 100, "x1": 400, "y1": 300}
                }
            },
            
            # Documento con imagen HTTP
            {
                "page_content": "",
                "metadata": {
                    "modality": "image",
                    "image_url": "https://example.com/charts/performance_metrics.jpg", 
                    "doc_id": "web_scraping_results",
                    "source": "web_analysis"
                }
            }
        ]
    
    @pytest.fixture  
    def reranker_with_mock_api(self, mock_api_response):
        """JinaReranker con API mock para testing sin llamadas reales"""
        with patch.dict(os.environ, {'JINA_API_KEY': 'test_key'}):
            reranker = JinaReranker()
            
            # Mock del método _make_request_with_retry
            with patch.object(reranker, '_make_request_with_retry') as mock_request:
                mock_request.return_value = mock_api_response["results"]
                yield reranker
    
    def test_end_to_end_multimodal_reranking(self, reranker_with_mock_api, mixed_candidate_set):
        """Test end-to-end de reranking multimodal completo"""
        query = "análisis de rendimiento financiero con gráficos"
        
        # Ejecutar reranking
        results, latency = reranker_with_mock_api.rerank(
            query, 
            mixed_candidate_set, 
            top_n=4, 
            return_documents=True
        )
        
        # Verificaciones básicas
        assert len(results) == 4
        assert latency > 0
        
        # Verificar estructura de resultados
        for result in results:
            assert "index" in result
            assert "relevance_score" in result
            assert result["index"] < len(mixed_candidate_set)
        
        # Verificar ordenamiento por relevancia
        scores = [r["relevance_score"] for r in results]
        assert scores == sorted(scores, reverse=True)
        
        # Verificar que se incluye contenido multimodal
        for result in results:
            if "text" in result:
                assert isinstance(result["text"], str)
            if "image" in result:
                assert isinstance(result["image"], str)
            if "original_metadata" in result:
                assert isinstance(result["original_metadata"], dict)
    
    def test_uri_handling_comprehensive(self, reranker_with_mock_api):
        """Test exhaustivo de manejo de diferentes tipos de URI"""
        uri_test_cases = [
            # URI local sin file://
            {
                "page_content": "Imagen local sin prefijo",
                "metadata": {"thumbnail_uri": "/var/data/rag/images/local_chart.png"}
            },
            
            # URI local con file://
            {
                "page_content": "Imagen local con prefijo", 
                "metadata": {"source_uri": "file:///var/data/rag/images/diagram.svg"}
            },
            
            # Data URI completa
            {
                "page_content": "Imagen embebida",
                "metadata": {
                    "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k="
                }
            },
            
            # URL HTTP
            {
                "page_content": "Imagen externa",
                "metadata": {"image_url": "https://cdn.example.com/analytics/q3_results.png"}
            },
            
            # Base64 sin prefijo data:
            {
                "page_content": "Base64 puro",
                "metadata": {"image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="}
            }
        ]
        
        results, _ = reranker_with_mock_api.rerank(
            "query test", 
            uri_test_cases, 
            return_documents=True
        )
        
        # Verificar que todas las imágenes se procesan correctamente
        for result in results:
            if "image" in result:
                image_ref = result["image"]
                # Debe ser una referencia válida procesada
                assert (image_ref.startswith('file://') or 
                       image_ref.startswith('http') or
                       image_ref.startswith('data:image/'))
    
    def test_base64_processing_robustness(self, reranker_with_mock_api):
        """Test robusto de procesamiento base64 con diferentes formatos"""
        base64_test_cases = [
            # PNG válido
            {
                "content": "PNG image",
                "metadata": {
                    "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
                }
            },
            
            # JPEG válido
            {
                "content": "JPEG image", 
                "metadata": {
                    "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD//2Q=="
                }
            },
            
            # WebP válido
            {
                "content": "WebP image",
                "metadata": {
                    "image": "data:image/webp;base64,UklGRh4AAABXRUJQVlA4TBEAAAAvAAAAAAfQ//73v/+BiOh/AAA="
                }
            },
            
            # Base64 malformado (debería ser manejado por fallback)
            {
                "content": "Corrupted base64",
                "metadata": {
                    "image": "data:image/png;base64,INVALID_BASE64!!!"
                }
            }
        ]
        
        # No debería lanzar excepciones incluso con datos corruptos
        results, _ = reranker_with_mock_api.rerank(
            "test query", 
            base64_test_cases
        )
        
        assert len(results) > 0  # Debería devolver resultados incluso con algunos errores
    
    def test_ranking_precision_metrics(self, reranker_with_mock_api):
        """Test métricas de precisión de ranking (NDCG, MRR, top-k)"""
        # Documentos con relevancia conocida para calcular métricas
        relevance_ground_truth = [0.9, 0.3, 0.8, 0.1, 0.7]  # Relevancia real esperada
        
        candidates = []
        for i, relevance in enumerate(relevance_ground_truth):
            candidates.append({
                "page_content": f"Document {i} with relevance {relevance}",
                "metadata": {
                    "doc_id": f"doc_{i}",
                    "ground_truth_relevance": relevance
                }
            })
        
        # Mock response que simula mejora de ranking
        improved_ranking = [0, 2, 4, 1, 3]  # Ordenamiento mejorado por índice
        mock_response = []
        for rank, idx in enumerate(improved_ranking):
            score = 0.95 - (rank * 0.1)  # Scores descendentes
            mock_response.append({
                "index": idx,
                "relevance_score": score,
                "score": score
            })
        
        with patch.object(reranker_with_mock_api, '_make_request_with_retry') as mock_request:
            mock_request.return_value = mock_response
            
            results, _ = reranker_with_mock_api.rerank(
                "test query for precision",
                candidates,
                top_n=5
            )
        
        # Calcular NDCG@3 simplificado
        top_3_indices = [r["index"] for r in results[:3]]
        top_3_relevances = [relevance_ground_truth[idx] for idx in top_3_indices]
        
        # El reranking debería mejorar el orden (doc más relevante primero)
        assert top_3_relevances[0] >= 0.7  # Documento altamente relevante en top 1
        
        # Verificar MRR (Mean Reciprocal Rank) para documentos relevantes (>0.7)
        relevant_docs = [i for i, rel in enumerate(relevance_ground_truth) if rel > 0.7]
        first_relevant_position = None
        
        for rank, result in enumerate(results):
            if result["index"] in relevant_docs:
                first_relevant_position = rank + 1  # 1-indexed
                break
        
        assert first_relevant_position is not None
        assert first_relevant_position <= 3  # Documento relevante en top-3
    
    def test_performance_under_load(self, reranker_with_mock_api):
        """Test rendimiento con datasets realistas de gran tamaño"""
        # Crear dataset grande con 50+ candidatos mixtos
        large_dataset = []
        for i in range(60):
            if i % 3 == 0:  # Texto puro
                doc = {
                    "page_content": f"Documento de texto número {i} con contenido relevante para análisis.",
                    "metadata": {
                        "modality": "text",
                        "doc_id": f"text_doc_{i}",
                        "page_number": i % 10
                    }
                }
            elif i % 3 == 1:  # Solo imagen
                doc = {
                    "page_content": "",
                    "metadata": {
                        "modality": "image", 
                        "thumbnail_uri": f"/var/data/rag/images/chart_{i}.png",
                        "width": 800 + (i * 10),
                        "height": 600 + (i * 5),
                        "doc_id": f"img_doc_{i}"
                    }
                }
            else:  # Mixto
                doc = {
                    "page_content": f"Documento mixto {i} con texto e imagen incorporada.",
                    "metadata": {
                        "modality": "mixed",
                        "image": f"data:image/png;base64,mock_base64_data_{i}",
                        "doc_id": f"mixed_doc_{i}",
                        "page_number": i % 15
                    }
                }
            large_dataset.append(doc)
        
        # Mock response para dataset grande
        mock_large_response = []
        for i in range(20):  # Top 20 resultados
            mock_large_response.append({
                "index": i,
                "relevance_score": 0.95 - (i * 0.02),
                "score": 0.95 - (i * 0.02)
            })
        
        with patch.object(reranker_with_mock_api, '_make_request_with_retry') as mock_request:
            mock_request.return_value = mock_large_response
            
            # Medir tiempo de procesamiento
            start_time = time.time()
            
            results, latency = reranker_with_mock_api.rerank(
                "consulta compleja para dataset grande",
                large_dataset,
                top_n=20
            )
            
            total_time = time.time() - start_time
        
        # Verificaciones de rendimiento
        assert len(results) == 20
        assert total_time < 5.0  # Objetivo: <5s para 60 candidatos -> 20 resultados
        assert latency > 0
        
        # Verificar métricas del reranker
        metrics = reranker_with_mock_api.get_metrics()
        assert metrics["total_documents_processed"] >= 60
        assert metrics["multimodal_docs"] > 0
    
    def test_edge_cases_comprehensive(self, reranker_with_mock_api):
        """Test casos edge comprehensive incluyendo errores de timeout"""
        edge_cases = [
            # Query sin contexto visual
            ("query puramente textual", [
                {"page_content": "Solo texto sin imágenes"}
            ]),
            
            # Candidatos solo texto
            ("query con imagen", [
                {"page_content": "Documento 1 de texto"},
                {"page_content": "Documento 2 de texto"} 
            ]),
            
            # Candidatos solo imagen
            ("query textual", [
                {"metadata": {"image": "file:///img1.jpg"}},
                {"metadata": {"thumbnail_uri": "/img2.png"}}
            ]),
            
            # Lista vacía
            ("query cualquiera", []),
            
            # Documento con metadata corrupta
            ("test query", [
                {"page_content": "Texto válido", "metadata": None},
                {"page_content": "Otro texto", "metadata": {"invalid": "structure"}}
            ])
        ]
        
        for query, candidates in edge_cases:
            try:
                results, latency = reranker_with_mock_api.rerank(query, candidates)
                
                # Casos válidos deberían devolver resultados
                if candidates:
                    assert isinstance(results, list)
                    assert isinstance(latency, float)
                    assert latency >= 0
                else:
                    # Lista vacía debería devolver lista vacía
                    assert results == []
                    assert latency == 0.0
                    
            except Exception as e:
                pytest.fail(f"Edge case failed: query='{query}', candidates={len(candidates)} docs - Error: {e}")
    
    def test_timeout_handling(self):
        """Test manejo de timeouts para operaciones multimodales"""
        with patch.dict(os.environ, {'JINA_API_KEY': 'test_key', 'MULTIMODAL_REQUEST_TIMEOUT': '5'}):
            reranker = JinaReranker(timeout=2, multimodal_timeout=5)
            
            # Verificar configuración de timeouts
            assert reranker.timeout == 2
            assert reranker.multimodal_timeout == 5
            
            # Mock documentos multimodales
            multimodal_docs = [
                {"text": "texto", "image": "file:///img.jpg"}
            ]
            text_only_docs = [
                {"text": "solo texto"}
            ]
            
            # Verificar detección de requests multimodales
            assert reranker._is_multimodal_request("query", multimodal_docs)
            assert not reranker._is_multimodal_request("query", text_only_docs)
    
    def test_fallback_mechanism_integration(self, reranker_with_mock_api):
        """Test integración completa del sistema de fallback"""
        problematic_docs = [
            {
                "page_content": "Texto válido",
                "metadata": {"image": "data:image/jpeg;base64,CORRUPTED_DATA"}
            },
            {
                "page_content": "Otro texto válido",
                "metadata": {"thumbnail_uri": "/nonexistent/image.jpg"}
            }
        ]
        
        # Simular fallo de API y activar fallback
        with patch.object(reranker_with_mock_api, '_make_request_with_retry') as mock_request:
            mock_request.side_effect = Exception("API timeout error")
            
            # El fallback debería manejar el error graciosamente
            results, latency = reranker_with_mock_api.rerank(
                "test query",
                problematic_docs,
                top_n=2
            )
            
            # Verificar que el fallback funcionó
            assert len(results) == 2
            assert all("fallback" in result for result in results)
            assert latency >= 0
            
            # Verificar métricas de fallback
            metrics = reranker_with_mock_api.get_metrics()
            assert metrics["fallback_count"] > 0
            assert metrics["error_count"] > 0
    
    def test_doc_objects_reranking_integration(self, reranker_with_mock_api):
        """Test integración de reranking con objetos Document (LangChain style)"""
        # Mock de objetos Document
        mock_docs = [
            Mock(
                page_content="Análisis de rendimiento Q3 2023",
                metadata={
                    "modality": "text",
                    "doc_id": "report_q3",
                    "rerank_score": None  # Se debería añadir
                }
            ),
            Mock(
                page_content="Gráfico de tendencias de mercado",
                metadata={
                    "modality": "mixed",
                    "thumbnail_uri": "/var/data/rag/images/market_trends.png",
                    "width": 1024,
                    "height": 768
                }
            ),
            Mock(
                page_content="",
                metadata={
                    "modality": "image",
                    "image_url": "https://charts.example.com/performance.svg",
                    "alt_text": "Performance metrics chart"
                }
            )
        ]
        
        # Ejecutar reranking de objetos
        reordered_docs, latency = reranker_with_mock_api.rerank_doc_objects(
            "query sobre rendimiento y gráficos",
            mock_docs,
            top_n=3
        )
        
        # Verificaciones
        assert len(reordered_docs) == 3
        assert latency > 0
        
        # Verificar que se añadió rerank_score a metadata
        for doc in reordered_docs:
            assert hasattr(doc, 'metadata')
            assert "rerank_score" in doc.metadata
            assert isinstance(doc.metadata["rerank_score"], (int, float))
            assert 0.0 <= doc.metadata["rerank_score"] <= 1.0
        
        # Verificar que metadata original se preserva
        for doc in reordered_docs:
            original_fields = ["modality", "doc_id", "thumbnail_uri", "width", "height", "image_url", "alt_text"]
            for field in original_fields:
                if field in doc.metadata and field != "rerank_score":
                    # Campo original debería estar preservado
                    assert doc.metadata[field] is not None
    
    def test_metrics_tracking_comprehensive(self, reranker_with_mock_api):
        """Test exhaustivo de tracking de métricas durante integración"""
        initial_metrics = reranker_with_mock_api.get_metrics()
        
        # Dataset mixto para generar diferentes tipos de métricas
        mixed_docs = [
            {"page_content": "Texto 1"},  # text_only
            {"metadata": {"image": "file:///img1.jpg"}},  # image_only  
            {"page_content": "Texto 2", "metadata": {"thumbnail_uri": "/img2.png"}},  # mixed
            {"page_content": "Texto 3"},  # text_only
        ]
        
        # Query multimodal
        multimodal_query = {"image": "data:image/png;base64,test"}
        
        # Ejecutar reranking
        results, latency = reranker_with_mock_api.rerank(
            multimodal_query,
            mixed_docs,
            top_n=4
        )
        
        # Verificar métricas finales
        final_metrics = reranker_with_mock_api.get_metrics()
        
        # Verificaciones de incrementos
        assert final_metrics["total_requests"] > initial_metrics["total_requests"]
        assert final_metrics["total_documents_processed"] >= 4
        assert final_metrics["multimodal_queries"] > initial_metrics["multimodal_queries"]
        assert final_metrics["multimodal_docs"] > initial_metrics["multimodal_docs"] 
        assert final_metrics["text_only_docs"] >= 2
        assert final_metrics["image_only_docs"] >= 1
        assert final_metrics["mixed_docs"] >= 1
        assert final_metrics["modality_detection_calls"] >= 4
        
        # Métricas calculadas
        assert final_metrics["avg_latency_ms"] > 0
        assert 0 <= final_metrics["error_rate"] <= 1


if __name__ == "__main__":
    # Ejecutar tests con pytest si se llama directamente
    pytest.main([__file__, "-v", "--tb=short"])