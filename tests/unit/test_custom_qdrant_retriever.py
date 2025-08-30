"""
Test suite for CustomQdrantRetriever multimodal functionality.

Tests comprehensive multimodal search capabilities including embedding generation,
mixed results handling, metadata preservation, and error scenarios.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import time
from typing import List, Dict, Any
from langchain_core.documents import Document

# Test fixtures and setup
class MockQdrantPoint:
    """Mock Qdrant point for testing"""
    def __init__(self, payload: Dict[str, Any], score: float = 0.8):
        self.payload = payload
        self.score = score

class MockQdrantResponse:
    """Mock Qdrant search response"""
    def __init__(self, points: List[MockQdrantPoint]):
        self.points = points

class MockEmbeddings:
    """Mock embeddings for testing"""
    def embed_query(self, text: str) -> List[float]:
        # Return mock 1024-dimensional vector for jina-embeddings-v4
        return [0.1] * 1024

@pytest.fixture
def mock_embeddings():
    """Mock jina-embeddings-v4 embeddings"""
    return MockEmbeddings()

@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client"""
    return Mock()

@pytest.fixture
def mock_text_result():
    """Mock text search result"""
    return MockQdrantPoint({
        "modality": "text",
        "doc_id": "doc_123",
        "page_number": 1,
        "source_uri": "/docs/example.pdf",
        "hash": "abc123",
        "embedding_model": "jina-embeddings-v4",
        "created_at": "2024-01-01T00:00:00",
        "page_content": "This is sample text content from the document.",
        "content_preview": "This is sample text...",
        "title": "Example Document",
        "author": "Test Author"
    }, score=0.95)

@pytest.fixture
def mock_image_result():
    """Mock image search result"""
    return MockQdrantPoint({
        "modality": "image",
        "doc_id": "doc_456",
        "page_number": 2,
        "source_uri": "/docs/visual_doc.pdf",
        "hash": "def456",
        "embedding_model": "jina-embeddings-v4",
        "created_at": "2024-01-01T00:00:00",
        "thumbnail_uri": "/thumbnails/img_001.jpg",
        "width": 800,
        "height": 600,
        "image_index": 0,
        "bbox": {"x0": 100, "y0": 150, "x1": 700, "y1": 550},
        "title": "Visual Document",
        "author": "Test Author"
    }, score=0.87)

class TestCustomQdrantRetriever:
    """Test suite for CustomQdrantRetriever multimodal functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        # Import here to avoid import issues
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        
        from app import CustomQdrantRetriever
        self.CustomQdrantRetriever = CustomQdrantRetriever
    
    def test_initialization(self, mock_qdrant_client, mock_embeddings):
        """Test CustomQdrantRetriever initialization with multimodal config"""
        retriever = self.CustomQdrantRetriever(
            client=mock_qdrant_client,
            collection_name="rag_multimodal",
            embeddings=mock_embeddings,
            k=15
        )
        
        assert retriever.client == mock_qdrant_client
        assert retriever.collection_name == "rag_multimodal"
        assert retriever.embeddings == mock_embeddings
        assert retriever.k == 15
    
    def test_query_embedding_generation(self, mock_qdrant_client, mock_embeddings):
        """Test query embedding generation using jina-embeddings-v4"""
        retriever = self.CustomQdrantRetriever(
            client=mock_qdrant_client,
            collection_name="rag_multimodal", 
            embeddings=mock_embeddings,
            k=10
        )
        
        # Mock Qdrant response
        mock_qdrant_client.query_points.return_value = MockQdrantResponse([])
        
        # Test different query types
        test_queries = [
            "What is machine learning?",  # Technical query
            "Show me the chart about sales",  # Descriptive query
            "AI",  # Short query
            "Explain the detailed methodology used in the comprehensive analysis of artificial intelligence algorithms and their applications in modern software development",  # Long query
        ]
        
        for query in test_queries:
            retriever.get_relevant_documents(query)
            
            # Verify embedding was called with correct query
            assert mock_embeddings.embed_query.called
            mock_embeddings.embed_query.assert_called_with(query)
            
            # Verify Qdrant was called with embedding
            mock_qdrant_client.query_points.assert_called_with(
                collection_name="rag_multimodal",
                query=[0.1] * 1024,  # Expected 1024-dimensional vector
                limit=10,
                with_payload=True
            )
    
    def test_mixed_search_results(self, mock_qdrant_client, mock_embeddings, mock_text_result, mock_image_result):
        """Test that mixed text and image results are handled correctly"""
        retriever = self.CustomQdrantRetriever(
            client=mock_qdrant_client,
            collection_name="rag_multimodal",
            embeddings=mock_embeddings,
            k=10
        )
        
        # Mock mixed results
        mixed_results = [mock_text_result, mock_image_result]
        mock_qdrant_client.query_points.return_value = MockQdrantResponse(mixed_results)
        
        documents = retriever.get_relevant_documents("test query")
        
        assert len(documents) == 2
        
        # Verify text document
        text_doc = documents[0]
        assert isinstance(text_doc, Document)
        assert text_doc.page_content == "This is sample text content from the document."
        assert text_doc.metadata['modality'] == 'text'
        assert text_doc.metadata['similarity_score'] == 0.95
        assert text_doc.metadata['doc_id'] == 'doc_123'
        assert text_doc.metadata['title'] == 'Example Document'
        
        # Verify image document
        image_doc = documents[1]
        assert isinstance(image_doc, Document)
        assert "Imagen 1 en página 2" in image_doc.page_content
        assert "800x600px" in image_doc.page_content
        assert "/thumbnails/img_001.jpg" in image_doc.page_content
        assert image_doc.metadata['modality'] == 'image'
        assert image_doc.metadata['similarity_score'] == 0.87
        assert image_doc.metadata['doc_id'] == 'doc_456'
        assert image_doc.metadata['thumbnail_uri'] == '/thumbnails/img_001.jpg'
        assert image_doc.metadata['width'] == 800
        assert image_doc.metadata['height'] == 600
    
    def test_metadata_preservation(self, mock_qdrant_client, mock_embeddings, mock_text_result):
        """Test that all required metadata fields are preserved"""
        retriever = self.CustomQdrantRetriever(
            client=mock_qdrant_client,
            collection_name="rag_multimodal",
            embeddings=mock_embeddings
        )
        
        mock_qdrant_client.query_points.return_value = MockQdrantResponse([mock_text_result])
        
        documents = retriever.get_relevant_documents("test query")
        doc = documents[0]
        
        # Required common fields
        assert 'modality' in doc.metadata
        assert 'doc_id' in doc.metadata
        assert 'page_number' in doc.metadata
        assert 'source_uri' in doc.metadata
        assert 'hash' in doc.metadata
        assert 'embedding_model' in doc.metadata
        assert 'created_at' in doc.metadata
        assert 'similarity_score' in doc.metadata  # Added for reranking
        
        # Optional fields when present
        assert 'title' in doc.metadata
        assert 'author' in doc.metadata
        
        # Verify page_content was extracted from payload
        assert 'page_content' not in doc.metadata  # Should be in page_content, not metadata
    
    def test_similarity_score_preservation(self, mock_qdrant_client, mock_embeddings):
        """Test that Qdrant similarity scores are preserved for reranking"""
        retriever = self.CustomQdrantRetriever(
            client=mock_qdrant_client,
            collection_name="rag_multimodal",
            embeddings=mock_embeddings
        )
        
        # Create results with different scores
        results = [
            MockQdrantPoint({"modality": "text", "page_content": "Content 1"}, score=0.95),
            MockQdrantPoint({"modality": "text", "page_content": "Content 2"}, score=0.80),
            MockQdrantPoint({"modality": "image", "thumbnail_uri": "/thumb1.jpg", "width": 100, "height": 100, "image_index": 0}, score=0.65),
        ]
        
        mock_qdrant_client.query_points.return_value = MockQdrantResponse(results)
        
        documents = retriever.get_relevant_documents("test query")
        
        # Verify scores are preserved in correct order
        assert documents[0].metadata['similarity_score'] == 0.95
        assert documents[1].metadata['similarity_score'] == 0.80
        assert documents[2].metadata['similarity_score'] == 0.65
    
    def test_k_parameter_handling(self, mock_qdrant_client, mock_embeddings):
        """Test that k parameter is handled correctly"""
        retriever = self.CustomQdrantRetriever(
            client=mock_qdrant_client,
            collection_name="rag_multimodal",
            embeddings=mock_embeddings,
            k=15  # Default k
        )
        
        mock_qdrant_client.query_points.return_value = MockQdrantResponse([])
        
        # Test with default k
        retriever.get_relevant_documents("test query")
        mock_qdrant_client.query_points.assert_called_with(
            collection_name="rag_multimodal",
            query=[0.1] * 1024,
            limit=15,  # Should use default k
            with_payload=True
        )
        
        # Test with custom k
        retriever.get_relevant_documents("test query", k=25)
        mock_qdrant_client.query_points.assert_called_with(
            collection_name="rag_multimodal",
            query=[0.1] * 1024,
            limit=25,  # Should use provided k
            with_payload=True
        )
    
    def test_empty_query_handling(self, mock_qdrant_client, mock_embeddings):
        """Test handling of edge case queries"""
        retriever = self.CustomQdrantRetriever(
            client=mock_qdrant_client,
            collection_name="rag_multimodal",
            embeddings=mock_embeddings
        )
        
        mock_qdrant_client.query_points.return_value = MockQdrantResponse([])
        
        # Test empty string
        documents = retriever.get_relevant_documents("")
        assert documents == []
        
        # Test whitespace only
        documents = retriever.get_relevant_documents("   ")
        assert documents == []
        
        # Test special characters
        documents = retriever.get_relevant_documents("¿Qué es IA?")
        assert isinstance(documents, list)
    
    @patch('app.logger')
    def test_qdrant_connection_error(self, mock_logger, mock_qdrant_client, mock_embeddings):
        """Test handling of Qdrant connection errors"""
        retriever = self.CustomQdrantRetriever(
            client=mock_qdrant_client,
            collection_name="rag_multimodal",
            embeddings=mock_embeddings
        )
        
        # Simulate connection error
        mock_qdrant_client.query_points.side_effect = ConnectionError("Cannot connect to Qdrant")
        
        with pytest.raises(ConnectionError):
            retriever.get_relevant_documents("test query")
    
    def test_embedding_generation_error(self, mock_qdrant_client):
        """Test handling of embedding generation errors"""
        # Mock embeddings that raises an error
        mock_embeddings = Mock()
        mock_embeddings.embed_query.side_effect = Exception("Embedding generation failed")
        
        retriever = self.CustomQdrantRetriever(
            client=mock_qdrant_client,
            collection_name="rag_multimodal",
            embeddings=mock_embeddings
        )
        
        with pytest.raises(Exception, match="Embedding generation failed"):
            retriever.get_relevant_documents("test query")
    
    def test_image_content_formatting(self, mock_qdrant_client, mock_embeddings):
        """Test proper formatting of image content"""
        retriever = self.CustomQdrantRetriever(
            client=mock_qdrant_client,
            collection_name="rag_multimodal",
            embeddings=mock_embeddings
        )
        
        # Test image with thumbnail
        image_with_thumb = MockQdrantPoint({
            "modality": "image",
            "page_number": 5,
            "thumbnail_uri": "/thumbs/chart_1.jpg",
            "width": 1200,
            "height": 800,
            "image_index": 2
        })
        
        # Test image without thumbnail
        image_without_thumb = MockQdrantPoint({
            "modality": "image", 
            "page_number": 3,
            "width": 600,
            "height": 400,
            "image_index": 0
        })
        
        mock_qdrant_client.query_points.return_value = MockQdrantResponse([
            image_with_thumb, image_without_thumb
        ])
        
        documents = retriever.get_relevant_documents("test query")
        
        # Verify content formatting with thumbnail
        assert "Imagen 3 en página 5" in documents[0].page_content
        assert "1200x800px" in documents[0].page_content
        assert "/thumbs/chart_1.jpg" in documents[0].page_content
        
        # Verify content formatting without thumbnail
        assert "Imagen 1 en página 3" in documents[1].page_content
        assert "600x400px" in documents[1].page_content
        assert "thumbnail:" not in documents[1].page_content
    
    def test_performance_benchmark(self, mock_qdrant_client, mock_embeddings, mock_text_result):
        """Test that searches complete within performance requirements (<2s)"""
        retriever = self.CustomQdrantRetriever(
            client=mock_qdrant_client,
            collection_name="rag_multimodal",
            embeddings=mock_embeddings
        )
        
        # Mock response with realistic delay
        def slow_response(*args, **kwargs):
            time.sleep(0.1)  # Simulate realistic query time
            return MockQdrantResponse([mock_text_result])
        
        mock_qdrant_client.query_points.side_effect = slow_response
        
        start_time = time.time()
        documents = retriever.get_relevant_documents("test query")
        elapsed = time.time() - start_time
        
        # Should complete in under 2 seconds
        assert elapsed < 2.0
        assert len(documents) == 1
    
    def test_large_result_set_handling(self, mock_qdrant_client, mock_embeddings):
        """Test handling of large result sets respects k limit"""
        retriever = self.CustomQdrantRetriever(
            client=mock_qdrant_client,
            collection_name="rag_multimodal",
            embeddings=mock_embeddings,
            k=5
        )
        
        # Create 10 mock results
        mock_results = [
            MockQdrantPoint({
                "modality": "text",
                "page_content": f"Content {i}",
                "doc_id": f"doc_{i}"
            }, score=0.9 - (i * 0.1))
            for i in range(10)
        ]
        
        mock_qdrant_client.query_points.return_value = MockQdrantResponse(mock_results)
        
        documents = retriever.get_relevant_documents("test query")
        
        # Should respect k=5 limit
        mock_qdrant_client.query_points.assert_called_with(
            collection_name="rag_multimodal",
            query=[0.1] * 1024,
            limit=5,
            with_payload=True
        )
    
    def test_multimodal_metadata_consistency(self, mock_qdrant_client, mock_embeddings, mock_text_result, mock_image_result):
        """Test that multimodal metadata is consistent and complete"""
        retriever = self.CustomQdrantRetriever(
            client=mock_qdrant_client,
            collection_name="rag_multimodal",
            embeddings=mock_embeddings
        )
        
        mock_qdrant_client.query_points.return_value = MockQdrantResponse([
            mock_text_result, mock_image_result
        ])
        
        documents = retriever.get_relevant_documents("test query")
        
        # Verify text document metadata
        text_doc = documents[0]
        assert text_doc.metadata['modality'] == 'text'
        assert text_doc.metadata['doc_id'] == 'doc_123'
        assert text_doc.metadata['page_number'] == 1
        assert text_doc.metadata['embedding_model'] == 'jina-embeddings-v4'
        
        # Verify image document metadata  
        image_doc = documents[1]
        assert image_doc.metadata['modality'] == 'image'
        assert image_doc.metadata['doc_id'] == 'doc_456'
        assert image_doc.metadata['page_number'] == 2
        assert image_doc.metadata['thumbnail_uri'] == '/thumbnails/img_001.jpg'
        assert image_doc.metadata['width'] == 800
        assert image_doc.metadata['height'] == 600
        assert image_doc.metadata['image_index'] == 0
        
        # Verify bbox is preserved as dict
        assert isinstance(image_doc.metadata['bbox'], dict)
        assert image_doc.metadata['bbox']['x0'] == 100
    
    def test_parallel_retriever_compatibility(self, mock_qdrant_client, mock_embeddings, mock_text_result):
        """Test compatibility with ParallelRetriever system"""
        from app import ParallelRetriever
        
        base_retriever = self.CustomQdrantRetriever(
            client=mock_qdrant_client,
            collection_name="rag_multimodal",
            embeddings=mock_embeddings
        )
        
        # Create parallel retriever
        parallel_retriever = ParallelRetriever(base_retriever, max_workers=2)
        
        mock_qdrant_client.query_points.return_value = MockQdrantResponse([mock_text_result])
        
        # Test single query (should use base retriever directly)
        documents = parallel_retriever.get_relevant_documents("test query")
        assert len(documents) == 1
        assert documents[0].metadata['modality'] == 'text'
        
        # Test batch queries
        queries = ["query 1", "query 2"]
        batch_results = parallel_retriever.batch_get_relevant_documents(queries)
        assert len(batch_results) == 2
        assert isinstance(batch_results[0], list)
        assert isinstance(batch_results[1], list)
    
    @patch('app.logger')
    def test_invalid_payload_handling(self, mock_logger, mock_qdrant_client, mock_embeddings):
        """Test handling of malformed or incomplete payloads"""
        retriever = self.CustomQdrantRetriever(
            client=mock_qdrant_client,
            collection_name="rag_multimodal",
            embeddings=mock_embeddings
        )
        
        # Mock result with missing required fields
        incomplete_result = MockQdrantPoint({
            "modality": "text"
            # Missing other required fields
        })
        
        mock_qdrant_client.query_points.return_value = MockQdrantResponse([incomplete_result])
        
        documents = retriever.get_relevant_documents("test query")
        
        # Should handle gracefully
        assert len(documents) == 1
        doc = documents[0]
        assert doc.metadata['modality'] == 'text'
        # Missing fields should be None or default values
        assert doc.page_content == ''  # Empty page_content
    
    def test_timeout_handling(self, mock_qdrant_client, mock_embeddings):
        """Test handling of search timeouts"""
        retriever = self.CustomQdrantRetriever(
            client=mock_qdrant_client,
            collection_name="rag_multimodal", 
            embeddings=mock_embeddings
        )
        
        # Simulate timeout
        mock_qdrant_client.query_points.side_effect = TimeoutError("Search timeout")
        
        with pytest.raises(TimeoutError):
            retriever.get_relevant_documents("test query")
    
    def test_special_characters_in_query(self, mock_qdrant_client, mock_embeddings, mock_text_result):
        """Test handling of queries with special characters"""
        retriever = self.CustomQdrantRetriever(
            client=mock_qdrant_client,
            collection_name="rag_multimodal",
            embeddings=mock_embeddings
        )
        
        mock_qdrant_client.query_points.return_value = MockQdrantResponse([mock_text_result])
        
        special_queries = [
            "¿Qué es inteligencia artificial?",  # Spanish characters
            "Machine Learning & AI",  # Ampersand
            "Cost: $100-$200",  # Currency symbols
            "Rate: 95%",  # Percentage
            "C++ programming",  # Plus signs
            "AI/ML algorithms",  # Forward slash
        ]
        
        for query in special_queries:
            documents = retriever.get_relevant_documents(query)
            assert isinstance(documents, list)
            mock_embeddings.embed_query.assert_called_with(query)


class TestMultimodalIntegration:
    """Integration tests for multimodal retrieval components"""
    
    def setup_method(self):
        """Setup test environment"""
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    @patch('app.EmbeddingFactory')
    def test_jina_embeddings_factory_integration(self, mock_factory):
        """Test integration with EmbeddingFactory for jina-embeddings-v4"""
        from app import embedding_model, embedding_provider, query_task_config
        
        # Verify correct configuration
        assert embedding_model == "jina-embeddings-v4"
        assert embedding_provider == "jina"
        assert query_task_config["task_type"] == "retrieval.query"
        assert query_task_config["late_chunking"] is False
    
    @patch('app.MULTIMODAL_COLLECTION_CONFIG')
    def test_multimodal_collection_config_usage(self, mock_config):
        """Test that multimodal collection config is used correctly"""
        mock_config.__getitem__.return_value = "test_multimodal_collection"
        
        # This would be tested through actual app initialization
        # but we can verify the config is imported and accessible
        from app import MULTIMODAL_COLLECTION_CONFIG
        assert MULTIMODAL_COLLECTION_CONFIG is not None


@pytest.mark.performance
class TestPerformanceRequirements:
    """Performance-specific tests for multimodal retrieval"""
    
    def test_embedding_generation_performance(self, mock_qdrant_client, mock_embeddings):
        """Test that embedding generation meets performance requirements"""
        from app import CustomQdrantRetriever
        
        retriever = CustomQdrantRetriever(
            client=mock_qdrant_client,
            collection_name="rag_multimodal",
            embeddings=mock_embeddings
        )
        
        mock_qdrant_client.query_points.return_value = MockQdrantResponse([])
        
        # Time multiple embedding generations
        start_time = time.time()
        for _ in range(10):
            retriever.get_relevant_documents("performance test query")
        elapsed = time.time() - start_time
        
        # Should average less than 0.2s per query (2s total for 10 queries)
        avg_time = elapsed / 10
        assert avg_time < 0.2
    
    def test_large_metadata_handling(self, mock_qdrant_client, mock_embeddings):
        """Test handling of documents with large metadata"""
        from app import CustomQdrantRetriever
        
        retriever = CustomQdrantRetriever(
            client=mock_qdrant_client,
            collection_name="rag_multimodal",
            embeddings=mock_embeddings
        )
        
        # Create result with large metadata
        large_metadata = {
            "modality": "text",
            "page_content": "x" * 5000,  # Large content
            "doc_id": "large_doc",
            "title": "Very long title " * 100,  # Large title
            "author": "Author with very long name " * 50,
            "custom_field": "z" * 2000  # Additional large field
        }
        
        large_result = MockQdrantPoint(large_metadata)
        mock_qdrant_client.query_points.return_value = MockQdrantResponse([large_result])
        
        documents = retriever.get_relevant_documents("test query")
        
        # Should handle without errors
        assert len(documents) == 1
        doc = documents[0]
        assert len(doc.page_content) == 5000
        assert doc.metadata['modality'] == 'text'


if __name__ == "__main__":
    # Run tests with coverage reporting
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--cov=app",
        "--cov-report=term-missing"
    ])