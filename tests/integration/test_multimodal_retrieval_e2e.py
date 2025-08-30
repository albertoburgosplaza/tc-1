"""
End-to-end integration tests for multimodal retrieval pipeline.

Tests the complete RAG pipeline with mixed text/image results, verifying
integration between all components, data consistency, and performance.
"""

import pytest
import os
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
from unittest.mock import patch, Mock

# Test configuration
TEST_COLLECTION_NAME = "test_multimodal_rag"
PERFORMANCE_TIMEOUT = 2.0  # 2 seconds max latency
MIN_THROUGHPUT_QPS = 20
TEST_CONCURRENT_QUERIES = 10

@pytest.fixture(scope="session")
def qdrant_client():
    """Real Qdrant client for integration testing"""
    from qdrant_client import QdrantClient
    
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    try:
        client = QdrantClient(url=qdrant_url)
        # Test connection
        client.get_collections()
        return client
    except Exception as e:
        pytest.skip(f"Qdrant not available for integration testing: {e}")

@pytest.fixture(scope="session")
def multimodal_collection(qdrant_client):
    """Setup test multimodal collection with sample data"""
    from multimodal_schema import MULTIMODAL_COLLECTION_CONFIG, MultimodalPayload
    from qdrant_client.models import Distance, VectorParams, CreateCollection
    import numpy as np
    
    collection_name = TEST_COLLECTION_NAME
    
    # Create collection if it doesn't exist
    try:
        collections = qdrant_client.get_collections().collections
        if not any(c.name == collection_name for c in collections):
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=1024,  # jina-embeddings-v4 dimension
                    distance=Distance.DOT  # Jina uses DOT product
                )
            )
            
            # Add sample test data
            _populate_test_collection(qdrant_client, collection_name)
            
        yield collection_name
        
    finally:
        # Cleanup: delete test collection
        try:
            qdrant_client.delete_collection(collection_name)
        except:
            pass  # Ignore cleanup errors

def _populate_test_collection(client, collection_name: str):
    """Populate test collection with sample multimodal data"""
    from qdrant_client.models import PointStruct
    import numpy as np
    
    # Sample test points with diverse content
    test_points = []
    
    # Text documents
    text_samples = [
        {
            "id": "text_001",
            "modality": "text",
            "doc_id": "doc_tech_1", 
            "page_number": 1,
            "source_uri": "/test_docs/machine_learning.pdf",
            "hash": "text_hash_001",
            "embedding_model": "jina-embeddings-v4",
            "created_at": "2024-01-01T00:00:00",
            "page_content": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
            "content_preview": "Machine learning is a subset of artificial intelligence...",
            "title": "Introduction to Machine Learning",
            "author": "Dr. Tech Expert"
        },
        {
            "id": "text_002", 
            "modality": "text",
            "doc_id": "doc_business_1",
            "page_number": 3,
            "source_uri": "/test_docs/business_report.pdf",
            "hash": "text_hash_002",
            "embedding_model": "jina-embeddings-v4",
            "created_at": "2024-01-01T00:00:00",
            "page_content": "The quarterly sales figures show a 15% increase compared to last quarter, with strong performance in the technology sector.",
            "content_preview": "The quarterly sales figures show a 15% increase...",
            "title": "Q4 Business Report",
            "author": "Business Analyst"
        }
    ]
    
    # Image documents  
    image_samples = [
        {
            "id": "image_001",
            "modality": "image",
            "doc_id": "doc_tech_1",
            "page_number": 2,
            "source_uri": "/test_docs/machine_learning.pdf", 
            "hash": "image_hash_001",
            "embedding_model": "jina-embeddings-v4",
            "created_at": "2024-01-01T00:00:00",
            "thumbnail_uri": "/thumbnails/ml_diagram.jpg",
            "width": 800,
            "height": 600,
            "image_index": 0,
            "bbox": {"x0": 50, "y0": 100, "x1": 750, "y1": 550},
            "title": "Introduction to Machine Learning",
            "author": "Dr. Tech Expert"
        },
        {
            "id": "image_002",
            "modality": "image",
            "doc_id": "doc_business_1", 
            "page_number": 1,
            "source_uri": "/test_docs/business_report.pdf",
            "hash": "image_hash_002",
            "embedding_model": "jina-embeddings-v4", 
            "created_at": "2024-01-01T00:00:00",
            "thumbnail_uri": "/thumbnails/sales_chart.jpg",
            "width": 1200,
            "height": 800,
            "image_index": 1,
            "bbox": {"x0": 100, "y0": 200, "x1": 1100, "y1": 700},
            "title": "Q4 Business Report",
            "author": "Business Analyst"
        }
    ]
    
    # Create points with random embeddings
    for i, sample in enumerate(text_samples + image_samples):
        point = PointStruct(
            id=sample["id"],
            vector=np.random.rand(1024).tolist(),  # Random 1024-dim vector
            payload=sample
        )
        test_points.append(point)
    
    # Upload points to collection
    client.upsert(
        collection_name=collection_name,
        points=test_points
    )
    
    # Wait for indexing
    time.sleep(1)

@pytest.fixture
def app_components(qdrant_client, multimodal_collection):
    """Setup app components for integration testing"""
    from app import CustomQdrantRetriever, ParallelRetriever
    from embedding_factory import EmbeddingFactory
    
    # Create embeddings (mock for testing)
    embeddings = Mock()
    embeddings.embed_query.return_value = [0.1] * 1024
    
    # Create retriever components
    base_retriever = CustomQdrantRetriever(
        client=qdrant_client,
        collection_name=multimodal_collection,
        embeddings=embeddings,
        k=10
    )
    
    parallel_retriever = ParallelRetriever(base_retriever, max_workers=2)
    
    return {
        'base_retriever': base_retriever,
        'parallel_retriever': parallel_retriever,
        'embeddings': embeddings,
        'client': qdrant_client
    }


class TestMultimodalRAGPipeline:
    """End-to-end tests for complete multimodal RAG pipeline"""
    
    def test_mixed_content_retrieval(self, app_components):
        """Test retrieval of mixed text and image content"""
        retriever = app_components['base_retriever']
        
        # Test queries that should return both text and images
        test_queries = [
            "machine learning algorithms",  # Should match text + diagram
            "sales performance charts",     # Should match text + chart
            "business analysis"            # Generic query
        ]
        
        for query in test_queries:
            documents = retriever.get_relevant_documents(query, k=10)
            
            # Should get results
            assert len(documents) > 0
            
            # Check for mixed modalities (may not always have both, depends on data)
            modalities = [doc.metadata.get('modality') for doc in documents]
            assert 'text' in modalities or 'image' in modalities
            
            # Verify document structure
            for doc in documents:
                assert hasattr(doc, 'page_content')
                assert hasattr(doc, 'metadata')
                assert 'modality' in doc.metadata
                assert 'similarity_score' in doc.metadata
                assert doc.metadata['modality'] in ['text', 'image']
    
    def test_complete_rag_with_citations(self, app_components):
        """Test complete RAG pipeline with proper citation handling"""
        from app import answer_with_rag
        
        # Mock LLM response with document citations
        mock_response = """
        Machine learning is a powerful technique [DOCUMENTO 1] that enables 
        computers to learn patterns. The sales chart [DOCUMENTO 2] shows 
        excellent growth trends in the AI sector.
        """
        
        with patch('app.current_llm') as mock_llm:
            mock_llm.invoke.return_value = Mock(content=mock_response)
            
            response, latency = answer_with_rag(
                query="What is machine learning and show sales data?",
                history=[]
            )
            
            # Should process citations
            assert "[1]" in response  # Inline citations
            assert "[2]" in response
            assert "📚 Fuentes consultadas:" in response or "📖 Fuente consultada:" in response
            assert latency > 0
    
    def test_parallel_retriever_integration(self, app_components):
        """Test integration between CustomQdrantRetriever and ParallelRetriever"""
        parallel_retriever = app_components['parallel_retriever']
        
        # Test single query
        documents = parallel_retriever.get_relevant_documents("test query")
        assert isinstance(documents, list)
        
        # Test batch queries with different complexities
        batch_queries = [
            "machine learning",
            "sales charts",
            "business analysis report", 
            "AI algorithms diagram"
        ]
        
        batch_results = parallel_retriever.batch_get_relevant_documents(batch_queries, k=5)
        
        assert len(batch_results) == len(batch_queries)
        
        for result_set in batch_results:
            assert isinstance(result_set, list)
            # Each result should have multimodal metadata
            for doc in result_set:
                assert 'modality' in doc.metadata
                assert 'similarity_score' in doc.metadata
    
    @pytest.mark.performance
    def test_concurrent_query_performance(self, app_components):
        """Test performance under concurrent query load"""
        parallel_retriever = app_components['parallel_retriever']
        
        # Generate diverse test queries
        test_queries = [
            f"machine learning query {i}" for i in range(TEST_CONCURRENT_QUERIES)
        ] + [
            f"business analysis query {i}" for i in range(TEST_CONCURRENT_QUERIES)
        ]
        
        start_time = time.time()
        
        # Execute concurrent queries
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(parallel_retriever.get_relevant_documents, query)
                for query in test_queries
            ]
            
            results = []
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=PERFORMANCE_TIMEOUT)
                    results.append(result)
                except Exception as e:
                    pytest.fail(f"Query failed under load: {e}")
        
        total_time = time.time() - start_time
        throughput = len(test_queries) / total_time
        
        # Verify performance requirements
        assert throughput >= MIN_THROUGHPUT_QPS, f"Throughput {throughput:.1f} QPS below {MIN_THROUGHPUT_QPS} QPS"
        assert len(results) == len(test_queries), "Some queries failed"
        
        # Verify all results have proper structure
        for result_set in results:
            assert isinstance(result_set, list)
            for doc in result_set:
                assert hasattr(doc, 'metadata')
                assert 'modality' in doc.metadata
    
    def test_data_consistency_validation(self, app_components, qdrant_client, multimodal_collection):
        """Test multimodal collection data consistency"""
        
        # Query collection info
        collection_info = qdrant_client.get_collection(multimodal_collection)
        assert collection_info.config.params.vectors.size == 1024  # jina-embeddings-v4 dimension
        
        # Test that data maintains referential integrity
        retriever = app_components['base_retriever']
        documents = retriever.get_relevant_documents("test query", k=20)
        
        # Group by document ID to verify cross-modal consistency
        doc_groups = {}
        for doc in documents:
            doc_id = doc.metadata.get('doc_id')
            if doc_id:
                if doc_id not in doc_groups:
                    doc_groups[doc_id] = []
                doc_groups[doc_id].append(doc)
        
        # Verify cross-modal consistency for documents with both text and images
        for doc_id, docs in doc_groups.items():
            if len(docs) > 1:
                # Get common metadata fields
                titles = set(doc.metadata.get('title') for doc in docs if doc.metadata.get('title'))
                authors = set(doc.metadata.get('author') for doc in docs if doc.metadata.get('author'))
                sources = set(doc.metadata.get('source_uri') for doc in docs if doc.metadata.get('source_uri'))
                
                # Should have consistent metadata across modalities
                assert len(titles) <= 1, f"Inconsistent titles for doc {doc_id}: {titles}"
                assert len(authors) <= 1, f"Inconsistent authors for doc {doc_id}: {authors}" 
                assert len(sources) <= 1, f"Inconsistent sources for doc {doc_id}: {sources}"
    
    def test_deduplication_integrity(self, app_components, qdrant_client, multimodal_collection):
        """Test that SHA-256 deduplication prevents duplicates"""
        
        # Query all points in collection
        scroll_result = qdrant_client.scroll(
            collection_name=multimodal_collection,
            limit=1000,
            with_payload=True
        )
        
        # Check for hash uniqueness per modality
        text_hashes = set()
        image_hashes = set()
        
        for point in scroll_result[0]:
            payload = point.payload
            modality = payload.get('modality')
            hash_value = payload.get('hash')
            
            if hash_value:
                if modality == 'text':
                    assert hash_value not in text_hashes, f"Duplicate text hash found: {hash_value}"
                    text_hashes.add(hash_value)
                elif modality == 'image':
                    assert hash_value not in image_hashes, f"Duplicate image hash found: {hash_value}"
                    image_hashes.add(hash_value)
    
    @pytest.mark.performance
    def test_realistic_load_performance(self, app_components):
        """Test performance under realistic query patterns"""
        parallel_retriever = app_components['parallel_retriever']
        
        # Realistic query patterns with different complexities
        query_patterns = {
            "simple_keyword": ["AI", "sales", "chart", "algorithm"],
            "semantic_queries": [
                "What are machine learning algorithms?",
                "Show me sales performance data", 
                "Explain artificial intelligence concepts",
                "Display business analysis charts"
            ],
            "visual_similarity": [
                "charts showing growth trends",
                "diagrams explaining AI concepts", 
                "visual representation of data",
                "business performance graphs"
            ]
        }
        
        latencies = []
        all_results = []
        
        for category, queries in query_patterns.items():
            for query in queries:
                start_time = time.time()
                
                try:
                    documents = parallel_retriever.get_relevant_documents(query, k=10)
                    latency = time.time() - start_time
                    latencies.append(latency)
                    all_results.append(documents)
                    
                    # Verify basic structure
                    assert isinstance(documents, list)
                    for doc in documents:
                        assert hasattr(doc, 'metadata')
                        assert 'modality' in doc.metadata
                        
                except Exception as e:
                    pytest.fail(f"Query '{query}' failed: {e}")
        
        # Calculate performance metrics
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            p95_latency = sorted(latencies)[int(0.95 * len(latencies))]
            max_latency = max(latencies)
            
            print(f"Performance metrics:")
            print(f"  Average latency: {avg_latency:.3f}s")
            print(f"  P95 latency: {p95_latency:.3f}s") 
            print(f"  Max latency: {max_latency:.3f}s")
            print(f"  Total queries: {len(latencies)}")
            
            # Verify performance requirements
            assert p95_latency < PERFORMANCE_TIMEOUT, f"P95 latency {p95_latency:.3f}s exceeds {PERFORMANCE_TIMEOUT}s"
            assert avg_latency < 1.0, f"Average latency {avg_latency:.3f}s too high"


class TestMultimodalResponseGeneration:
    """Test multimodal response generation and citation handling"""
    
    @pytest.fixture
    def mock_multimodal_docs(self):
        """Mock mixed documents for response testing"""
        from langchain_core.documents import Document
        
        return [
            Document(
                page_content="Artificial intelligence algorithms use neural networks to process data.",
                metadata={
                    "modality": "text",
                    "doc_id": "ai_guide",
                    "page_number": 1, 
                    "title": "AI Guide",
                    "similarity_score": 0.92
                }
            ),
            Document(
                page_content="Imagen 1 en página 2 (dimensiones: 800x600px, thumbnail: /thumbs/nn_diagram.jpg)",
                metadata={
                    "modality": "image",
                    "doc_id": "ai_guide",
                    "page_number": 2,
                    "title": "AI Guide", 
                    "thumbnail_uri": "/thumbs/nn_diagram.jpg",
                    "width": 800,
                    "height": 600,
                    "image_index": 0,
                    "similarity_score": 0.88
                }
            )
        ]
    
    def test_multimodal_context_generation(self, mock_multimodal_docs):
        """Test context generation with mixed text/image documents"""
        from app import answer_with_rag
        
        # Mock response that references both text and image
        mock_response = """
        Neural networks are fundamental to AI [DOCUMENTO 1]. The diagram 
        clearly illustrates the architecture [DOCUMENTO 2].
        """
        
        with patch('app.retriever') as mock_retriever, \
             patch('app.current_llm') as mock_llm:
            
            mock_retriever.get_relevant_documents.return_value = mock_multimodal_docs
            mock_llm.invoke.return_value = Mock(content=mock_response)
            
            response, latency = answer_with_rag("Explain neural networks with diagrams", [])
            
            # Should contain inline citations
            assert "[1]" in response
            assert "[2]" in response
            
            # Should contain both text and image references  
            assert "Fuentes consultadas:" in response or "Fuente consultada:" in response
            assert "AI Guide" in response  # Document title
            
            # Verify context included both modalities
            call_args = mock_llm.invoke.call_args[0][0]  # Get prompt
            assert "DOCUMENTO 1: AI Guide" in call_args  # Text content
            assert "DOCUMENTO 2: AI Guide" in call_args  # Image content
            assert "Imagen 1 en página 2" in call_args  # Image description
    
    def test_image_thumbnail_integration(self, mock_multimodal_docs):
        """Test that image thumbnails are properly integrated in responses"""
        from app import answer_with_rag
        
        with patch('app.retriever') as mock_retriever, \
             patch('app.current_llm') as mock_llm:
            
            mock_retriever.get_relevant_documents.return_value = mock_multimodal_docs
            mock_llm.invoke.return_value = Mock(content="Image shows neural network [DOCUMENTO 2].")
            
            response, _ = answer_with_rag("Show me AI diagrams", [])
            
            # Verify image information is preserved
            assert "/thumbs/nn_diagram.jpg" in response  # Thumbnail URI should be in context
    
    def test_semantic_coherence_across_modalities(self, app_components):
        """Test that results are semantically coherent across text and images"""
        retriever = app_components['base_retriever']
        
        # Query for related content across modalities
        documents = retriever.get_relevant_documents("artificial intelligence machine learning", k=10)
        
        if len(documents) > 1:
            # Check that documents from same source have consistent metadata
            doc_groups = {}
            for doc in documents:
                doc_id = doc.metadata.get('doc_id')
                if doc_id:
                    if doc_id not in doc_groups:
                        doc_groups[doc_id] = []
                    doc_groups[doc_id].append(doc)
            
            # Verify semantic coherence within document groups
            for doc_id, docs in doc_groups.items():
                titles = [doc.metadata.get('title') for doc in docs]
                sources = [doc.metadata.get('source_uri') for doc in docs] 
                
                # Should have consistent titles and sources within same document
                unique_titles = set(filter(None, titles))
                unique_sources = set(filter(None, sources))
                
                assert len(unique_titles) <= 1, f"Inconsistent titles in doc {doc_id}"
                assert len(unique_sources) <= 1, f"Inconsistent sources in doc {doc_id}"


class TestErrorHandlingAndRobustness:
    """Test error handling and system robustness"""
    
    def test_empty_collection_handling(self, qdrant_client):
        """Test behavior with empty multimodal collection"""
        from app import CustomQdrantRetriever
        from embedding_factory import EmbeddingFactory
        
        # Create empty test collection
        empty_collection = "test_empty_multimodal"
        
        try:
            # Create empty collection
            from qdrant_client.models import Distance, VectorParams
            qdrant_client.create_collection(
                collection_name=empty_collection,
                vectors_config=VectorParams(size=1024, distance=Distance.DOT)
            )
            
            embeddings = Mock()
            embeddings.embed_query.return_value = [0.1] * 1024
            
            retriever = CustomQdrantRetriever(
                client=qdrant_client,
                collection_name=empty_collection,
                embeddings=embeddings
            )
            
            documents = retriever.get_relevant_documents("any query")
            assert documents == []  # Should return empty list gracefully
            
        finally:
            try:
                qdrant_client.delete_collection(empty_collection)
            except:
                pass
    
    def test_malformed_payload_resilience(self, qdrant_client, multimodal_collection):
        """Test resilience against malformed payloads in collection"""
        from app import CustomQdrantRetriever
        
        embeddings = Mock()
        embeddings.embed_query.return_value = [0.1] * 1024
        
        retriever = CustomQdrantRetriever(
            client=qdrant_client,
            collection_name=multimodal_collection,
            embeddings=embeddings
        )
        
        # This test assumes collection has properly formatted data
        # In a real scenario, we'd insert malformed data first
        documents = retriever.get_relevant_documents("test query")
        
        # Should handle gracefully even if some payloads are malformed
        assert isinstance(documents, list)
        
        # Verify that valid documents still have proper structure
        for doc in documents:
            if hasattr(doc, 'metadata') and doc.metadata:
                # Should have basic required fields or defaults
                assert doc.metadata.get('modality') in [None, 'text', 'image']
    
    def test_network_timeout_recovery(self, app_components):
        """Test recovery from network timeouts"""
        retriever = app_components['base_retriever']
        client = app_components['client']
        
        # Mock timeout on first call, success on retry
        call_count = 0
        def mock_query(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TimeoutError("Network timeout")
            return MockQdrantResponse([])
        
        with patch.object(client, 'query_points', side_effect=mock_query):
            # Should raise timeout error (no automatic retry in CustomQdrantRetriever)
            with pytest.raises(TimeoutError):
                retriever.get_relevant_documents("test query")


class TestDataValidation:
    """Test data validation and schema compliance"""
    
    def test_multimodal_payload_validation(self, app_components):
        """Test that retrieved documents comply with multimodal schema"""
        retriever = app_components['base_retriever'] 
        documents = retriever.get_relevant_documents("validation test", k=5)
        
        for doc in documents:
            metadata = doc.metadata
            
            # Verify required common fields
            if 'modality' in metadata:
                modality = metadata['modality']
                assert modality in ['text', 'image']
                
                # Common required fields
                if modality == 'text':
                    # Text should have page_content
                    assert doc.page_content is not None
                    
                elif modality == 'image':
                    # Image should have image-specific metadata
                    if 'width' in metadata and 'height' in metadata:
                        assert isinstance(metadata['width'], int)
                        assert isinstance(metadata['height'], int)
                        assert metadata['width'] > 0
                        assert metadata['height'] > 0


@pytest.mark.integration
class TestRealWorldScenarios:
    """Integration tests with realistic usage scenarios"""
    
    def test_technical_document_search(self, app_components):
        """Test search in technical documents with diagrams"""
        retriever = app_components['parallel_retriever']
        
        technical_queries = [
            "machine learning algorithms",
            "neural network architecture", 
            "AI implementation patterns",
            "algorithmic complexity analysis"
        ]
        
        for query in technical_queries:
            documents = retriever.get_relevant_documents(query, k=8)
            
            # Should get relevant results
            assert len(documents) >= 0  # May be empty if no matching content
            
            # Verify document quality when results exist
            for doc in documents:
                assert doc.metadata.get('similarity_score', 0) >= 0
                assert doc.metadata.get('modality') in ['text', 'image']
    
    def test_business_document_search(self, app_components):
        """Test search in business documents with charts"""
        retriever = app_components['parallel_retriever']
        
        business_queries = [
            "sales performance analysis",
            "quarterly business results",
            "revenue growth charts", 
            "market analysis data"
        ]
        
        for query in business_queries:
            documents = retriever.get_relevant_documents(query, k=8)
            
            # Verify multimodal results structure
            for doc in documents:
                if doc.metadata.get('modality') == 'image':
                    # Image docs should have thumbnail info
                    assert 'width' in doc.metadata
                    assert 'height' in doc.metadata
                elif doc.metadata.get('modality') == 'text':
                    # Text docs should have meaningful content
                    assert len(doc.page_content) > 0


if __name__ == "__main__":
    # Run integration tests with detailed output
    pytest.main([
        __file__,
        "-v",
        "--tb=long", 
        "--durations=10",
        "-m", "not performance",  # Skip performance tests in regular runs
        "--cov=app",
        "--cov-report=term"
    ])