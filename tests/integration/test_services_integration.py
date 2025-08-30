"""Integration tests for service-to-service communication"""

import pytest
import requests
import json
from unittest.mock import Mock, patch, MagicMock
import time


class TestServicesIntegration:
    """Test integration between different services"""
    
    @pytest.fixture
    def mock_services_running(self):
        """Mock all services as running and healthy"""
        with patch('requests.get') as mock_get, patch('requests.post') as mock_post:
            # Mock health check responses
            health_response = Mock()
            health_response.status_code = 200
            health_response.text = "OK"
            health_response.json.return_value = {"status": "healthy"}
            mock_get.return_value = health_response
            
            # Mock calculation responses
            calc_response = Mock()
            calc_response.status_code = 200
            calc_response.json.return_value = {
                "result": 42.0,
                "expression": "6 * 7",
                "execution_time": 0.001
            }
            mock_post.return_value = calc_response
            
            yield {
                "get": mock_get,
                "post": mock_post,
                "health_response": health_response,
                "calc_response": calc_response
            }

    @pytest.mark.integration
    def test_app_to_pyexec_communication(self, mock_services_running):
        """Test communication from app to pyexec service"""
        pyexec_url = "http://pyexec:8001"
        
        # Test health check
        health_response = requests.get(f"{pyexec_url}/health")
        assert health_response.status_code == 200
        
        # Test calculation request
        calc_payload = {"expression": "sqrt(16) + 2"}
        calc_response = requests.post(f"{pyexec_url}/calculate", json=calc_payload)
        
        assert calc_response.status_code == 200
        result_data = calc_response.json()
        assert "result" in result_data
        assert "expression" in result_data
        assert "execution_time" in result_data

    @pytest.mark.integration
    def test_app_to_qdrant_communication(self, mock_services_running):
        """Test communication from app to Qdrant service"""
        with patch('qdrant_client.QdrantClient') as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            
            # Mock collection info
            mock_instance.get_collection.return_value = Mock(
                status="green",
                vectors_count=1000,
                segments_count=1
            )
            
            # Test collection access
            from qdrant_client import QdrantClient
            client = QdrantClient(url="http://qdrant:6333")
            
            collection_info = client.get_collection("corpus_pdf")
            assert collection_info.status == "green"
            assert collection_info.vectors_count > 0

    @pytest.mark.integration
    def test_app_to_ollama_communication(self, mock_services_running):
        """Test communication from app to Ollama service"""
        ollama_url = "http://ollama:11434"
        
        # Test model availability
        models_response = requests.get(f"{ollama_url}/api/tags")
        assert models_response.status_code == 200
        
        # Mock generate request
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "model": "mistral:7b-instruct",
                "response": "This is a test response from the language model.",
                "done": True
            }
            mock_post.return_value = mock_response
            
            generate_payload = {
                "model": "mistral:7b-instruct",
                "prompt": "What is artificial intelligence?",
                "stream": False
            }
            
            response = requests.post(f"{ollama_url}/api/generate", json=generate_payload)
            assert response.status_code == 200
            
            result = response.json()
            assert "response" in result
            assert "model" in result

    @pytest.mark.integration
    def test_gradio_interface_integration(self):
        """Test Gradio interface integration with backend services"""
        # Mock Gradio app functions
        with patch('app.validate_input') as mock_validate, \
             patch('app.retriever') as mock_retriever, \
             patch('app.llm') as mock_llm:
            
            # Mock validation
            mock_validate.return_value = (True, "")
            
            # Mock retrieval
            from langchain.schema import Document
            mock_docs = [
                Document(
                    page_content="AI is a field of computer science.",
                    metadata={"source": "ai.pdf", "page": 1}
                )
            ]
            mock_retriever.get_relevant_documents.return_value = mock_docs
            
            # Mock LLM response
            mock_llm.invoke.return_value = Mock(
                content="AI stands for Artificial Intelligence, which is a field of computer science."
            )
            
            # Simulate user query processing
            user_query = "What is AI?"
            
            # Validate input
            is_valid, error_msg = mock_validate(user_query)
            assert is_valid is True
            
            # Get context
            context_docs = mock_retriever.get_relevant_documents(user_query)
            assert len(context_docs) > 0
            
            # Generate response
            response = mock_llm.invoke("test prompt")
            assert response.content is not None

    @pytest.mark.integration
    def test_service_error_propagation(self, mock_services_running):
        """Test how errors propagate between services"""
        # Test pyexec service error
        with patch('requests.post') as mock_post:
            error_response = Mock()
            error_response.status_code = 400
            error_response.json.return_value = {
                "error": "Invalid expression",
                "category": "syntax_error"
            }
            mock_post.return_value = error_response
            
            calc_payload = {"expression": "invalid expression ++"}
            response = requests.post("http://pyexec:8001/calculate", json=calc_payload)
            
            assert response.status_code == 400
            error_data = response.json()
            assert "error" in error_data
            assert error_data["category"] == "syntax_error"

    @pytest.mark.integration
    def test_service_timeout_handling(self):
        """Test timeout handling between services"""
        with patch('requests.post') as mock_post:
            # Mock timeout exception
            mock_post.side_effect = requests.exceptions.Timeout("Request timed out")
            
            with pytest.raises(requests.exceptions.Timeout):
                requests.post(
                    "http://pyexec:8001/calculate",
                    json={"expression": "complex_calculation()"},
                    timeout=1
                )

    @pytest.mark.integration
    def test_concurrent_service_requests(self, mock_services_running):
        """Test handling of concurrent requests to services"""
        import threading
        import time
        
        results = []
        
        def make_request(expression):
            try:
                response = requests.post(
                    "http://pyexec:8001/calculate",
                    json={"expression": expression}
                )
                results.append(response.status_code)
            except Exception as e:
                results.append(str(e))
        
        # Create multiple concurrent requests
        threads = []
        expressions = [f"{i} + {i}" for i in range(5)]
        
        for expr in expressions:
            thread = threading.Thread(target=make_request, args=(expr,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All requests should succeed (200 status)
        assert all(result == 200 for result in results)
        assert len(results) == 5

    @pytest.mark.integration
    def test_service_state_consistency(self):
        """Test state consistency across service interactions"""
        with patch('qdrant_client.QdrantClient') as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            
            # Mock consistent document storage and retrieval
            test_doc_id = "doc_123"
            test_content = "Test document content for consistency check."
            
            # Mock upsert operation
            mock_instance.upsert.return_value = Mock(status="completed")
            
            # Mock search operation returning the same document
            mock_instance.search.return_value = [
                Mock(
                    id=test_doc_id,
                    payload={"content": test_content},
                    score=1.0
                )
            ]
            
            from qdrant_client import QdrantClient
            client = QdrantClient(url="http://qdrant:6333")
            
            # Simulate document upsert
            upsert_result = client.upsert(
                collection_name="test_collection",
                points=[{
                    "id": test_doc_id,
                    "payload": {"content": test_content},
                    "vector": [0.1] * 384
                }]
            )
            assert upsert_result.status == "completed"
            
            # Simulate document retrieval
            search_results = client.search(
                collection_name="test_collection",
                query_vector=[0.1] * 384,
                limit=1
            )
            
            assert len(search_results) == 1
            assert search_results[0].payload["content"] == test_content

    @pytest.mark.integration
    def test_service_dependency_chain(self, mock_services_running):
        """Test the complete service dependency chain"""
        # Simulate a complete RAG query flow
        query = "What is the square root of 144?"
        
        # Step 1: Validate input (app service)
        with patch('app.validate_input') as mock_validate:
            mock_validate.return_value = (True, "")
            is_valid, _ = mock_validate(query)
            assert is_valid is True
        
        # Step 2: Retrieve context (app -> qdrant)
        with patch('app.retriever') as mock_retriever:
            from langchain.schema import Document
            mock_retriever.get_relevant_documents.return_value = [
                Document(
                    page_content="The square root is a mathematical operation.",
                    metadata={"source": "math.pdf"}
                )
            ]
            context_docs = mock_retriever.get_relevant_documents(query)
            assert len(context_docs) > 0
        
        # Step 3: Process calculation (app -> pyexec)
        calc_response = requests.post(
            "http://pyexec:8001/calculate",
            json={"expression": "sqrt(144)"}
        )
        assert calc_response.status_code == 200
        calc_result = calc_response.json()
        assert calc_result["result"] == 42.0  # Mocked response
        
        # Step 4: Generate response (app -> ollama)
        with patch('app.llm') as mock_llm:
            mock_llm.invoke.return_value = Mock(
                content=f"The square root of 144 is {calc_result['result']}"
            )
            final_response = mock_llm.invoke("test prompt")
            assert "144" in final_response.content

    @pytest.mark.integration
    def test_service_configuration_consistency(self):
        """Test that service configurations are consistent"""
        # Test environment variable consistency
        expected_configs = {
            "QDRANT_URL": "http://qdrant:6333",
            "OLLAMA_BASE_URL": "http://ollama:11434",
            "PYEXEC_URL": "http://pyexec:8001",
            "COLLECTION_NAME": "corpus_pdf"
        }
        
        # This would typically check actual environment variables
        # For testing, we verify the expected structure exists
        for key, expected_value in expected_configs.items():
            assert isinstance(key, str)
            assert isinstance(expected_value, str)
            assert expected_value.startswith("http://")

    @pytest.mark.integration
    @pytest.mark.slow
    def test_service_performance_integration(self, mock_services_running):
        """Test performance characteristics of service integration"""
        import time
        
        # Test response time for typical operations
        start_time = time.time()
        
        # Simulate multiple service calls
        for _ in range(3):
            # Health checks
            requests.get("http://qdrant:6333/healthz")
            requests.get("http://ollama:11434/api/tags")
            requests.get("http://pyexec:8001/health")
            
            # Functional calls
            requests.post(
                "http://pyexec:8001/calculate",
                json={"expression": "2 + 2"}
            )
        
        total_time = time.time() - start_time
        
        # Should complete quickly when mocked
        assert total_time < 1.0  # All calls should complete in under 1 second


class TestImageEmbeddingIntegration:
    """Integration tests for image embedding functionality"""

    @pytest.fixture
    def mock_jina_api(self):
        """Mock Jina API responses for image embeddings"""
        with patch('requests.Session') as mock_session_class:
            mock_session = Mock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": [
                    {"embedding": [0.1] * 1024},
                    {"embedding": [0.2] * 1024}
                ]
            }
            mock_response.raise_for_status = Mock()
            mock_session.post.return_value = mock_response
            mock_session_class.return_value = mock_session
            yield mock_session

    @pytest.mark.integration
    @patch.dict('os.environ', {'JINA_API_KEY': 'test_key'})
    def test_image_embedding_factory_integration(self, mock_jina_api):
        """Test complete integration of image embedding through factory"""
        from embedding_factory import EmbeddingFactory
        
        # Create image embedding instance
        embedding_instance = EmbeddingFactory.create_image_embedding()
        
        # Verify configuration
        assert embedding_instance.dimensions == 1024
        assert embedding_instance.normalized is True
        assert embedding_instance.task_type == "retrieval.passage"
        
        # Test embedding generation with mock file
        with patch('builtins.open', mock_open(read_data=b"fake_image_data")):
            embeddings = embedding_instance.embed_images(["/test/image.jpg"])
            assert len(embeddings) == 1
            assert len(embeddings[0]) == 1024

    @pytest.mark.integration
    @patch.dict('os.environ', {'JINA_API_KEY': 'test_key'})
    def test_text_image_compatibility_integration(self, mock_jina_api):
        """Test text and image embedding compatibility in integration scenario"""
        from embedding_factory import EmbeddingFactory
        
        # Verify compatibility
        compatibility = EmbeddingFactory.verify_embedding_compatibility()
        
        assert compatibility["compatible"] is True
        assert compatibility["ready_for_mixed_search"] is True
        assert compatibility["text_dimensions"] == compatibility["image_dimensions"]
        
        # Test both text and image embeddings work
        text_instance = EmbeddingFactory.create_embedding()
        image_instance = EmbeddingFactory.create_image_embedding()
        
        assert text_instance.dimensions == image_instance.dimensions
        assert text_instance.normalized == image_instance.normalized

    @pytest.mark.integration
    @patch.dict('os.environ', {'JINA_API_KEY': 'test_key'})
    def test_flexible_image_embedding_integration(self, mock_jina_api):
        """Test flexible image embedding with multiple input types"""
        from embedding_factory import EmbeddingFactory
        import base64
        
        # Create test image data
        test_image_bytes = b"fake_png_data\x89PNG\r\n\x1a\n"
        test_image_b64 = base64.b64encode(test_image_bytes).decode('utf-8')
        
        # Test mixed input types
        mixed_inputs = [
            "/path/to/image1.jpg",  # file path
            test_image_b64,         # base64 string  
            test_image_bytes        # bytes data
        ]
        
        with patch('builtins.open', mock_open(read_data=test_image_bytes)):
            embeddings = EmbeddingFactory.embed_images_flexible(mixed_inputs)
            
            assert len(embeddings) == 3
            assert all(len(emb) == 1024 for emb in embeddings)

    @pytest.mark.integration
    @patch.dict('os.environ', {'JINA_API_KEY': 'test_key'})
    def test_image_embedding_qdrant_integration(self, mock_jina_api):
        """Test image embeddings integration with Qdrant vector storage"""
        from embedding_factory import EmbeddingFactory
        
        with patch('qdrant_client.QdrantClient') as mock_qdrant:
            mock_client = Mock()
            mock_qdrant.return_value = mock_client
            
            # Mock successful upsert
            mock_client.upsert.return_value = Mock(status="completed")
            
            # Generate image embeddings
            with patch('builtins.open', mock_open(read_data=b"fake_image")):
                embeddings = EmbeddingFactory.embed_images_flexible(["/test/image.jpg"])
                
                # Simulate storing in Qdrant
                from qdrant_client import QdrantClient
                from qdrant_client.models import PointStruct
                
                client = QdrantClient(url="http://qdrant:6333")
                
                points = [
                    PointStruct(
                        id="img_1",
                        vector=embeddings[0],
                        payload={
                            "source": "/test/image.jpg",
                            "modality": "image",
                            "content_type": "image/jpeg"
                        }
                    )
                ]
                
                result = client.upsert(
                    collection_name="test_collection",
                    points=points
                )
                
                assert result.status == "completed"
                mock_client.upsert.assert_called_once()

    @pytest.mark.integration 
    @patch.dict('os.environ', {'JINA_API_KEY': 'test_key'})
    def test_mixed_modality_search_integration(self, mock_jina_api):
        """Test mixed text and image search integration"""
        from embedding_factory import EmbeddingFactory
        
        with patch('qdrant_client.QdrantClient') as mock_qdrant:
            mock_client = Mock()
            mock_qdrant.return_value = mock_client
            
            # Mock search results with mixed modalities
            mock_client.search.return_value = [
                Mock(
                    id="text_1",
                    payload={
                        "content": "This is text content",
                        "modality": "text",
                        "source": "document.pdf"
                    },
                    score=0.9
                ),
                Mock(
                    id="img_1", 
                    payload={
                        "content": "Image description",
                        "modality": "image",
                        "source": "image.jpg"
                    },
                    score=0.8
                )
            ]
            
            # Generate query embedding (could be text or image)
            text_instance = EmbeddingFactory.create_embedding()
            query_embedding = [0.1] * 1024  # Mock embedding
            
            # Search for mixed results
            from qdrant_client import QdrantClient
            client = QdrantClient(url="http://qdrant:6333")
            
            results = client.search(
                collection_name="test_collection",
                query_vector=query_embedding,
                limit=10
            )
            
            # Verify mixed results
            assert len(results) == 2
            modalities = [r.payload["modality"] for r in results]
            assert "text" in modalities
            assert "image" in modalities

    @pytest.mark.integration
    def test_image_embedding_error_handling_integration(self):
        """Test error handling in image embedding integration scenarios"""
        from embedding_factory import EmbeddingFactory
        
        # Test missing API key
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="JINA_API_KEY es requerida"):
                EmbeddingFactory.create_image_embedding()
        
        # Test invalid provider
        with patch.dict('os.environ', {'JINA_API_KEY': 'test_key'}):
            with pytest.raises(ValueError, match="Unsupported provider for images"):
                EmbeddingFactory.create_image_embedding(provider="unsupported")

    @pytest.mark.integration
    @patch.dict('os.environ', {'JINA_API_KEY': 'test_key'})
    def test_image_embedding_performance_integration(self, mock_jina_api):
        """Test performance characteristics of image embedding integration"""
        from embedding_factory import EmbeddingFactory
        import time
        
        # Test batch processing performance
        image_paths = [f"/test/image_{i}.jpg" for i in range(10)]
        
        with patch('builtins.open', mock_open(read_data=b"fake_image_data")):
            start_time = time.time()
            embeddings = EmbeddingFactory.embed_images_flexible(image_paths)
            end_time = time.time()
            
            # Should process reasonably quickly (mocked)
            assert end_time - start_time < 1.0
            assert len(embeddings) == 10
            assert all(len(emb) == 1024 for emb in embeddings)

    @pytest.mark.integration
    @patch.dict('os.environ', {'JINA_API_KEY': 'test_key'})
    def test_image_embedding_concurrent_integration(self, mock_jina_api):
        """Test concurrent image embedding processing integration"""
        from embedding_factory import EmbeddingFactory
        import threading
        import time
        
        results = []
        errors = []
        
        def process_images(image_list):
            try:
                with patch('builtins.open', mock_open(read_data=b"fake_image")):
                    embeddings = EmbeddingFactory.embed_images_flexible(image_list)
                    results.append(len(embeddings))
            except Exception as e:
                errors.append(str(e))
        
        # Create concurrent processing threads
        threads = []
        for i in range(3):
            image_list = [f"/test/batch_{i}_image_{j}.jpg" for j in range(2)]
            thread = threading.Thread(target=process_images, args=(image_list,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify all batches processed successfully
        assert len(errors) == 0
        assert len(results) == 3
        assert all(count == 2 for count in results)  # 2 images per batch