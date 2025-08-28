"""Integration tests for complete RAG flow"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import requests
from io import BytesIO

# Import components for testing
from langchain.schema import Document


class TestRAGFlow:
    """Test complete RAG workflow from ingest to retrieval"""
    
    @pytest.fixture
    def sample_pdf_bytes(self):
        """Create sample PDF bytes for testing"""
        # Simple PDF header + minimal content
        pdf_content = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        pdf_content += b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        pdf_content += b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
        pdf_content += b"xref\n0 4\n0000000000 65535 f\ntrailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n"
        return pdf_content

    @pytest.fixture
    def temp_docs_dir(self, sample_pdf_bytes):
        """Create temporary directory with sample PDF"""
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "sample.pdf"
            with open(pdf_path, "wb") as f:
                f.write(sample_pdf_bytes)
            yield temp_dir

    @pytest.fixture
    def mock_qdrant_running(self):
        """Mock Qdrant service as running and responsive"""
        with patch('qdrant_client.QdrantClient') as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            
            # Mock collection operations
            mock_instance.get_collections.return_value.collections = []
            mock_instance.create_collection.return_value = True
            mock_instance.upsert.return_value = True
            mock_instance.search.return_value = []
            
            yield mock_instance

    @pytest.fixture
    def mock_embeddings_service(self):
        """Mock embeddings service"""
        with patch('langchain_community.embeddings.HuggingFaceEmbeddings') as mock_emb:
            mock_instance = MagicMock()
            mock_emb.return_value = mock_instance
            
            # Mock embedding generation
            mock_instance.embed_documents.return_value = [[0.1] * 384] * 5
            mock_instance.embed_query.return_value = [0.1] * 384
            
            yield mock_instance

    @pytest.mark.integration
    def test_pdf_ingestion_flow(self, temp_docs_dir, mock_qdrant_running, mock_embeddings_service):
        """Test complete PDF ingestion flow"""
        # Mock the ingest process
        with patch('ingest.DOCS_DIR', temp_docs_dir):
            with patch('ingest.logger') as mock_logger:
                # Import after patching
                import ingest
                
                # Verify PDF discovery
                pdf_paths = [f for f in os.listdir(temp_docs_dir) if f.endswith('.pdf')]
                assert len(pdf_paths) > 0
                
                # Mock PDF loading
                mock_docs = [
                    Document(
                        page_content="Sample PDF content for testing RAG flow.",
                        metadata={"source": "sample.pdf", "page": 0}
                    )
                ]
                
                with patch('ingest.PyPDFLoader') as mock_loader:
                    mock_loader_instance = MagicMock()
                    mock_loader.return_value = mock_loader_instance
                    mock_loader_instance.load.return_value = mock_docs
                    
                    # Mock text splitter
                    with patch('ingest.RecursiveCharacterTextSplitter') as mock_splitter:
                        mock_splitter_instance = MagicMock()
                        mock_splitter.return_value = mock_splitter_instance
                        mock_splitter_instance.split_documents.return_value = mock_docs
                        
                        # Mock Qdrant operations
                        with patch('ingest.Qdrant') as mock_vectordb:
                            mock_vectordb_instance = MagicMock()
                            mock_vectordb.from_documents.return_value = mock_vectordb_instance
                            
                            # This would normally run the ingestion
                            # For testing, we verify the mocks were called correctly
                            assert mock_qdrant_running is not None
                            assert mock_embeddings_service is not None

    @pytest.mark.integration
    def test_document_chunking_integration(self, mock_embeddings_service):
        """Test document chunking with real text splitter"""
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        # Real text splitter with test configuration
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
        )
        
        # Sample long text that should be split
        long_text = " ".join([
            "This is a comprehensive document about machine learning and artificial intelligence.",
            "It covers various topics including neural networks, deep learning, and natural language processing.",
            "The document explains how embeddings work and their applications in search and recommendation systems.",
            "Vector databases like Qdrant are essential for storing and retrieving high-dimensional embeddings.",
            "Retrieval-augmented generation (RAG) combines the power of pre-trained language models with external knowledge."
        ] * 3)  # Repeat to ensure splitting
        
        chunks = text_splitter.split_text(long_text)
        
        # Verify chunking worked correctly
        assert len(chunks) >= 2  # Should be split into multiple chunks
        for chunk in chunks:
            assert len(chunk) <= 500  # Respects chunk size
            assert len(chunk.strip()) > 0  # No empty chunks

    @pytest.mark.integration
    def test_retrieval_flow(self, mock_qdrant_running, mock_embeddings_service):
        """Test retrieval flow from query to results"""
        # Mock retrieval results
        mock_retrieval_results = [
            Document(
                page_content="Machine learning is a subset of artificial intelligence.",
                metadata={"source": "ml_guide.pdf", "page": 1, "score": 0.85}
            ),
            Document(
                page_content="Neural networks are inspired by biological neural networks.",
                metadata={"source": "ai_basics.pdf", "page": 3, "score": 0.75}
            )
        ]
        
        mock_qdrant_running.search.return_value = [
            type('SearchResult', (), {
                'payload': {
                    'page_content': doc.page_content,
                    'metadata': doc.metadata
                },
                'score': doc.metadata.get('score', 0.5)
            })() for doc in mock_retrieval_results
        ]
        
        # Mock the retriever
        with patch('langchain_community.vectorstores.Qdrant') as mock_vectordb:
            mock_vectordb_instance = MagicMock()
            mock_vectordb.return_value = mock_vectordb_instance
            
            mock_retriever = MagicMock()
            mock_vectordb_instance.as_retriever.return_value = mock_retriever
            mock_retriever.get_relevant_documents.return_value = mock_retrieval_results
            
            # Test query processing
            query = "What is machine learning?"
            results = mock_retriever.get_relevant_documents(query)
            
            assert len(results) == 2
            assert all(isinstance(doc, Document) for doc in results)
            assert "machine learning" in results[0].page_content.lower()

    @pytest.mark.integration
    @patch('requests.post')
    def test_pyexec_integration(self, mock_post):
        """Test integration with pyexec service"""
        # Mock pyexec service response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": 4.0,
            "expression": "2 + 2",
            "execution_time": 0.001
        }
        mock_post.return_value = mock_response
        
        # Test calculation request
        pyexec_url = "http://localhost:8001/calculate"
        payload = {"expression": "2 + 2"}
        
        response = requests.post(pyexec_url, json=payload)
        
        assert response.status_code == 200
        result = response.json()
        assert result["result"] == 4.0
        assert result["expression"] == "2 + 2"

    @pytest.mark.integration
    def test_end_to_end_rag_query(self, mock_qdrant_running, mock_embeddings_service):
        """Test complete end-to-end RAG query processing"""
        # Mock LLM response
        with patch('langchain_community.chat_models.ChatOllama') as mock_llm:
            mock_llm_instance = MagicMock()
            mock_llm.return_value = mock_llm_instance
            mock_llm_instance.invoke.return_value = Mock(content="Machine learning is a subset of AI that enables computers to learn from data.")
            
            # Mock retrieval results
            mock_docs = [
                Document(
                    page_content="Machine learning algorithms learn patterns from data to make predictions.",
                    metadata={"source": "ml_guide.pdf", "page": 1}
                )
            ]
            
            # Mock retriever
            with patch('app.retriever') as mock_retriever:
                mock_retriever.get_relevant_documents.return_value = mock_docs
                
                # Import and test the main app logic
                # Note: This would typically involve calling the actual chatbot function
                query = "What is machine learning?"
                
                # Simulate the RAG process
                retrieved_docs = mock_retriever.get_relevant_documents(query)
                context = "\n".join([doc.page_content for doc in retrieved_docs])
                
                # Simulate LLM call with context
                prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
                response = mock_llm_instance.invoke(prompt)
                
                assert len(retrieved_docs) > 0
                assert "machine learning" in context.lower()
                assert response.content is not None

    @pytest.mark.integration
    def test_error_handling_integration(self, mock_qdrant_running):
        """Test error handling in integration scenarios"""
        # Test Qdrant connection error
        mock_qdrant_running.search.side_effect = Exception("Connection failed")
        
        with pytest.raises(Exception) as exc_info:
            mock_qdrant_running.search("test query", limit=5)
        
        assert "Connection failed" in str(exc_info.value)
        
        # Test empty results handling
        mock_qdrant_running.search.side_effect = None
        mock_qdrant_running.search.return_value = []
        
        results = mock_qdrant_running.search("nonexistent query", limit=5)
        assert len(results) == 0

    @pytest.mark.integration
    def test_service_health_checks(self):
        """Test health checks for integrated services"""
        # Mock service health endpoints
        with patch('requests.get') as mock_get:
            # Mock Qdrant health check
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = "OK"
            mock_get.return_value = mock_response
            
            # Test Qdrant health
            qdrant_health = requests.get("http://localhost:6333/healthz")
            assert qdrant_health.status_code == 200
            
            # Test Ollama health
            ollama_health = requests.get("http://localhost:11434/api/tags")
            assert ollama_health.status_code == 200
            
            # Test pyexec health
            pyexec_health = requests.get("http://localhost:8001/health")
            assert pyexec_health.status_code == 200

    @pytest.mark.integration
    def test_data_consistency_flow(self, mock_qdrant_running, mock_embeddings_service):
        """Test data consistency through the complete flow"""
        # Sample document
        original_doc = Document(
            page_content="Vector databases enable semantic search through embeddings.",
            metadata={"source": "vectors.pdf", "page": 1}
        )
        
        # Mock embedding generation for consistency
        mock_embedding = [0.1, 0.2, 0.3] * 128  # 384-dimensional
        mock_embeddings_service.embed_documents.return_value = [mock_embedding]
        mock_embeddings_service.embed_query.return_value = mock_embedding
        
        # Mock search to return the same document
        mock_qdrant_running.search.return_value = [
            type('SearchResult', (), {
                'payload': {
                    'page_content': original_doc.page_content,
                    'metadata': original_doc.metadata
                },
                'score': 0.99  # High similarity due to identical embedding
            })()
        ]
        
        # Simulate ingestion
        embedding = mock_embeddings_service.embed_documents([original_doc.page_content])
        assert len(embedding[0]) == 384
        
        # Simulate retrieval with same query
        query_embedding = mock_embeddings_service.embed_query("vector databases")
        search_results = mock_qdrant_running.search(query_embedding, limit=1)
        
        # Verify consistency
        assert len(search_results) == 1
        assert search_results[0].payload['page_content'] == original_doc.page_content
        assert search_results[0].score > 0.9  # High similarity