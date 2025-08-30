"""Pytest configuration and shared fixtures"""

import pytest
import os
import tempfile
import shutil
from unittest.mock import Mock, MagicMock
from typing import Generator, Dict, Any
import logging

# Configure test logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Test environment variables
@pytest.fixture(scope="session")
def test_env() -> Dict[str, str]:
    """Test environment configuration"""
    return {
        "QDRANT_URL": "http://localhost:6333",
        "COLLECTION_NAME": "test_collection",
        "GOOGLE_API_KEY": "test_google_api_key",
        "JINA_API_KEY": "test_jina_api_key",
        "PYEXEC_URL": "http://localhost:8001",
        "MAX_QUERY_LENGTH": "2000",
        "MIN_QUERY_LENGTH": "1",
        "MAX_HISTORY_CHARS": "8000"
    }

@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Create temporary directory for tests"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)

@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for testing"""
    mock_client = MagicMock()
    mock_client.search.return_value = []
    mock_client.get_collection.return_value = True
    return mock_client

@pytest.fixture
def mock_google_llm():
    """Mock Google Gemini LLM for testing"""
    mock_client = MagicMock()
    mock_client.invoke.return_value = "Test response from Google Gemini"
    return mock_client

@pytest.fixture
def mock_embeddings():
    """Mock embeddings for testing"""
    mock_emb = MagicMock()
    mock_emb.embed_query.return_value = [0.1] * 384  # Standard embedding size
    mock_emb.embed_documents.return_value = [[0.1] * 384]
    return mock_emb

@pytest.fixture
def sample_pdf_content() -> str:
    """Sample PDF text content for testing"""
    return """
    This is a sample PDF document for testing purposes.
    
    It contains multiple paragraphs with technical information
    about Python programming and machine learning concepts.
    
    The document discusses vector databases, embeddings,
    and retrieval-augmented generation (RAG) systems.
    
    This content will be used to test chunking, embedding,
    and retrieval functionality in our RAG system.
    """

@pytest.fixture
def sample_chunks() -> list[str]:
    """Sample text chunks for testing"""
    return [
        "This is a sample PDF document for testing purposes.",
        "It contains multiple paragraphs with technical information about Python programming and machine learning concepts.",
        "The document discusses vector databases, embeddings, and retrieval-augmented generation (RAG) systems.",
        "This content will be used to test chunking, embedding, and retrieval functionality in our RAG system."
    ]

@pytest.fixture
def mock_requests():
    """Mock requests module for HTTP testing"""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"status": "ok"}
    mock_resp.text = "OK"
    
    mock = MagicMock()
    mock.get.return_value = mock_resp
    mock.post.return_value = mock_resp
    return mock

@pytest.fixture(autouse=True)
def setup_test_env(test_env):
    """Automatically set up test environment variables"""
    for key, value in test_env.items():
        os.environ[key] = value
    yield
    # Cleanup is automatic since we're just setting env vars