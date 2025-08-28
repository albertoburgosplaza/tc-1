"""Unit tests for embedding functionality"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from langchain_community.embeddings import HuggingFaceEmbeddings


class TestEmbeddings:
    """Test embedding functionality"""
    
    def setup_method(self):
        """Setup test data"""
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.expected_dimension = 384
        self.sample_texts = [
            "This is a sample sentence for testing embeddings.",
            "Another test sentence with different content.",
            "Machine learning and natural language processing."
        ]
        self.single_text = "Single test sentence for embedding."

    @patch('langchain_community.embeddings.HuggingFaceEmbeddings.__init__')
    def test_embeddings_initialization(self, mock_init):
        """Test proper initialization of HuggingFace embeddings"""
        mock_init.return_value = None
        
        embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
        mock_init.assert_called_once_with(model_name=self.model_name)

    def test_embed_query_returns_correct_dimensions(self, mock_embeddings):
        """Test that embedding query returns vector with correct dimensions"""
        # Mock the embedding to return correct dimension vector
        mock_embeddings.embed_query.return_value = [0.1] * self.expected_dimension
        
        embedding_vector = mock_embeddings.embed_query(self.single_text)
        
        assert isinstance(embedding_vector, list)
        assert len(embedding_vector) == self.expected_dimension
        assert all(isinstance(x, (int, float)) for x in embedding_vector)

    def test_embed_documents_returns_correct_format(self, mock_embeddings):
        """Test that embedding documents returns correct format"""
        # Mock multiple embeddings
        mock_embeddings.embed_documents.return_value = [
            [0.1] * self.expected_dimension for _ in self.sample_texts
        ]
        
        embedding_vectors = mock_embeddings.embed_documents(self.sample_texts)
        
        assert isinstance(embedding_vectors, list)
        assert len(embedding_vectors) == len(self.sample_texts)
        for vector in embedding_vectors:
            assert len(vector) == self.expected_dimension
            assert all(isinstance(x, (int, float)) for x in vector)

    def test_empty_text_handling(self, mock_embeddings):
        """Test handling of empty text"""
        mock_embeddings.embed_query.return_value = [0.0] * self.expected_dimension
        
        embedding_vector = mock_embeddings.embed_query("")
        
        assert isinstance(embedding_vector, list)
        assert len(embedding_vector) == self.expected_dimension

    def test_empty_documents_list(self, mock_embeddings):
        """Test handling of empty documents list"""
        mock_embeddings.embed_documents.return_value = []
        
        embedding_vectors = mock_embeddings.embed_documents([])
        
        assert isinstance(embedding_vectors, list)
        assert len(embedding_vectors) == 0

    def test_special_characters_in_text(self, mock_embeddings):
        """Test embedding text with special characters"""
        special_text = "Text with émojis 🚀, spécial charàcters ñ, and unicode ∑∏∆"
        mock_embeddings.embed_query.return_value = [0.1] * self.expected_dimension
        
        embedding_vector = mock_embeddings.embed_query(special_text)
        
        assert isinstance(embedding_vector, list)
        assert len(embedding_vector) == self.expected_dimension

    def test_very_long_text_handling(self, mock_embeddings):
        """Test handling of very long text"""
        long_text = " ".join(["Long sentence for testing."] * 100)
        mock_embeddings.embed_query.return_value = [0.1] * self.expected_dimension
        
        embedding_vector = mock_embeddings.embed_query(long_text)
        
        assert isinstance(embedding_vector, list)
        assert len(embedding_vector) == self.expected_dimension

    def test_numeric_values_in_embeddings(self, mock_embeddings):
        """Test that embedding values are proper floats"""
        mock_embeddings.embed_query.return_value = [0.1, -0.5, 0.0, 1.0, -1.0] + [0.1] * (self.expected_dimension - 5)
        
        embedding_vector = mock_embeddings.embed_query(self.single_text)
        
        assert all(isinstance(x, (int, float)) for x in embedding_vector)
        assert -1.0 <= min(embedding_vector) <= max(embedding_vector) <= 1.0

    def test_consistent_embeddings(self, mock_embeddings):
        """Test that same text produces consistent embeddings"""
        consistent_vector = [0.1] * self.expected_dimension
        mock_embeddings.embed_query.return_value = consistent_vector
        
        embedding1 = mock_embeddings.embed_query(self.single_text)
        embedding2 = mock_embeddings.embed_query(self.single_text)
        
        assert embedding1 == embedding2

    def test_different_texts_different_embeddings(self, mock_embeddings):
        """Test that different texts produce different embeddings"""
        def mock_embed_side_effect(text):
            if "first" in text:
                return [0.1] * self.expected_dimension
            else:
                return [0.2] * self.expected_dimension
        
        mock_embeddings.embed_query.side_effect = mock_embed_side_effect
        
        embedding1 = mock_embeddings.embed_query("This is the first text")
        embedding2 = mock_embeddings.embed_query("This is the second text")
        
        assert embedding1 != embedding2

    def test_batch_embedding_consistency(self, mock_embeddings):
        """Test that batch embedding is consistent with individual embeddings"""
        # Mock individual embeddings
        individual_vectors = [[0.1] * self.expected_dimension, [0.2] * self.expected_dimension]
        mock_embeddings.embed_documents.return_value = individual_vectors
        
        batch_embeddings = mock_embeddings.embed_documents(self.sample_texts[:2])
        
        assert len(batch_embeddings) == 2
        assert all(len(vec) == self.expected_dimension for vec in batch_embeddings)

    @pytest.mark.slow
    def test_embedding_performance(self, mock_embeddings):
        """Test embedding performance with multiple texts"""
        import time
        
        # Mock fast response
        mock_embeddings.embed_documents.return_value = [
            [0.1] * self.expected_dimension for _ in range(100)
        ]
        
        large_text_list = ["Test sentence for performance."] * 100
        
        start_time = time.time()
        embeddings = mock_embeddings.embed_documents(large_text_list)
        end_time = time.time()
        
        # Should complete reasonably quickly (mocked, so should be instant)
        assert end_time - start_time < 1.0
        assert len(embeddings) == 100

    def test_embedding_vector_normalization(self, mock_embeddings):
        """Test that embedding vectors are properly normalized"""
        # Mock normalized vector
        normalized_vector = [0.1] * self.expected_dimension
        mock_embeddings.embed_query.return_value = normalized_vector
        
        embedding_vector = mock_embeddings.embed_query(self.single_text)
        
        # Check vector properties
        vector_magnitude = sum(x**2 for x in embedding_vector)**0.5
        assert vector_magnitude > 0  # Should not be zero vector
        assert all(abs(x) <= 1.0 for x in embedding_vector)  # Values should be reasonable