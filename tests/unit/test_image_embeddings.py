"""Unit tests for image embedding functionality"""

import pytest
import base64
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock, mock_open
from PIL import Image
import io

# Import the modules we're testing
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from embedding_factory import EmbeddingFactory
from jina_embeddings import JinaEmbeddings


class TestImageEmbeddings:
    """Test image embedding functionality"""
    
    def setup_method(self):
        """Setup test data"""
        self.expected_dimension = 1024  # Jina v4 default
        self.test_provider = "jina"
        self.test_model = "jina-embeddings-v4"
        
        # Create sample image data
        self.sample_image_b64 = self._create_test_image_base64()
        self.sample_image_bytes = base64.b64decode(self.sample_image_b64)
        
        # Sample paths
        self.sample_image_paths = [
            "/path/to/image1.jpg",
            "/path/to/image2.png",
            "/path/to/image3.gif"
        ]

    def _create_test_image_base64(self) -> str:
        """Create a small test image in base64 format"""
        # Create a simple 10x10 red image
        img = Image.new('RGB', (10, 10), color='red')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()
        return base64.b64encode(image_bytes).decode('utf-8')

    @patch.dict(os.environ, {'JINA_API_KEY': 'test_key', 'IMAGE_EMBEDDING_PROVIDER': 'jina'})
    def test_create_image_embedding_success(self):
        """Test successful creation of image embedding instance"""
        with patch('embedding_factory.EmbeddingFactory._create_jina_embedding') as mock_create:
            mock_embedding = Mock()
            mock_create.return_value = mock_embedding
            
            result = EmbeddingFactory.create_image_embedding()
            
            assert result == mock_embedding
            mock_create.assert_called_once()

    def test_create_image_embedding_google_provider_error(self):
        """Test that Google provider raises appropriate error for images"""
        with pytest.raises(ValueError, match="Google embeddings no soporta imágenes"):
            EmbeddingFactory.create_image_embedding(provider="google")

    def test_create_image_embedding_unsupported_provider(self):
        """Test error for unsupported provider"""
        with pytest.raises(ValueError, match="Unsupported provider for images"):
            EmbeddingFactory.create_image_embedding(provider="unsupported")

    @patch.dict(os.environ, {'JINA_API_KEY': 'test_key'})
    def test_embed_images_flexible_single_path(self):
        """Test embedding single image from file path"""
        with patch.object(EmbeddingFactory, 'create_image_embedding') as mock_create, \
             patch.object(EmbeddingFactory, '_is_base64_image', return_value=False):
            
            mock_embedding = Mock()
            mock_embedding.embed_images.return_value = [[0.1] * self.expected_dimension]
            mock_create.return_value = mock_embedding
            
            result = EmbeddingFactory.embed_images_flexible("/path/to/test.jpg")
            
            assert len(result) == 1
            assert len(result[0]) == self.expected_dimension
            mock_embedding.embed_images.assert_called_once_with(["/path/to/test.jpg"])

    @patch.dict(os.environ, {'JINA_API_KEY': 'test_key'})
    def test_embed_images_flexible_base64(self):
        """Test embedding image from base64 string"""
        with patch.object(EmbeddingFactory, 'create_image_embedding') as mock_create, \
             patch.object(EmbeddingFactory, '_is_base64_image', return_value=True):
            
            mock_embedding = Mock()
            mock_embedding.embed_images_data.return_value = [[0.2] * self.expected_dimension]
            mock_create.return_value = mock_embedding
            
            result = EmbeddingFactory.embed_images_flexible(self.sample_image_b64)
            
            assert len(result) == 1
            assert len(result[0]) == self.expected_dimension
            mock_embedding.embed_images_data.assert_called_once()

    @patch.dict(os.environ, {'JINA_API_KEY': 'test_key'})
    def test_embed_images_flexible_bytes(self):
        """Test embedding image from bytes data"""
        with patch.object(EmbeddingFactory, 'create_image_embedding') as mock_create:
            mock_embedding = Mock()
            mock_embedding.embed_images_data.return_value = [[0.3] * self.expected_dimension]
            mock_create.return_value = mock_embedding
            
            result = EmbeddingFactory.embed_images_flexible(self.sample_image_bytes)
            
            assert len(result) == 1
            assert len(result[0]) == self.expected_dimension
            mock_embedding.embed_images_data.assert_called_once_with([self.sample_image_bytes])

    @patch.dict(os.environ, {'JINA_API_KEY': 'test_key'})
    def test_embed_images_flexible_mixed_list(self):
        """Test embedding mixed list of images (paths, base64, bytes)"""
        with patch.object(EmbeddingFactory, 'create_image_embedding') as mock_create, \
             patch.object(EmbeddingFactory, '_is_base64_image') as mock_is_b64:
            
            mock_embedding = Mock()
            mock_embedding.embed_images.return_value = [[0.1] * self.expected_dimension]
            mock_embedding.embed_images_data.return_value = [[0.2] * self.expected_dimension, [0.3] * self.expected_dimension]
            mock_create.return_value = mock_embedding
            
            # Mock base64 detection
            mock_is_b64.side_effect = lambda x: x == self.sample_image_b64
            
            mixed_images = [
                "/path/to/image.jpg",  # file path
                self.sample_image_b64,  # base64
                self.sample_image_bytes  # bytes
            ]
            
            result = EmbeddingFactory.embed_images_flexible(mixed_images)
            
            assert len(result) == 3
            assert all(len(vec) == self.expected_dimension for vec in result)

    def test_is_base64_image_valid_png(self):
        """Test base64 image detection for valid PNG"""
        result = EmbeddingFactory._is_base64_image(self.sample_image_b64)
        assert result is True

    def test_is_base64_image_invalid_short(self):
        """Test base64 image detection rejects short strings"""
        short_string = "short"
        result = EmbeddingFactory._is_base64_image(short_string)
        assert result is False

    def test_is_base64_image_invalid_characters(self):
        """Test base64 image detection rejects invalid characters"""
        invalid_string = "invalid_characters!@#$%^&*()" * 20
        result = EmbeddingFactory._is_base64_image(invalid_string)
        assert result is False

    def test_is_base64_image_invalid_decode(self):
        """Test base64 image detection handles decode errors"""
        # Valid base64 but not an image
        text_b64 = base64.b64encode(b"This is just text, not an image").decode('utf-8')
        result = EmbeddingFactory._is_base64_image(text_b64)
        assert result is False

    @patch.dict(os.environ, {'JINA_API_KEY': 'test_key', 'EMBEDDING_PROVIDER': 'jina'})
    def test_verify_embedding_compatibility_success(self):
        """Test embedding compatibility verification with matching settings"""
        with patch.object(EmbeddingFactory, 'get_model_dimensions', return_value=1024), \
             patch.object(EmbeddingFactory, 'create_embedding') as mock_text, \
             patch.object(EmbeddingFactory, 'create_image_embedding') as mock_image:
            
            # Mock embedding instances with compatible settings
            mock_text_embedding = Mock()
            mock_text_embedding.model = "jina-embeddings-v4"
            mock_text_embedding.dimensions = 1024
            mock_text_embedding.normalized = True
            mock_text_embedding.task_type = "retrieval.passage"
            
            mock_image_embedding = Mock()
            mock_image_embedding.model = "jina-embeddings-v4"
            mock_image_embedding.dimensions = 1024
            mock_image_embedding.normalized = True
            mock_image_embedding.task_type = "retrieval.passage"
            
            mock_text.return_value = mock_text_embedding
            mock_image.return_value = mock_image_embedding
            
            result = EmbeddingFactory.verify_embedding_compatibility()
            
            assert result["compatible"] is True
            assert result["same_dimensions"] is True
            assert result["critical_settings_match"] is True
            assert result["ready_for_mixed_search"] is True
            assert result["text_dimensions"] == 1024
            assert result["image_dimensions"] == 1024

    @patch.dict(os.environ, {'JINA_API_KEY': 'test_key'})
    def test_verify_embedding_compatibility_dimension_mismatch(self):
        """Test compatibility verification with dimension mismatch"""
        with patch.object(EmbeddingFactory, 'get_model_dimensions') as mock_dims, \
             patch.object(EmbeddingFactory, 'create_embedding') as mock_text, \
             patch.object(EmbeddingFactory, 'create_image_embedding') as mock_image:
            
            # Mock different dimensions
            mock_dims.side_effect = lambda model, provider: 768 if "text" in model else 1024
            
            mock_text_embedding = Mock()
            mock_text_embedding.dimensions = 768
            mock_text_embedding.normalized = True
            mock_text_embedding.task_type = "retrieval.passage"
            
            mock_image_embedding = Mock()
            mock_image_embedding.dimensions = 1024
            mock_image_embedding.normalized = True
            mock_image_embedding.task_type = "retrieval.passage"
            
            mock_text.return_value = mock_text_embedding
            mock_image.return_value = mock_image_embedding
            
            result = EmbeddingFactory.verify_embedding_compatibility(
                text_model_name="text-model", 
                image_model_name="image-model"
            )
            
            assert result["compatible"] is False
            assert result["same_dimensions"] is False
            assert result["critical_settings_match"] is False
            assert result["ready_for_mixed_search"] is False

    def test_embed_images_flexible_unsupported_type(self):
        """Test error handling for unsupported image type"""
        with pytest.raises(ValueError, match="Unsupported image type"):
            EmbeddingFactory.embed_images_flexible(123)  # Invalid type

    @patch.dict(os.environ, {'JINA_API_KEY': 'test_key'})
    def test_embed_images_flexible_base64_decode_error(self):
        """Test error handling for invalid base64"""
        with patch.object(EmbeddingFactory, 'create_image_embedding') as mock_create, \
             patch.object(EmbeddingFactory, '_is_base64_image', return_value=True), \
             patch('base64.b64decode', side_effect=Exception("Invalid base64")):
            
            mock_embedding = Mock()
            mock_create.return_value = mock_embedding
            
            with pytest.raises(ValueError, match="Invalid base64 image data"):
                EmbeddingFactory.embed_images_flexible("invalid_base64_string")


class TestJinaImageEmbeddings:
    """Test Jina-specific image embedding functionality"""
    
    def setup_method(self):
        """Setup test data for Jina tests"""
        self.api_key = "test_jina_key"
        self.expected_dimension = 1024
        self.sample_paths = ["/path/to/test1.jpg", "/path/to/test2.png"]
        self.sample_image_data = [b"fake_image_data_1", b"fake_image_data_2"]

    @patch('jina_embeddings.requests.Session')
    def test_embed_images_success(self, mock_session):
        """Test successful image embedding from file paths"""
        # Mock file reading
        with patch('builtins.open', mock_open(read_data=b"fake_image_data")):
            # Mock API response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": [
                    {"embedding": [0.1] * self.expected_dimension},
                    {"embedding": [0.2] * self.expected_dimension}
                ]
            }
            mock_response.raise_for_status = Mock()
            
            mock_session_instance = Mock()
            mock_session_instance.post.return_value = mock_response
            mock_session.return_value = mock_session_instance
            
            embeddings = JinaEmbeddings(api_key=self.api_key)
            result = embeddings.embed_images(self.sample_paths)
            
            assert len(result) == 2
            assert all(len(vec) == self.expected_dimension for vec in result)
            assert result[0] == [0.1] * self.expected_dimension
            assert result[1] == [0.2] * self.expected_dimension

    @patch('jina_embeddings.requests.Session')
    def test_embed_images_data_success(self, mock_session):
        """Test successful image embedding from bytes data"""
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"embedding": [0.3] * self.expected_dimension},
                {"embedding": [0.4] * self.expected_dimension}
            ]
        }
        mock_response.raise_for_status = Mock()
        
        mock_session_instance = Mock()
        mock_session_instance.post.return_value = mock_response
        mock_session.return_value = mock_session_instance
        
        embeddings = JinaEmbeddings(api_key=self.api_key)
        result = embeddings.embed_images_data(self.sample_image_data)
        
        assert len(result) == 2
        assert all(len(vec) == self.expected_dimension for vec in result)

    def test_embed_images_empty_list(self):
        """Test handling of empty image list"""
        embeddings = JinaEmbeddings(api_key=self.api_key)
        result = embeddings.embed_images([])
        assert result == []

    def test_embed_images_data_empty_list(self):
        """Test handling of empty image data list"""
        embeddings = JinaEmbeddings(api_key=self.api_key)
        result = embeddings.embed_images_data([])
        assert result == []

    def test_embed_images_file_not_found(self):
        """Test error handling for non-existent image files"""
        with patch('builtins.open', side_effect=FileNotFoundError("File not found")):
            embeddings = JinaEmbeddings(api_key=self.api_key)
            
            with pytest.raises(ValueError, match="No se pudo leer la imagen"):
                embeddings.embed_images(["/nonexistent/image.jpg"])

    @patch('jina_embeddings.requests.Session')
    def test_embed_images_api_error(self, mock_session):
        """Test handling of API errors during image embedding"""
        with patch('builtins.open', mock_open(read_data=b"fake_image_data")):
            # Mock API error response
            mock_response = Mock()
            mock_response.status_code = 400
            mock_response.raise_for_status.side_effect = Exception("API Error")
            
            mock_session_instance = Mock()
            mock_session_instance.post.return_value = mock_response
            mock_session.return_value = mock_session_instance
            
            embeddings = JinaEmbeddings(api_key=self.api_key)
            
            with pytest.raises(Exception):
                embeddings.embed_images(["/path/to/test.jpg"])

    @patch('jina_embeddings.requests.Session')
    def test_embed_images_data_encoding_error(self, mock_session):
        """Test handling of encoding errors for image data"""
        # Mock base64 encoding error
        with patch('base64.b64encode', side_effect=Exception("Encoding error")):
            embeddings = JinaEmbeddings(api_key=self.api_key)
            
            with pytest.raises(ValueError, match="No se pudieron codificar los datos de imagen"):
                embeddings.embed_images_data([b"fake_data"])

    def test_jina_embeddings_initialization_with_image_defaults(self):
        """Test JinaEmbeddings initialization with image-optimized settings"""
        embeddings = JinaEmbeddings(
            api_key=self.api_key,
            task_type="retrieval.passage",
            dimensions=1024,
            normalized=True,
            late_chunking=False
        )
        
        assert embeddings.api_key == self.api_key
        assert embeddings.task_type == "retrieval.passage"
        assert embeddings.dimensions == 1024
        assert embeddings.normalized is True
        assert embeddings.late_chunking is False


class TestImageEmbeddingIntegration:
    """Integration tests for image embedding functionality"""
    
    def setup_method(self):
        """Setup integration test data"""
        self.test_env = {
            'JINA_API_KEY': 'test_key',
            'IMAGE_EMBEDDING_PROVIDER': 'jina',
            'IMAGE_EMBEDDING_MODEL': 'jina-embeddings-v4'
        }

    @patch.dict(os.environ, {'JINA_API_KEY': 'test_key'})
    def test_full_image_embedding_workflow(self):
        """Test complete workflow from factory to embedding generation"""
        with patch.object(EmbeddingFactory, '_create_jina_embedding') as mock_create:
            # Create mock embedding instance
            mock_embedding = Mock()
            mock_embedding.model = "jina-embeddings-v4"
            mock_embedding.dimensions = 1024
            mock_embedding.normalized = True
            mock_embedding.task_type = "retrieval.passage"
            mock_embedding.embed_images.return_value = [[0.1] * 1024, [0.2] * 1024]
            mock_create.return_value = mock_embedding
            
            # Test factory method
            embedding_instance = EmbeddingFactory.create_image_embedding()
            assert embedding_instance == mock_embedding
            
            # Test embedding generation
            with patch.object(EmbeddingFactory, '_is_base64_image', return_value=False):
                result = EmbeddingFactory.embed_images_flexible(["/path1.jpg", "/path2.png"])
                
                assert len(result) == 2
                assert all(len(vec) == 1024 for vec in result)

    @patch.dict(os.environ, {'JINA_API_KEY': 'test_key', 'EMBEDDING_PROVIDER': 'jina'})
    def test_text_image_compatibility_workflow(self):
        """Test full compatibility check workflow"""
        with patch.object(EmbeddingFactory, 'get_model_dimensions', return_value=1024), \
             patch.object(EmbeddingFactory, 'create_embedding') as mock_text, \
             patch.object(EmbeddingFactory, 'create_image_embedding') as mock_image:
            
            # Create compatible mock instances
            text_mock = Mock()
            text_mock.model = "jina-embeddings-v4"
            text_mock.dimensions = 1024
            text_mock.normalized = True
            text_mock.task_type = "retrieval.passage"
            
            image_mock = Mock()
            image_mock.model = "jina-embeddings-v4"
            image_mock.dimensions = 1024
            image_mock.normalized = True
            image_mock.task_type = "retrieval.passage"
            
            mock_text.return_value = text_mock
            mock_image.return_value = image_mock
            
            # Test compatibility verification
            compatibility = EmbeddingFactory.verify_embedding_compatibility()
            
            assert compatibility["ready_for_mixed_search"] is True
            assert compatibility["critical_settings_match"] is True
            assert compatibility["text_dimensions"] == compatibility["image_dimensions"]

    def test_error_propagation_in_workflow(self):
        """Test that errors are properly propagated through the workflow"""
        with patch.dict(os.environ, {'JINA_API_KEY': 'test_key'}):
            with patch.object(EmbeddingFactory, 'create_image_embedding', side_effect=Exception("API Error")):
                with pytest.raises(Exception, match="API Error"):
                    EmbeddingFactory.embed_images_flexible(["/path/to/image.jpg"])