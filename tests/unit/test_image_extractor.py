"""
Tests unitarios para el módulo image_extractor.py
"""

import pytest
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
from io import BytesIO
import hashlib

# Importar módulo a probar
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from image_extractor import (
    ExtractedImage, ImageExtractionError, ImageExtractor,
    extract_images_from_pdf, generate_thumbnail, save_processed_image,
    save_all_images, cleanup_document_images, get_image_metadata,
    _validate_image_format, _calculate_image_hash, 
    _normalize_image_orientation, _resize_image
)

class TestImageFormatValidation:
    """Tests para validación de formatos de imagen"""
    
    def test_validate_supported_formats(self):
        """Test que formatos soportados sean validados correctamente"""
        assert _validate_image_format('PNG') == True
        assert _validate_image_format('png') == True
        assert _validate_image_format('JPEG') == True
        assert _validate_image_format('jpeg') == True
        assert _validate_image_format('WEBP') == True
        assert _validate_image_format('webp') == True
    
    def test_validate_unsupported_formats(self):
        """Test que formatos no soportados sean rechazados"""
        assert _validate_image_format('BMP') == False
        assert _validate_image_format('TIFF') == False
        assert _validate_image_format('GIF') == False
        assert _validate_image_format('SVG') == False
        assert _validate_image_format('') == False

class TestImageHashing:
    """Tests para cálculo de hash de imágenes"""
    
    def test_calculate_hash_consistency(self):
        """Test que el hash sea consistente para los mismos datos"""
        image_data = b"test image data"
        hash1 = _calculate_image_hash(image_data)
        hash2 = _calculate_image_hash(image_data)
        
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex string
        assert isinstance(hash1, str)
    
    def test_calculate_hash_different_data(self):
        """Test que datos diferentes produzcan hashes diferentes"""
        data1 = b"test image data 1"
        data2 = b"test image data 2"
        
        hash1 = _calculate_image_hash(data1)
        hash2 = _calculate_image_hash(data2)
        
        assert hash1 != hash2

class TestImageProcessing:
    """Tests para procesamiento de imágenes"""
    
    def create_test_image(self, width=100, height=100, format='PNG'):
        """Crea imagen de prueba en memoria"""
        image = Image.new('RGB', (width, height), color='red')
        buffer = BytesIO()
        image.save(buffer, format=format)
        return buffer.getvalue()
    
    def test_resize_image_no_resize_needed(self):
        """Test que imágenes pequeñas no se redimensionen"""
        small_image = self.create_test_image(500, 500)
        resized = _resize_image(small_image, max_dimension=1024)
        
        # Verificar que la imagen no cambió de tamaño
        original = Image.open(BytesIO(small_image))
        processed = Image.open(BytesIO(resized))
        
        assert original.size == processed.size
    
    def test_resize_image_with_resize(self):
        """Test redimensionamiento de imagen grande"""
        # Crear imagen que necesite redimensionamiento
        large_image = self.create_test_image(2000, 1500)
        resized = _resize_image(large_image, max_dimension=1024)
        
        processed = Image.open(BytesIO(resized))
        
        # El lado mayor debe ser <= 1024
        assert max(processed.size) <= 1024
        # Debe mantener aspect ratio aproximadamente
        original_ratio = 2000 / 1500
        new_ratio = processed.size[0] / processed.size[1]
        assert abs(original_ratio - new_ratio) < 0.01
    
    def test_normalize_image_orientation(self):
        """Test normalización de orientación EXIF"""
        test_image = self.create_test_image()
        
        # Test que no falle con imagen sin EXIF
        normalized = _normalize_image_orientation(test_image)
        assert isinstance(normalized, bytes)
        assert len(normalized) > 0
    
    def test_generate_thumbnail(self):
        """Test generación de thumbnails"""
        original_image = self.create_test_image(1000, 800)
        thumbnail = generate_thumbnail(original_image, max_size=256)
        
        thumb_img = Image.open(BytesIO(thumbnail))
        
        # Verificar tamaño máximo
        assert max(thumb_img.size) <= 256
        # Verificar que es JPEG
        assert thumb_img.format == 'JPEG'
    
    def test_generate_thumbnail_small_image(self):
        """Test thumbnail de imagen ya pequeña"""
        small_image = self.create_test_image(100, 100)
        thumbnail = generate_thumbnail(small_image, max_size=256)
        
        thumb_img = Image.open(BytesIO(thumbnail))
        
        # Debe mantener tamaño original si ya es pequeña
        assert thumb_img.size == (100, 100)

class TestExtractedImageStructure:
    """Tests para estructura ExtractedImage"""
    
    def test_extracted_image_creation(self):
        """Test creación de ExtractedImage"""
        test_data = b"test image data"
        
        extracted = ExtractedImage(
            image_data=test_data,
            format='PNG',
            width=100,
            height=100,
            page_number=1,
            image_index=0,
            bbox=(0, 0, 100, 100),
            hash='testhash123'
        )
        
        assert extracted.image_data == test_data
        assert extracted.format == 'PNG'
        assert extracted.width == 100
        assert extracted.height == 100
        assert extracted.page_number == 1
        assert extracted.image_index == 0
        assert extracted.bbox == (0, 0, 100, 100)
        assert extracted.hash == 'testhash123'

class TestMetadataGeneration:
    """Tests para generación de metadata"""
    
    def test_get_image_metadata(self):
        """Test generación de metadata de imagen"""
        test_data = b"test image data"
        
        extracted = ExtractedImage(
            image_data=test_data,
            format='PNG',
            width=800,
            height=600,
            page_number=2,
            image_index=1,
            bbox=(10, 20, 110, 120),
            hash='abc123def456'
        )
        
        metadata = get_image_metadata(extracted, 'test_doc')
        
        assert metadata['doc_id'] == 'test_doc'
        assert metadata['page_number'] == 2
        assert metadata['image_index'] == 1
        assert metadata['format'] == 'PNG'
        assert metadata['width'] == 800
        assert metadata['height'] == 600
        assert metadata['hash'] == 'abc123def456'
        assert metadata['bbox']['x0'] == 10
        assert metadata['bbox']['y0'] == 20
        assert metadata['bbox']['x1'] == 110
        assert metadata['bbox']['y1'] == 120
        assert metadata['processed_size'] == len(test_data)

class TestStorageOperations:
    """Tests para operaciones de almacenamiento"""
    
    def setUp(self):
        """Setup para cada test"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_image_data = self.create_test_image()
        
    def tearDown(self):
        """Cleanup después de cada test"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def create_test_image(self):
        """Crea imagen de prueba"""
        image = Image.new('RGB', (100, 100), color='blue')
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        return buffer.getvalue()
    
    def test_save_processed_image(self):
        """Test guardado de imagen procesada"""
        self.setUp()
        
        try:
            extracted = ExtractedImage(
                image_data=self.test_image_data,
                format='PNG',
                width=100,
                height=100,
                page_number=1,
                image_index=0,
                bbox=(0, 0, 100, 100),
                hash='test_hash_123'
            )
            
            paths = save_processed_image(extracted, 'test_doc', self.temp_dir)
            
            # Verificar que se devuelven las rutas correctas
            assert 'image_path' in paths
            assert 'thumbnail_path' in paths
            assert 'image_uri' in paths
            assert 'thumbnail_uri' in paths
            
            # Verificar que los archivos existen
            assert os.path.exists(paths['image_path'])
            assert os.path.exists(paths['thumbnail_path'])
            
            # Verificar estructura de directorios
            expected_image_path = os.path.join(self.temp_dir, 'test_doc', 'p1', 'test_hash_123.png')
            expected_thumb_path = os.path.join(self.temp_dir, 'test_doc', 'p1', 'thumbs', 'test_hash_123.jpg')
            
            assert os.path.normpath(paths['image_path']) == os.path.normpath(expected_image_path)
            assert os.path.normpath(paths['thumbnail_path']) == os.path.normpath(expected_thumb_path)
            
        finally:
            self.tearDown()
    
    def test_cleanup_document_images(self):
        """Test limpieza de imágenes de documento"""
        self.setUp()
        
        try:
            # Crear estructura de directorios de prueba
            doc_dir = os.path.join(self.temp_dir, 'test_doc')
            os.makedirs(doc_dir)
            test_file = os.path.join(doc_dir, 'test.txt')
            with open(test_file, 'w') as f:
                f.write('test')
            
            # Verificar que existe
            assert os.path.exists(doc_dir)
            assert os.path.exists(test_file)
            
            # Limpiar
            result = cleanup_document_images('test_doc', self.temp_dir)
            
            # Verificar limpieza
            assert result == True
            assert not os.path.exists(doc_dir)
            
        finally:
            self.tearDown()

@patch('image_extractor.fitz')
class TestPDFExtraction:
    """Tests para extracción desde PDF usando mocks"""
    
    def test_extract_images_from_pdf_no_images(self, mock_fitz):
        """Test extracción de PDF sin imágenes"""
        # Configurar mock
        mock_doc = MagicMock()
        mock_doc.page_count = 1
        mock_page = MagicMock()
        mock_page.get_images.return_value = []
        mock_doc.load_page.return_value = mock_page
        mock_fitz.open.return_value = mock_doc
        
        # Test con archivo ficticio
        with tempfile.NamedTemporaryFile(suffix='.pdf') as temp_file:
            result = extract_images_from_pdf(temp_file.name)
            
            assert result == []
            mock_fitz.open.assert_called_once()
            mock_doc.close.assert_called_once()
    
    def test_extract_images_from_pdf_with_images(self, mock_fitz):
        """Test extracción de PDF con imágenes"""
        # Crear imagen de prueba
        test_image = Image.new('RGB', (100, 100), color='green')
        img_bytes = BytesIO()
        test_image.save(img_bytes, format='PNG')
        test_image_data = img_bytes.getvalue()
        
        # Configurar mock
        mock_doc = MagicMock()
        mock_doc.page_count = 1
        mock_page = MagicMock()
        mock_page.get_images.return_value = [(123, 0, 100, 100, 8, 'DeviceRGB', '', 'Im1', 'FlateDecode', (0, 0, 100, 100))]
        mock_page.rect.width = 200
        mock_page.rect.height = 200
        mock_doc.load_page.return_value = mock_page
        mock_doc.extract_image.return_value = {
            'image': test_image_data,
            'ext': 'png',
            'width': 100,
            'height': 100
        }
        mock_fitz.open.return_value = mock_doc
        
        # Test
        with tempfile.NamedTemporaryFile(suffix='.pdf') as temp_file:
            result = extract_images_from_pdf(temp_file.name)
            
            assert len(result) == 1
            extracted = result[0]
            assert isinstance(extracted, ExtractedImage)
            assert extracted.format == 'PNG'
            assert extracted.width == 100
            assert extracted.height == 100
            assert extracted.page_number == 1
            assert extracted.image_index == 0

class TestImageExtractorClass:
    """Tests para la clase ImageExtractor"""
    
    def setUp(self):
        """Setup para tests"""
        self.temp_dir = tempfile.mkdtemp()
        self.extractor = ImageExtractor(base_images_dir=self.temp_dir, max_pdf_size_mb=1)
    
    def tearDown(self):
        """Cleanup después de tests"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_extractor_initialization(self):
        """Test inicialización del extractor"""
        extractor = ImageExtractor(base_images_dir='/tmp/test', max_pdf_size_mb=50)
        
        assert str(extractor.base_images_dir) == '/tmp/test'
        assert extractor.max_pdf_size_bytes == 50 * 1024 * 1024
    
    def test_validate_pdf_file_not_exists(self):
        """Test validación de PDF que no existe"""
        self.setUp()
        
        try:
            is_valid, msg = self.extractor.validate_pdf_file('/nonexistent/file.pdf')
            
            assert is_valid == False
            assert 'no encontrado' in msg.lower()
            
        finally:
            self.tearDown()
    
    def test_validate_pdf_file_wrong_extension(self):
        """Test validación de archivo con extensión incorrecta"""
        self.setUp()
        
        try:
            with tempfile.NamedTemporaryFile(suffix='.txt') as temp_file:
                temp_file.write(b'test content')
                temp_file.flush()
                
                is_valid, msg = self.extractor.validate_pdf_file(temp_file.name)
                
                assert is_valid == False
                assert 'PDF' in msg
                
        finally:
            self.tearDown()
    
    def test_validate_pdf_file_too_large(self):
        """Test validación de PDF muy grande"""
        self.setUp()
        
        try:
            # Crear extractor con límite muy pequeño
            small_extractor = ImageExtractor(max_pdf_size_mb=0.001)  # 1KB limit
            
            with tempfile.NamedTemporaryFile(suffix='.pdf') as temp_file:
                # Escribir más de 1KB
                temp_file.write(b'x' * 2000)
                temp_file.flush()
                
                is_valid, msg = small_extractor.validate_pdf_file(temp_file.name)
                
                assert is_valid == False
                assert 'grande' in msg.lower()
                
        finally:
            self.tearDown()
    
    def test_ensure_storage_directory(self):
        """Test creación de directorio de almacenamiento"""
        self.setUp()
        
        try:
            # Eliminar directorio si existe
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
            
            # Intentar asegurar directorio
            self.extractor.ensure_storage_directory()
            
            # Verificar que se creó
            assert os.path.exists(self.temp_dir)
            assert os.path.isdir(self.temp_dir)
            
        finally:
            self.tearDown()

if __name__ == '__main__':
    pytest.main([__file__, '-v'])