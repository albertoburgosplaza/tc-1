"""
Tests unitarios completos para el módulo image_storage.py

Cubre todas las funcionalidades del servicio de almacenamiento local de imágenes:
- Validación de rutas y seguridad
- Almacenamiento y recuperación de imágenes
- Generación de URIs locales
- Control de permisos y ACL
- Limpieza de archivos huérfanos
- Casos edge y manejo de errores
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

# Importar módulos a probar
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from image_storage import (
    ImageStorage, PathValidator, ImageStorageError,
    ALLOWED_EXTENSIONS, DANGEROUS_PATTERNS, MAX_FILENAME_LENGTH
)

class TestPathValidator:
    """Tests para la clase PathValidator y validaciones de seguridad"""
    
    def setUp(self):
        """Setup para cada test"""
        self.temp_dir = tempfile.mkdtemp()
        self.validator = PathValidator(self.temp_dir)
    
    def tearDown(self):
        """Cleanup después de cada test"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_validator_initialization(self):
        """Test inicialización del validator"""
        self.setUp()
        try:
            assert self.validator.base_dir.exists()
            assert self.validator.base_dir.is_dir()
            assert os.access(self.validator.base_dir, os.W_OK)
        finally:
            self.tearDown()
    
    def test_validate_filename_valid(self):
        """Test validación de nombres de archivo válidos"""
        self.setUp()
        try:
            valid_names = [
                'image.png',
                'document.jpg',
                'test_file.webp',
                'hash123abc.jpeg',
                'a' * 64 + '.png'  # Hash SHA-256 típico
            ]
            
            for filename in valid_names:
                assert self.validator.validate_filename(filename) == True
                
        finally:
            self.tearDown()
    
    def test_validate_filename_dangerous_patterns(self):
        """Test rechazo de patrones peligrosos en nombres de archivo"""
        self.setUp()
        try:
            dangerous_names = [
                '../../../etc/passwd',
                '..\\windows\\system32',
                '/etc/passwd',
                '~/.bashrc',
                'file\x00.png',  # Null byte
                'file<test>.png',
                'file|test.png',
                'file:test.png',
                'file"test.png',
                'file?test.png',
                'file*test.png'
            ]
            
            for filename in dangerous_names:
                with pytest.raises(ImageStorageError):
                    self.validator.validate_filename(filename)
                    
        finally:
            self.tearDown()
    
    def test_validate_filename_edge_cases(self):
        """Test casos edge de validación de nombres"""
        self.setUp()
        try:
            # Archivo vacío
            with pytest.raises(ImageStorageError):
                self.validator.validate_filename('')
            
            # Nombre muy largo
            long_name = 'a' * (MAX_FILENAME_LENGTH + 1) + '.png'
            with pytest.raises(ImageStorageError):
                self.validator.validate_filename(long_name)
            
            # Caracteres de control
            with pytest.raises(ImageStorageError):
                self.validator.validate_filename('file\t.png')
                
        finally:
            self.tearDown()
    
    def test_validate_extension_allowed(self):
        """Test validación de extensiones permitidas"""
        self.setUp()
        try:
            allowed_files = [
                'test.png',
                'test.PNG',
                'test.jpg',
                'test.jpeg',
                'test.JPEG',
                'test.webp',
                'test.WEBP'
            ]
            
            for filename in allowed_files:
                assert self.validator.validate_extension(filename) == True
                
        finally:
            self.tearDown()
    
    def test_validate_extension_forbidden(self):
        """Test rechazo de extensiones no permitidas"""
        self.setUp()
        try:
            forbidden_files = [
                'test.bmp',
                'test.gif',
                'test.tiff',
                'test.svg',
                'test.exe',
                'test.txt',
                'test.'  # Extensión vacía
            ]
            
            for filename in forbidden_files:
                with pytest.raises(ImageStorageError):
                    self.validator.validate_extension(filename)
                    
        finally:
            self.tearDown()
    
    def test_validate_doc_id_valid(self):
        """Test validación de doc_ids válidos"""
        self.setUp()
        try:
            valid_ids = [
                'document123',
                'test-file_01',
                'doc.pdf',
                'simple_doc',
                'Document_With_Numbers_123'
            ]
            
            for doc_id in valid_ids:
                assert self.validator.validate_doc_id(doc_id) == True
                
        finally:
            self.tearDown()
    
    def test_validate_doc_id_dangerous(self):
        """Test rechazo de doc_ids peligrosos"""
        self.setUp()
        try:
            dangerous_ids = [
                '../../../etc',
                '/etc/passwd',
                '~root',
                '',  # Vacío
                'a' * 101,  # Muy largo
                'doc with spaces',  # Espacios
                'doc/with/slashes',
                'doc\x00null'  # Null byte
            ]
            
            for doc_id in dangerous_ids:
                with pytest.raises(ImageStorageError):
                    self.validator.validate_doc_id(doc_id)
                    
        finally:
            self.tearDown()
    
    def test_sanitize_path_component(self):
        """Test sanitización de componentes de ruta"""
        self.setUp()
        try:
            test_cases = [
                ('normal_file.png', 'normal_file.png'),
                ('file<>:.png', 'file___.png'),
                ('...file...', 'file'),
                ('file\x00null.png', 'file_null.png'),
                ('a' * 300, 'a' * MAX_FILENAME_LENGTH)
            ]
            
            for input_name, expected in test_cases:
                result = self.validator.sanitize_path_component(input_name)
                assert result == expected
                
        finally:
            self.tearDown()
    
    def test_resolve_safe_path_valid(self):
        """Test resolución de rutas seguras"""
        self.setUp()
        try:
            # Ruta normal
            safe_path = self.validator.resolve_safe_path('doc1', 'p1', 'image.png')
            expected_path = self.validator.base_dir / 'doc1' / 'p1' / 'image.png'
            assert safe_path == expected_path
            
            # Verificar que está dentro del directorio base
            safe_path.relative_to(self.validator.base_dir)
            
        finally:
            self.tearDown()
    
    def test_resolve_safe_path_directory_traversal(self):
        """Test prevención de directory traversal"""
        self.setUp()
        try:
            # Intentos de escapar del directorio base
            dangerous_paths = [
                ('../../../etc', 'passwd'),
                ('..', '..', '..', 'etc', 'passwd'),
                ('normal', '..', '..', 'etc'),
                ('/etc/passwd',),
                ('~root',)
            ]
            
            for path_components in dangerous_paths:
                with pytest.raises(ImageStorageError):
                    self.validator.resolve_safe_path(*path_components)
                    
        finally:
            self.tearDown()

class TestImageStorage:
    """Tests para la clase principal ImageStorage"""
    
    def setUp(self):
        """Setup para cada test"""
        self.temp_dir = tempfile.mkdtemp()
        self.storage = ImageStorage(self.temp_dir)
        
    def tearDown(self):
        """Cleanup después de cada test"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def create_test_image(self, width=100, height=100, format='PNG'):
        """Crea imagen de prueba en memoria"""
        image = Image.new('RGB', (width, height), color='red')
        buffer = BytesIO()
        image.save(buffer, format=format)
        return buffer.getvalue()
    
    def test_storage_initialization(self):
        """Test inicialización del storage"""
        self.setUp()
        try:
            assert self.storage.base_dir.exists()
            assert self.storage.validator is not None
            
        finally:
            self.tearDown()
    
    def test_get_document_path(self):
        """Test obtención de ruta de documento"""
        self.setUp()
        try:
            doc_path = self.storage.get_document_path('test_doc')
            expected_path = self.storage.base_dir / 'test_doc'
            assert doc_path == expected_path
            
        finally:
            self.tearDown()
    
    def test_get_page_path(self):
        """Test obtención de ruta de página"""
        self.setUp()
        try:
            page_path = self.storage.get_page_path('test_doc', 1)
            expected_path = self.storage.base_dir / 'test_doc' / 'p1'
            assert page_path == expected_path
            
            # Test números de página inválidos
            with pytest.raises(ImageStorageError):
                self.storage.get_page_path('test_doc', 0)
                
            with pytest.raises(ImageStorageError):
                self.storage.get_page_path('test_doc', 10000)
                
        finally:
            self.tearDown()
    
    def test_get_image_path(self):
        """Test obtención de ruta de imagen"""
        self.setUp()
        try:
            image_path = self.storage.get_image_path('test_doc', 1, 'hash123', 'png')
            expected_path = self.storage.base_dir / 'test_doc' / 'p1' / 'hash123.png'
            assert image_path == expected_path
            
        finally:
            self.tearDown()
    
    def test_get_thumbnail_path(self):
        """Test obtención de ruta de thumbnail"""
        self.setUp()
        try:
            thumb_path = self.storage.get_thumbnail_path('test_doc', 1, 'hash123')
            expected_path = self.storage.base_dir / 'test_doc' / 'p1' / 'thumbs' / 'hash123.jpg'
            assert thumb_path == expected_path
            
        finally:
            self.tearDown()
    
    def test_generate_local_uri(self):
        """Test generación de URIs locales"""
        self.setUp()
        try:
            # URI de imagen
            image_uri = self.storage.generate_local_uri('test_doc', 1, 'hash123', 'png')
            expected_uri = 'local://images/test_doc/p1/hash123.png'
            assert image_uri == expected_uri
            
            # URI de thumbnail
            thumb_uri = self.storage.generate_local_uri('test_doc', 1, 'hash123', 'png', is_thumbnail=True)
            expected_uri = 'local://images/test_doc/p1/thumbs/hash123.jpg'
            assert thumb_uri == expected_uri
            
        finally:
            self.tearDown()
    
    def test_create_directory_structure(self):
        """Test creación de estructura de directorios"""
        self.setUp()
        try:
            page_path, thumbs_path = self.storage.create_directory_structure('test_doc', 1)
            
            # Verificar que los directorios se crearon
            assert page_path.exists()
            assert page_path.is_dir()
            assert thumbs_path.exists()
            assert thumbs_path.is_dir()
            
            # Verificar permisos
            assert oct(page_path.stat().st_mode)[-3:] == '755'
            assert oct(thumbs_path.stat().st_mode)[-3:] == '755'
            
        finally:
            self.tearDown()
    
    def test_save_image_basic(self):
        """Test guardado básico de imagen"""
        self.setUp()
        try:
            test_image_data = self.create_test_image()
            test_thumbnail_data = self.create_test_image(50, 50)
            
            result = self.storage.save_image(
                doc_id='test_doc',
                page_number=1,
                image_hash='hash123abc',
                image_data=test_image_data,
                extension='png',
                thumbnail_data=test_thumbnail_data
            )
            
            # Verificar resultado
            assert 'image_path' in result
            assert 'image_uri' in result
            assert 'thumbnail_path' in result
            assert 'thumbnail_uri' in result
            
            # Verificar que los archivos existen
            assert Path(result['image_path']).exists()
            assert Path(result['thumbnail_path']).exists()
            
            # Verificar URIs
            assert result['image_uri'] == 'local://images/test_doc/p1/hash123abc.png'
            assert result['thumbnail_uri'] == 'local://images/test_doc/p1/thumbs/hash123abc.jpg'
            
        finally:
            self.tearDown()
    
    def test_save_image_without_thumbnail(self):
        """Test guardado de imagen sin thumbnail"""
        self.setUp()
        try:
            test_image_data = self.create_test_image()
            
            result = self.storage.save_image(
                doc_id='test_doc',
                page_number=2,
                image_hash='hash456def',
                image_data=test_image_data,
                extension='jpg'
            )
            
            # Verificar que no hay thumbnail
            assert result['thumbnail_path'] is None
            assert result['thumbnail_uri'] is None
            
            # Pero sí hay imagen
            assert result['image_path'] is not None
            assert Path(result['image_path']).exists()
            
        finally:
            self.tearDown()
    
    def test_save_image_invalid_data(self):
        """Test guardado con datos inválidos"""
        self.setUp()
        try:
            # Datos vacíos
            with pytest.raises(ImageStorageError):
                self.storage.save_image('test_doc', 1, 'hash', b'', 'png')
            
            # Doc ID inválido
            with pytest.raises(ImageStorageError):
                self.storage.save_image('../etc', 1, 'hash', b'data', 'png')
                
        finally:
            self.tearDown()
    
    def test_image_exists(self):
        """Test verificación de existencia de imágenes"""
        self.setUp()
        try:
            # Imagen que no existe
            assert self.storage.image_exists('test_doc', 1, 'hash123', 'png') == False
            
            # Crear imagen
            test_data = self.create_test_image()
            self.storage.save_image('test_doc', 1, 'hash123', test_data, 'png')
            
            # Ahora debe existir
            assert self.storage.image_exists('test_doc', 1, 'hash123', 'png') == True
            
            # Con extensión incorrecta no debe existir
            assert self.storage.image_exists('test_doc', 1, 'hash123', 'jpg') == False
            
        finally:
            self.tearDown()
    
    def test_thumbnail_exists(self):
        """Test verificación de existencia de thumbnails"""
        self.setUp()
        try:
            # Thumbnail que no existe
            assert self.storage.thumbnail_exists('test_doc', 1, 'hash123') == False
            
            # Crear imagen con thumbnail
            test_data = self.create_test_image()
            thumb_data = self.create_test_image(50, 50)
            self.storage.save_image('test_doc', 1, 'hash123', test_data, 'png', thumb_data)
            
            # Ahora debe existir
            assert self.storage.thumbnail_exists('test_doc', 1, 'hash123') == True
            
        finally:
            self.tearDown()
    
    def test_resolve_uri_to_path(self):
        """Test resolución de URI a path"""
        self.setUp()
        try:
            # Crear imagen
            test_data = self.create_test_image()
            thumb_data = self.create_test_image(50, 50)
            result = self.storage.save_image('test_doc', 1, 'hash123', test_data, 'png', thumb_data)
            
            # Resolver URI de imagen
            image_path = self.storage.resolve_uri_to_path(result['image_uri'])
            assert image_path is not None
            assert image_path.exists()
            assert str(image_path) == result['image_path']
            
            # Resolver URI de thumbnail
            thumb_path = self.storage.resolve_uri_to_path(result['thumbnail_uri'])
            assert thumb_path is not None
            assert thumb_path.exists()
            
            # URI inválida
            invalid_path = self.storage.resolve_uri_to_path('invalid://uri')
            assert invalid_path is None
            
        finally:
            self.tearDown()
    
    def test_get_image_info(self):
        """Test obtención de información de imagen"""
        self.setUp()
        try:
            # Imagen que no existe
            info = self.storage.get_image_info('test_doc', 1, 'hash123')
            assert info is None
            
            # Crear imagen
            test_data = self.create_test_image(200, 150)
            thumb_data = self.create_test_image(50, 50)
            self.storage.save_image('test_doc', 1, 'hash123', test_data, 'png', thumb_data)
            
            # Obtener información
            info = self.storage.get_image_info('test_doc', 1, 'hash123', 'png')
            
            assert info is not None
            assert info['doc_id'] == 'test_doc'
            assert info['page_number'] == 1
            assert info['hash'] == 'hash123'
            assert info['format'] == 'PNG'
            assert info['exists'] == True
            assert info['has_thumbnail'] == True
            assert info['has_dimensions'] == True
            assert info['width'] == 200
            assert info['height'] == 150
            
        finally:
            self.tearDown()
    
    def test_file_exists_with_uri(self):
        """Test file_exists con URI"""
        self.setUp()
        try:
            # Crear imagen
            test_data = self.create_test_image()
            result = self.storage.save_image('test_doc', 1, 'hash123', test_data, 'png')
            
            # Verificar con URI
            assert self.storage.file_exists(result['image_uri']) == True
            
            # URI que no existe
            fake_uri = 'local://images/test_doc/p1/nonexistent.png'
            assert self.storage.file_exists(fake_uri) == False
            
        finally:
            self.tearDown()
    
    def test_file_exists_with_path(self):
        """Test file_exists con path"""
        self.setUp()
        try:
            # Crear imagen
            test_data = self.create_test_image()
            result = self.storage.save_image('test_doc', 1, 'hash123', test_data, 'png')
            
            # Verificar con path
            assert self.storage.file_exists(result['image_path']) == True
            
            # Path que no existe
            fake_path = self.storage.base_dir / 'nonexistent.png'
            assert self.storage.file_exists(fake_path) == False
            
        finally:
            self.tearDown()
    
    def test_get_document_images(self):
        """Test obtención de imágenes de documento"""
        self.setUp()
        try:
            # Documento sin imágenes
            images = self.storage.get_document_images('test_doc')
            assert len(images) == 0
            
            # Crear varias imágenes
            test_data = self.create_test_image()
            thumb_data = self.create_test_image(50, 50)
            
            self.storage.save_image('test_doc', 1, 'hash1', test_data, 'png', thumb_data)
            self.storage.save_image('test_doc', 1, 'hash2', test_data, 'jpg')
            self.storage.save_image('test_doc', 2, 'hash3', test_data, 'webp', thumb_data)
            
            # Obtener todas las imágenes
            images = self.storage.get_document_images('test_doc')
            
            assert len(images) == 3
            
            # Verificar ordenamiento por página y hash
            assert images[0]['page_number'] <= images[1]['page_number'] <= images[2]['page_number']
            
            # Verificar que cada imagen tiene la información esperada
            for img in images:
                assert 'doc_id' in img
                assert 'page_number' in img
                assert 'hash' in img
                assert 'format' in img
                assert 'image_path' in img
                assert 'image_uri' in img
                
        finally:
            self.tearDown()
    
    def test_cleanup_document(self):
        """Test limpieza de documento"""
        self.setUp()
        try:
            # Crear imágenes
            test_data = self.create_test_image()
            self.storage.save_image('test_doc', 1, 'hash1', test_data, 'png')
            self.storage.save_image('test_doc', 2, 'hash2', test_data, 'jpg')
            
            # Verificar que existen
            assert len(self.storage.get_document_images('test_doc')) == 2
            
            # Limpiar documento
            result = self.storage.cleanup_document('test_doc')
            assert result == True
            
            # Verificar que no hay imágenes
            assert len(self.storage.get_document_images('test_doc')) == 0
            
        finally:
            self.tearDown()
    
    def test_save_images_batch(self):
        """Test guardado en lote de imágenes"""
        self.setUp()
        try:
            test_data = self.create_test_image()
            thumb_data = self.create_test_image(50, 50)
            
            images_batch = [
                {
                    'doc_id': 'batch_doc',
                    'page_number': 1,
                    'image_hash': 'hash1',
                    'image_data': test_data,
                    'extension': 'png',
                    'thumbnail_data': thumb_data
                },
                {
                    'doc_id': 'batch_doc',
                    'page_number': 1,
                    'image_hash': 'hash2',
                    'image_data': test_data,
                    'extension': 'jpg'
                },
                {
                    'doc_id': 'batch_doc',
                    'page_number': 2,
                    'image_hash': 'hash3',
                    'image_data': test_data,
                    'extension': 'webp',
                    'thumbnail_data': thumb_data
                }
            ]
            
            results = self.storage.save_images_batch(images_batch)
            
            # Verificar que se guardaron todas
            assert len(results) == 3
            
            # Verificar que no hay errores
            errors = [r for r in results if 'error' in r]
            assert len(errors) == 0
            
            # Verificar que se crearon los archivos
            for result in results:
                assert Path(result['image_path']).exists()
                
        finally:
            self.tearDown()
    
    def test_ensure_permissions(self):
        """Test establecimiento de permisos"""
        self.setUp()
        try:
            # Crear imagen
            test_data = self.create_test_image()
            self.storage.save_image('test_doc', 1, 'hash1', test_data, 'png')
            
            # Establecer permisos
            result = self.storage.ensure_permissions('test_doc')
            assert result == True
            
            # Verificar permisos de directorio
            doc_path = self.storage.get_document_path('test_doc')
            assert oct(doc_path.stat().st_mode)[-3:] == '755'
            
        finally:
            self.tearDown()
    
    def test_cleanup_orphaned_files(self):
        """Test limpieza de archivos huérfanos"""
        self.setUp()
        try:
            # Crear estructura válida
            test_data = self.create_test_image()
            self.storage.save_image('valid_doc', 1, 'hash1', test_data, 'png')
            
            # Crear archivos huérfanos manualmente
            orphan_dir = self.storage.base_dir / 'invalid_doc'
            orphan_dir.mkdir()
            orphan_file = orphan_dir / 'invalid_file.txt'
            orphan_file.write_text('orphan content')
            
            # También crear archivo huérfano en ubicación incorrecta
            bad_file = self.storage.base_dir / 'bad_file.png'
            bad_file.write_bytes(test_data)
            
            # Ejecutar limpieza
            stats = self.storage.cleanup_orphaned_files()
            
            # Verificar estadísticas
            assert stats['orphaned_files'] >= 1
            assert stats['orphaned_dirs'] >= 1
            
            # Verificar que los archivos válidos siguen ahí
            assert len(self.storage.get_document_images('valid_doc')) == 1
            
            # Verificar que los huérfanos fueron eliminados
            assert not orphan_dir.exists()
            assert not bad_file.exists()
            
        finally:
            self.tearDown()
    
    def test_get_storage_stats(self):
        """Test obtención de estadísticas de almacenamiento"""
        self.setUp()
        try:
            # Stats iniciales (vacío)
            stats = self.storage.get_storage_stats()
            assert stats['total_documents'] == 0
            assert stats['total_images'] == 0
            assert stats['total_thumbnails'] == 0
            assert stats['storage_size_bytes'] == 0
            
            # Crear imágenes
            test_data = self.create_test_image()
            thumb_data = self.create_test_image(50, 50)
            
            self.storage.save_image('doc1', 1, 'hash1', test_data, 'png', thumb_data)
            self.storage.save_image('doc1', 2, 'hash2', test_data, 'jpg')
            self.storage.save_image('doc2', 1, 'hash3', test_data, 'webp', thumb_data)
            
            # Stats después de crear imágenes
            stats = self.storage.get_storage_stats()
            assert stats['total_documents'] == 2
            assert stats['total_images'] == 3
            assert stats['total_thumbnails'] == 2  # Solo hash1 y hash3 tienen thumbnails
            assert stats['storage_size_bytes'] > 0
            
        finally:
            self.tearDown()

class TestSecurityValidations:
    """Tests específicos para validaciones de seguridad"""
    
    def setUp(self):
        """Setup para cada test"""
        self.temp_dir = tempfile.mkdtemp()
        self.storage = ImageStorage(self.temp_dir)
        
    def tearDown(self):
        """Cleanup después de cada test"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_directory_traversal_attacks(self):
        """Test prevención de ataques directory traversal"""
        self.setUp()
        try:
            test_data = b'fake image data'
            
            # Intentos de directory traversal en doc_id
            malicious_doc_ids = [
                '../../../etc',
                '..\\..\\windows',
                '/etc/passwd',
                '~root'
            ]
            
            for bad_doc_id in malicious_doc_ids:
                with pytest.raises(ImageStorageError):
                    self.storage.save_image(bad_doc_id, 1, 'hash', test_data, 'png')
                    
        finally:
            self.tearDown()
    
    def test_null_byte_injection(self):
        """Test prevención de inyección de null bytes"""
        self.setUp()
        try:
            test_data = b'fake image data'
            
            # Null bytes en diferentes parámetros
            with pytest.raises(ImageStorageError):
                self.storage.save_image('doc\x00id', 1, 'hash', test_data, 'png')
                
        finally:
            self.tearDown()
    
    def test_unicode_and_special_characters(self):
        """Test manejo de caracteres Unicode y especiales"""
        self.setUp()
        try:
            test_data = b'fake image data'
            
            # Algunos caracteres Unicode válidos en doc_id
            unicode_doc_id = 'document_123'  # Solo ASCII seguro
            
            result = self.storage.save_image(unicode_doc_id, 1, 'hash123', test_data, 'png')
            assert result is not None
            
            # Caracteres problemáticos deben ser rechazados
            with pytest.raises(ImageStorageError):
                self.storage.save_image('doc with spaces', 1, 'hash', test_data, 'png')
                
        finally:
            self.tearDown()
    
    def test_path_length_limits(self):
        """Test límites de longitud de rutas"""
        self.setUp()
        try:
            test_data = b'fake image data'
            
            # Doc ID muy largo
            long_doc_id = 'a' * 101
            with pytest.raises(ImageStorageError):
                self.storage.save_image(long_doc_id, 1, 'hash', test_data, 'png')
                
            # Hash muy largo (normal para SHA-256 es 64 chars)
            normal_hash = 'a' * 64
            result = self.storage.save_image('doc', 1, normal_hash, test_data, 'png')
            assert result is not None
            
        finally:
            self.tearDown()

class TestErrorHandling:
    """Tests para manejo de errores y casos edge"""
    
    def setUp(self):
        """Setup para cada test"""
        self.temp_dir = tempfile.mkdtemp()
        self.storage = ImageStorage(self.temp_dir)
        
    def tearDown(self):
        """Cleanup después de cada test"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('builtins.open')
    def test_permission_error_handling(self, mock_open):
        """Test manejo de errores de permisos"""
        self.setUp()
        try:
            # Simular error de permisos
            mock_open.side_effect = PermissionError("Access denied")
            
            test_data = b'fake image data'
            
            with pytest.raises(ImageStorageError) as exc_info:
                self.storage.save_image('doc', 1, 'hash', test_data, 'png')
            
            assert "permisos" in str(exc_info.value).lower()
            
        finally:
            self.tearDown()
    
    @patch('builtins.open')
    def test_disk_space_error_handling(self, mock_open):
        """Test manejo de errores de espacio en disco"""
        self.setUp()
        try:
            # Simular error de espacio en disco
            mock_open.side_effect = OSError("No space left on device")
            
            test_data = b'fake image data'
            
            with pytest.raises(ImageStorageError) as exc_info:
                self.storage.save_image('doc', 1, 'hash', test_data, 'png')
            
            assert "filesystem" in str(exc_info.value).lower()
            
        finally:
            self.tearDown()
    
    def test_invalid_page_numbers(self):
        """Test manejo de números de página inválidos"""
        self.setUp()
        try:
            invalid_pages = [0, -1, 10000, 'invalid', None]
            
            for page_num in invalid_pages:
                with pytest.raises((ImageStorageError, TypeError)):
                    if page_num is None:
                        self.storage.get_page_path('doc', page_num)
                    else:
                        self.storage.get_page_path('doc', page_num)
                        
        finally:
            self.tearDown()
    
    def test_corrupted_image_data(self):
        """Test manejo de datos de imagen corruptos"""
        self.setUp()
        try:
            # Datos corruptos pero válidos para guardado
            corrupted_data = b'not an image but valid data'
            
            # El guardado debe funcionar (es solo I/O de bytes)
            result = self.storage.save_image('doc', 1, 'hash', corrupted_data, 'png')
            assert result is not None
            
            # Pero get_image_info puede fallar al leer dimensiones
            info = self.storage.get_image_info('doc', 1, 'hash', 'png')
            assert info is not None
            assert info['has_dimensions'] == False  # No se pudieron obtener dimensiones
            
        finally:
            self.tearDown()

if __name__ == '__main__':
    # Ejecutar tests con pytest
    pytest.main([__file__, '-v', '--tb=short'])