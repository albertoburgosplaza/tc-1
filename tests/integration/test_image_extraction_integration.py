"""
Tests de integración para el módulo image_extractor.py
Estos tests validan el flujo end-to-end con archivos PDF reales
"""

import pytest
import os
import tempfile
import shutil
from pathlib import Path

# Importar módulo a probar
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from image_extractor import ImageExtractor, ImageExtractionError

class TestImageExtractionIntegration:
    """Tests de integración end-to-end para extracción de imágenes"""
    
    def setUp(self):
        """Setup para cada test"""
        self.temp_dir = tempfile.mkdtemp()
        self.extractor = ImageExtractor(base_images_dir=self.temp_dir)
        self.docs_dir = Path(__file__).parent.parent.parent / 'docs'
    
    def tearDown(self):
        """Cleanup después de cada test"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_process_pdf_with_real_documents(self):
        """Test procesamiento con documentos PDF reales del directorio docs/"""
        self.setUp()
        
        try:
            # Buscar PDFs en el directorio docs
            if not self.docs_dir.exists():
                pytest.skip("Directorio docs/ no encontrado")
            
            pdf_files = list(self.docs_dir.glob("*.pdf"))
            if not pdf_files:
                pytest.skip("No hay archivos PDF en docs/")
            
            # Probar con el primer PDF encontrado
            test_pdf = pdf_files[0]
            doc_id = test_pdf.stem
            
            # Procesar PDF
            result = self.extractor.process_pdf(str(test_pdf), doc_id=doc_id)
            
            # Validar resultado
            assert isinstance(result, dict)
            assert result['success'] == True
            assert result['doc_id'] == doc_id
            assert 'images_found' in result
            assert 'images_saved' in result
            assert 'processing_time' in result
            assert 'saved_images' in result
            
            # Si se encontraron imágenes, validar estructura
            if result['images_found'] > 0:
                assert result['images_saved'] == result['images_found']
                assert len(result['saved_images']) == result['images_found']
                
                # Validar cada imagen guardada
                for img_info in result['saved_images']:
                    assert 'doc_id' in img_info
                    assert 'page_number' in img_info
                    assert 'hash' in img_info
                    assert 'image_path' in img_info
                    assert 'thumbnail_path' in img_info
                    assert 'image_uri' in img_info
                    assert 'thumbnail_uri' in img_info
                    
                    # Verificar que archivos existen
                    assert os.path.exists(img_info['image_path'])
                    assert os.path.exists(img_info['thumbnail_path'])
                    
                    # Verificar estructura de directorios
                    expected_pattern = f"{doc_id}/p{img_info['page_number']}/"
                    assert expected_pattern in img_info['image_path']
                    assert f"{expected_pattern}thumbs/" in img_info['thumbnail_path']
            
            print(f"✅ Procesado {test_pdf.name}: {result['images_found']} imágenes encontradas")
            
        finally:
            self.tearDown()
    
    def test_process_multiple_pdfs(self):
        """Test procesamiento de múltiples PDFs"""
        self.setUp()
        
        try:
            if not self.docs_dir.exists():
                pytest.skip("Directorio docs/ no encontrado")
            
            pdf_files = list(self.docs_dir.glob("*.pdf"))[:3]  # Máximo 3 PDFs para test
            if not pdf_files:
                pytest.skip("No hay archivos PDF en docs/")
            
            total_images = 0
            
            for pdf_file in pdf_files:
                doc_id = pdf_file.stem
                result = self.extractor.process_pdf(str(pdf_file), doc_id=doc_id)
                
                assert result['success'] == True
                total_images += result['images_found']
                
                print(f"✅ {pdf_file.name}: {result['images_found']} imágenes")
            
            print(f"🎯 Total de imágenes procesadas: {total_images}")
            
        finally:
            self.tearDown()
    
    def test_get_document_images_after_processing(self):
        """Test obtener imágenes de documento después del procesamiento"""
        self.setUp()
        
        try:
            if not self.docs_dir.exists():
                pytest.skip("Directorio docs/ no encontrado")
            
            pdf_files = list(self.docs_dir.glob("*.pdf"))
            if not pdf_files:
                pytest.skip("No hay archivos PDF en docs/")
            
            # Procesar primer PDF
            test_pdf = pdf_files[0]
            doc_id = test_pdf.stem
            
            result = self.extractor.process_pdf(str(test_pdf), doc_id=doc_id)
            
            # Obtener imágenes almacenadas
            stored_images = self.extractor.get_document_images(doc_id)
            
            # Validar que coincide con lo procesado
            assert len(stored_images) == result['images_saved']
            
            # Validar estructura de cada imagen
            for img_info in stored_images:
                assert 'doc_id' in img_info
                assert 'page_number' in img_info  
                assert 'hash' in img_info
                assert 'format' in img_info
                assert 'image_path' in img_info
                assert 'thumbnail_path' in img_info or img_info['thumbnail_path'] is None
                assert 'image_uri' in img_info
                assert 'thumbnail_uri' in img_info or img_info['thumbnail_uri'] is None
                
                assert img_info['doc_id'] == doc_id
                assert os.path.exists(img_info['image_path'])
            
        finally:
            self.tearDown()
    
    def test_cleanup_and_reprocess(self):
        """Test limpieza y reprocesamiento de documento"""
        self.setUp()
        
        try:
            if not self.docs_dir.exists():
                pytest.skip("Directorio docs/ no encontrado")
            
            pdf_files = list(self.docs_dir.glob("*.pdf"))
            if not pdf_files:
                pytest.skip("No hay archivos PDF en docs/")
            
            test_pdf = pdf_files[0]
            doc_id = test_pdf.stem
            
            # Primer procesamiento
            result1 = self.extractor.process_pdf(str(test_pdf), doc_id=doc_id)
            images_after_first = self.extractor.get_document_images(doc_id)
            
            # Segundo procesamiento (debe limpiar automáticamente)
            result2 = self.extractor.process_pdf(str(test_pdf), doc_id=doc_id)
            images_after_second = self.extractor.get_document_images(doc_id)
            
            # Validar que el segundo procesamiento funciona igual
            assert result1['images_found'] == result2['images_found']
            assert len(images_after_first) == len(images_after_second)
            
            # Limpieza manual
            cleanup_result = self.extractor.cleanup_document(doc_id)
            assert cleanup_result == True
            
            # Verificar que no hay imágenes después de limpieza
            images_after_cleanup = self.extractor.get_document_images(doc_id)
            assert len(images_after_cleanup) == 0
            
        finally:
            self.tearDown()
    
    def test_error_handling_with_invalid_pdf(self):
        """Test manejo de errores con PDF inválido"""
        self.setUp()
        
        try:
            # Test con archivo inexistente
            with pytest.raises(ImageExtractionError):
                self.extractor.process_pdf("/nonexistent/file.pdf")
            
            # Test con archivo no PDF
            with tempfile.NamedTemporaryFile(suffix='.txt') as temp_file:
                temp_file.write(b'not a pdf')
                temp_file.flush()
                
                with pytest.raises(ImageExtractionError):
                    self.extractor.process_pdf(temp_file.name)
            
        finally:
            self.tearDown()
    
    def test_process_pdf_without_saving(self):
        """Test procesamiento sin guardar archivos (solo extracción)"""
        self.setUp()
        
        try:
            if not self.docs_dir.exists():
                pytest.skip("Directorio docs/ no encontrado")
            
            pdf_files = list(self.docs_dir.glob("*.pdf"))
            if not pdf_files:
                pytest.skip("No hay archivos PDF en docs/")
            
            test_pdf = pdf_files[0]
            doc_id = test_pdf.stem
            
            # Procesar sin guardar
            result = self.extractor.process_pdf(str(test_pdf), doc_id=doc_id, save_images=False)
            
            # Validar resultado
            assert result['success'] == True
            assert 'saved_images' in result
            
            # No deben existir archivos guardados
            doc_dir = Path(self.temp_dir) / doc_id
            assert not doc_dir.exists()
            
            # Pero debe haber metadata de imágenes procesadas
            if result['images_found'] > 0:
                assert len(result['saved_images']) == result['images_found']
                for img_info in result['saved_images']:
                    # Metadata debe estar presente pero no rutas de archivo
                    assert 'doc_id' in img_info
                    assert 'hash' in img_info
                    assert 'format' in img_info
                    assert 'image_path' not in img_info or not os.path.exists(img_info.get('image_path', ''))
            
        finally:
            self.tearDown()

class TestPerformanceAndStress:
    """Tests de rendimiento y estrés"""
    
    def setUp(self):
        """Setup para tests de rendimiento"""
        self.temp_dir = tempfile.mkdtemp()
        self.extractor = ImageExtractor(base_images_dir=self.temp_dir)
        self.docs_dir = Path(__file__).parent.parent.parent / 'docs'
    
    def tearDown(self):
        """Cleanup después de tests"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_processing_time_reasonable(self):
        """Test que el tiempo de procesamiento sea razonable"""
        self.setUp()
        
        try:
            if not self.docs_dir.exists():
                pytest.skip("Directorio docs/ no encontrado")
            
            pdf_files = list(self.docs_dir.glob("*.pdf"))
            if not pdf_files:
                pytest.skip("No hay archivos PDF en docs/")
            
            test_pdf = pdf_files[0]
            doc_id = test_pdf.stem
            
            result = self.extractor.process_pdf(str(test_pdf), doc_id=doc_id)
            
            processing_time = result['processing_time']
            images_count = result['images_found']
            
            # Tiempo razonable: menos de 5 segundos por imagen o 30 segundos total
            max_time = max(5.0 * max(1, images_count), 30.0)
            
            assert processing_time < max_time, f"Procesamiento muy lento: {processing_time:.2f}s para {images_count} imágenes"
            
            print(f"⏱️ Tiempo de procesamiento: {processing_time:.2f}s para {images_count} imágenes")
            
        finally:
            self.tearDown()

if __name__ == '__main__':
    pytest.main([__file__, '-v'])