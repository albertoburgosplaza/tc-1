"""
Tests de integración para el sistema de observabilidad y métricas multimodal
"""

import pytest
import time
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import requests
from io import BytesIO

from image_extractor import ImageExtractor, ImageExtractionMetrics, image_extraction_metrics
from embedding_factory import EmbeddingFactory, EmbeddingMetrics, embedding_metrics
from image_storage import ImageStorage, ImageStorageMetrics, image_storage_metrics


class TestMultimodalObservabilityIntegration:
    """Tests de integración para observabilidad multimodal end-to-end"""
    
    @pytest.fixture
    def temp_storage_dir(self):
        """Crea directorio temporal para tests"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def sample_image_data(self):
        """Genera datos de imagen de muestra para tests"""
        from PIL import Image
        
        # Crear imagen simple de 100x100 
        img = Image.new('RGB', (100, 100), color='red')
        img_bytes = BytesIO()
        img.save(img_bytes, format='JPEG')
        return img_bytes.getvalue()
    
    @pytest.fixture
    def reset_all_metrics(self):
        """Fixture que resetea todas las métricas antes de cada test"""
        image_extraction_metrics.reset()
        embedding_metrics.reset()
        image_storage_metrics.reset()
        yield
        # Reset después del test también para limpieza
        image_extraction_metrics.reset()
        embedding_metrics.reset()
        image_storage_metrics.reset()
    
    def test_end_to_end_metrics_collection(self, temp_storage_dir, sample_image_data, reset_all_metrics):
        """Test que las métricas se colectan correctamente en flujo end-to-end"""
        
        # 1. Test métricas de almacenamiento de imágenes
        storage = ImageStorage(temp_storage_dir)
        
        # Simular guardado de imagen
        start_time = time.time()
        storage.save_image(
            doc_id="test_doc",
            page_number=1,
            image_hash="abcd1234" * 8,  # Hash SHA-256 válido (64 chars)
            image_data=sample_image_data,
            extension="jpg",
            thumbnail_data=sample_image_data  # Usar misma imagen como thumbnail
        )
        processing_time = time.time() - start_time
        
        # Verificar que se registraron métricas de almacenamiento
        storage_metrics = image_storage_metrics.get_metrics_summary()
        assert storage_metrics['storage_summary']['images_stored'] == 1
        assert storage_metrics['storage_summary']['thumbnails_stored'] == 1
        assert storage_metrics['storage_summary']['total_storage_operations'] == 1
        assert storage_metrics['storage_summary']['storage_errors'] == 0
        assert storage_metrics['performance_metrics']['total_bytes_stored'] > 0
        
        # 2. Test métricas de embeddings (simulado)
        with patch('jina_embeddings.JinaEmbeddings') as mock_jina:
            # Simular respuesta de Jina
            mock_instance = Mock()
            mock_instance.embed_images_data.return_value = [[0.1] * 1024]  # Vector de 1024 dims
            mock_jina.return_value = mock_instance
            
            # Simular creación de embeddings
            factory = EmbeddingFactory()
            embeddings = factory.embed_images_flexible([sample_image_data])
            
            # Verificar que se registraron métricas de embeddings
            embedding_metrics_summary = embedding_metrics.get_metrics_summary()
            assert embedding_metrics_summary['embedding_summary']['total_image_embeddings'] == 1
            assert embedding_metrics_summary['embedding_summary']['image_embedding_errors'] == 0
            assert embedding_metrics_summary['performance_metrics']['total_embedding_time_seconds'] > 0
    
    def test_app_status_endpoint_includes_multimodal_metrics(self, reset_all_metrics):
        """Test que el endpoint /status incluye métricas multimodales"""
        
        # Agregar datos de prueba a las métricas
        image_extraction_metrics.record_pdf_processing(5, 4, 1.5)
        embedding_metrics.record_image_embedding_batch(4, 0.8, True)
        image_storage_metrics.record_image_storage(True, 0.3, 1024, False)
        
        # Importar función de status de app.py
        from app import get_app_status
        
        # Obtener status
        status = get_app_status()
        
        # Verificar que incluye sección multimodal
        assert 'multimodal' in status
        assert status['multimodal']['multimodal_enabled'] is True
        
        # Verificar que incluye métricas de extracción
        assert 'image_extraction' in status['multimodal']
        extraction_metrics = status['multimodal']['image_extraction']
        assert extraction_metrics['extraction_summary']['pdfs_processed'] == 1
        assert extraction_metrics['extraction_summary']['total_images_found'] == 5
        
        # Verificar que incluye métricas de embeddings
        assert 'image_embeddings' in status['multimodal']
        embedding_metrics_summary = status['multimodal']['image_embeddings']
        assert embedding_metrics_summary['embedding_summary']['total_image_embeddings'] == 4
        
        # Verificar que incluye métricas de almacenamiento
        assert 'image_storage' in status['multimodal']
        storage_metrics_summary = status['multimodal']['image_storage']
        assert storage_metrics_summary['storage_summary']['images_stored'] == 1
    
    @patch('requests.get')
    def test_status_endpoint_http_response(self, mock_get, reset_all_metrics):
        """Test que el endpoint HTTP /status responde correctamente con métricas multimodales"""
        
        # Agregar datos de prueba
        image_extraction_metrics.record_pdf_processing(3, 2, 0.8)
        
        # Simular request HTTP al endpoint /status
        from app import HealthHandler
        from unittest.mock import MagicMock
        
        # Crear mock handler
        handler = HealthHandler()
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler.wfile = MagicMock()
        handler.path = "/status"
        
        # Ejecutar handler
        handler.do_GET()
        
        # Verificar que se envió respuesta correcta
        handler.send_response.assert_called_with(200)
        handler.send_header.assert_called_with('Content-Type', 'application/json')
        handler.end_headers.assert_called_once()
        
        # Verificar que se escribió JSON al response
        assert handler.wfile.write.called
        
        # Deserializar y verificar estructura del JSON
        json_data = handler.wfile.write.call_args[0][0].decode('utf-8')
        status_response = json.loads(json_data)
        
        assert 'multimodal' in status_response
        assert 'metrics' in status_response
        assert 'services' in status_response
        assert 'timestamp' in status_response
    
    def test_metrics_persistence_across_operations(self, temp_storage_dir, sample_image_data, reset_all_metrics):
        """Test que las métricas persisten correctamente a través de múltiples operaciones"""
        
        storage = ImageStorage(temp_storage_dir)
        
        # Realizar múltiples operaciones de almacenamiento
        doc_ids = ["doc1", "doc2", "doc3"]
        for i, doc_id in enumerate(doc_ids):
            storage.save_image(
                doc_id=doc_id,
                page_number=i + 1,
                image_hash=f"hash{i:08d}" + "0" * 56,  # Hash SHA-256 válido
                image_data=sample_image_data,
                extension="jpg"
            )
        
        # Verificar acumulación de métricas
        storage_metrics = image_storage_metrics.get_metrics_summary()
        assert storage_metrics['storage_summary']['images_stored'] == 3
        assert storage_metrics['storage_summary']['total_storage_operations'] == 3
        assert storage_metrics['performance_metrics']['total_bytes_stored'] > 0
        
        # Registrar más operaciones y verificar que se acumulan
        storage.save_image(
            doc_id="doc4",
            page_number=1,
            image_hash="hash0004" + "0" * 56,
            image_data=sample_image_data,
            extension="png"
        )
        
        updated_metrics = image_storage_metrics.get_metrics_summary()
        assert updated_metrics['storage_summary']['images_stored'] == 4
        assert updated_metrics['storage_summary']['total_storage_operations'] == 4
    
    def test_error_metrics_integration(self, temp_storage_dir, reset_all_metrics):
        """Test que los errores se registran correctamente en las métricas"""
        
        storage = ImageStorage(temp_storage_dir)
        
        # Simular error de almacenamiento (datos vacíos)
        with pytest.raises(Exception):
            storage.save_image(
                doc_id="test_doc",
                page_number=1,
                image_hash="testhash" + "0" * 56,
                image_data=b"",  # Datos vacíos que causarán error
                extension="jpg"
            )
        
        # Verificar que se registró el error
        storage_metrics = image_storage_metrics.get_metrics_summary()
        assert storage_metrics['storage_summary']['storage_errors'] > 0
        assert storage_metrics['storage_summary']['success_rate_percent'] < 100.0
    
    def test_metrics_aggregation_accuracy(self, reset_all_metrics):
        """Test que las métricas agregadas son precisas"""
        
        # Simular múltiples operaciones con valores conocidos
        test_times = [0.1, 0.2, 0.3, 0.4, 0.5]
        test_images = [1, 2, 3, 4, 5]
        
        for i, (proc_time, img_count) in enumerate(zip(test_times, test_images)):
            image_extraction_metrics.record_pdf_processing(
                images_found=img_count,
                images_saved=img_count,
                processing_time=proc_time
            )
        
        metrics = image_extraction_metrics.get_metrics_summary()
        
        # Verificar agregaciones exactas
        assert metrics['extraction_summary']['pdfs_processed'] == 5
        assert metrics['extraction_summary']['total_images_found'] == sum(test_images)  # 15
        assert metrics['performance_metrics']['total_processing_time_seconds'] == sum(test_times)  # 1.5
        assert metrics['performance_metrics']['avg_processing_time_seconds'] == sum(test_times) / len(test_times)  # 0.3
        assert metrics['performance_metrics']['max_processing_time_seconds'] == max(test_times)  # 0.5
        assert metrics['performance_metrics']['min_processing_time_seconds'] == min(test_times)  # 0.1
    
    def test_concurrent_metrics_operations(self, reset_all_metrics):
        """Test que las métricas manejan operaciones concurrentes básicas"""
        import threading
        
        def worker_extraction():
            for i in range(10):
                image_extraction_metrics.record_pdf_processing(1, 1, 0.1)
        
        def worker_storage():
            for i in range(10):
                image_storage_metrics.record_image_storage(True, 0.05, 512, False)
        
        # Ejecutar workers en paralelo
        threads = []
        threads.append(threading.Thread(target=worker_extraction))
        threads.append(threading.Thread(target=worker_storage))
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Verificar que todas las operaciones se registraron
        extraction_summary = image_extraction_metrics.get_metrics_summary()
        storage_summary = image_storage_metrics.get_metrics_summary()
        
        assert extraction_summary['extraction_summary']['pdfs_processed'] == 10
        assert storage_summary['storage_summary']['images_stored'] == 10
    
    @patch('app.check_service_health')
    def test_status_endpoint_with_service_failures(self, mock_health_check, reset_all_metrics):
        """Test que el endpoint /status maneja fallos de servicios correctamente"""
        
        # Simular fallo de Qdrant pero PyExec saludable
        def side_effect(url, endpoint, timeout=None):
            if "qdrant" in url:
                return {"status": "unhealthy", "error": "connection_error"}
            else:
                return {"status": "healthy", "latency_ms": 10.0}
        
        mock_health_check.side_effect = side_effect
        
        # Agregar métricas multimodales
        image_extraction_metrics.record_pdf_processing(2, 2, 0.5)
        
        from app import get_app_status
        status = get_app_status()
        
        # Verificar que el status general refleja el problema
        assert status['status'] == 'degraded'
        
        # Pero las métricas multimodales siguen disponibles
        assert 'multimodal' in status
        assert status['multimodal']['multimodal_enabled'] is True
        
        # Y las métricas tienen datos
        assert status['multimodal']['image_extraction']['extraction_summary']['pdfs_processed'] == 2
    
    def test_metrics_observability_end_to_end_flow(self, temp_storage_dir, sample_image_data, reset_all_metrics):
        """Test flujo completo de observabilidad desde extracción hasta reporte"""
        
        # 1. Simular extracción de imágenes con métricas
        extractor = ImageExtractor(temp_storage_dir)
        
        # Crear PDF temporal de prueba
        import fitz
        temp_pdf = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
        doc = fitz.open()  # Crear documento vacío
        page = doc.new_page()
        
        # Insertar imagen en el PDF
        img_rect = fitz.Rect(100, 100, 200, 200)
        page.insert_image(img_rect, stream=sample_image_data)
        
        doc.save(temp_pdf.name)
        doc.close()
        
        try:
            # 2. Procesar PDF (esto activará métricas de extracción)
            with patch('image_extractor.extract_images_from_pdf') as mock_extract:
                # Simular extracción exitosa
                from image_extractor import ExtractedImage
                mock_image = ExtractedImage(
                    image_data=sample_image_data,
                    format="JPEG",
                    width=100,
                    height=100,
                    page_number=1,
                    image_index=0,
                    bbox=(100, 100, 200, 200),
                    hash="testhash" + "0" * 56
                )
                mock_extract.return_value = ([mock_image], {'total_kept': 1, 'total_filtered': 0})
                
                # Procesar PDF
                result = extractor.process_pdf(temp_pdf.name, doc_id="test_doc")
                
                # Verificar resultado del procesamiento
                assert result['success'] is True
                assert result['images_found'] == 1
                assert result['processing_time'] > 0
            
            # 3. Verificar métricas de extracción
            extraction_metrics = image_extraction_metrics.get_metrics_summary()
            assert extraction_metrics['extraction_summary']['pdfs_processed'] == 1
            assert extraction_metrics['extraction_summary']['total_images_found'] == 1
            
            # 4. Simular generación de embeddings con métricas
            with patch('jina_embeddings.JinaEmbeddings') as mock_jina_emb:
                mock_emb_instance = Mock()
                mock_emb_instance.embed_images_data.return_value = [[0.1] * 1024]
                mock_jina_emb.return_value = mock_emb_instance
                
                # Generar embeddings
                factory = EmbeddingFactory()
                embeddings = factory.embed_images_flexible([sample_image_data])
                
                # Verificar métricas de embeddings
                embedding_metrics_summary = embedding_metrics.get_metrics_summary()
                assert embedding_metrics_summary['embedding_summary']['total_image_embeddings'] == 1
            
            # 5. Verificar que todas las métricas se pueden obtener desde app.py
            from app import get_app_status
            app_status = get_app_status()
            
            assert 'multimodal' in app_status
            multimodal_section = app_status['multimodal']
            
            # Verificar que todas las subsecciones están presentes
            assert 'image_extraction' in multimodal_section
            assert 'image_embeddings' in multimodal_section  
            assert 'image_storage' in multimodal_section
            assert multimodal_section['multimodal_enabled'] is True
            
            # Verificar que los datos son coherentes
            assert multimodal_section['image_extraction']['extraction_summary']['pdfs_processed'] == 1
            assert multimodal_section['image_embeddings']['embedding_summary']['total_image_embeddings'] == 1
            assert multimodal_section['image_storage']['storage_summary']['images_stored'] >= 0
            
        finally:
            # Limpiar archivo temporal
            Path(temp_pdf.name).unlink(missing_ok=True)
    
    def test_metrics_logging_integration(self, temp_storage_dir, reset_all_metrics, caplog):
        """Test que las métricas generan logs estructurados apropiados"""
        
        storage = ImageStorage(temp_storage_dir)
        
        # Ejecutar operación que debe generar logs
        with patch('image_storage.image_storage_metrics') as mock_metrics:
            mock_metrics.record_image_storage = Mock()
            mock_metrics.record_thumbnail_storage = Mock()
            
            # Simular guardado
            try:
                storage.save_image(
                    doc_id="log_test",
                    page_number=1,
                    image_hash="loghash1" + "0" * 56,
                    image_data=b"test_data",
                    extension="jpg"
                )
            except Exception:
                pass  # Esperamos error por datos inválidos
        
        # Verificar que se llamaron las funciones de métricas
        mock_metrics.record_image_storage.assert_called()
    
    def test_metrics_reset_integration(self, reset_all_metrics):
        """Test que el reset de métricas funciona correctamente en integración"""
        
        # Agregar datos a todas las métricas
        image_extraction_metrics.record_pdf_processing(5, 4, 1.0)
        embedding_metrics.record_image_embedding_batch(4, 0.8, True)
        image_storage_metrics.record_image_storage(True, 0.3, 1024, False)
        
        # Verificar que tienen datos
        assert image_extraction_metrics.get_metrics_summary()['extraction_summary']['pdfs_processed'] > 0
        assert embedding_metrics.get_metrics_summary()['embedding_summary']['total_image_embeddings'] > 0
        assert image_storage_metrics.get_metrics_summary()['storage_summary']['images_stored'] > 0
        
        # Reset todas las métricas
        image_extraction_metrics.reset()
        embedding_metrics.reset()
        image_storage_metrics.reset()
        
        # Verificar que se resetearon
        assert image_extraction_metrics.get_metrics_summary()['extraction_summary']['pdfs_processed'] == 0
        assert embedding_metrics.get_metrics_summary()['embedding_summary']['total_image_embeddings'] == 0
        assert image_storage_metrics.get_metrics_summary()['storage_summary']['images_stored'] == 0
    
    def test_observability_system_comprehensive_coverage(self, reset_all_metrics):
        """Test que el sistema de observabilidad cubre todos los aspectos críticos"""
        
        # Simular operaciones completas con diferentes resultados
        
        # 1. Extracción exitosa y con errores
        image_extraction_metrics.record_pdf_processing(10, 8, 2.0, {'total_filtered': 2, 'total_kept': 8})
        image_extraction_metrics.record_extraction_error("FileNotFoundError")
        
        # 2. Embeddings exitosos y con errores
        embedding_metrics.record_image_embedding_batch(8, 1.5, True)
        embedding_metrics.record_image_embedding_batch(3, 0.8, False)
        embedding_metrics.record_embedding_error("APIError", 3)
        
        # 3. Almacenamiento con duplicados y errores
        image_storage_metrics.record_image_storage(True, 0.3, 1024, False)  # Exitoso
        image_storage_metrics.record_image_storage(True, 0.1, 512, True)   # Duplicado
        image_storage_metrics.record_storage_error("PermissionError")
        image_storage_metrics.record_cleanup_operation(orphaned_files=2)
        
        # Obtener status completo
        from app import get_app_status
        status = get_app_status()
        
        multimodal = status['multimodal']
        
        # Verificar cobertura completa de observabilidad
        
        # Extracción
        extraction = multimodal['image_extraction']
        assert extraction['extraction_summary']['pdfs_processed'] == 1
        assert extraction['extraction_summary']['total_extraction_errors'] == 1
        assert extraction['filtering_metrics']['total_filtered'] == 2
        assert extraction['error_breakdown']['FileNotFoundError'] == 1
        
        # Embeddings  
        embeddings = multimodal['image_embeddings']
        assert embeddings['embedding_summary']['total_image_embeddings'] == 8
        assert embeddings['embedding_summary']['image_embedding_errors'] == 3
        assert embeddings['error_breakdown']['APIError'] == 3
        
        # Almacenamiento
        storage = multimodal['image_storage']
        assert storage['storage_summary']['images_stored'] == 1
        assert storage['storage_summary']['duplicate_images_avoided'] == 1
        assert storage['maintenance_metrics']['orphaned_files_cleaned'] == 2
        assert storage['error_breakdown']['PermissionError'] == 1
        
        # Verificar que todas las secciones tienen métricas de rendimiento
        for section_name, section_data in multimodal.items():
            if section_name != 'multimodal_enabled':
                assert 'performance_metrics' in section_data
                assert 'total_storage_time_seconds' in section_data['performance_metrics'] or \
                       'total_embedding_time_seconds' in section_data['performance_metrics'] or \
                       'total_processing_time_seconds' in section_data['performance_metrics']
    
    def test_status_endpoint_json_structure_validation(self, reset_all_metrics):
        """Test que la estructura JSON del endpoint /status es válida y completa"""
        
        # Agregar métricas mínimas para validar estructura
        image_extraction_metrics.record_pdf_processing(1, 1, 0.1)
        
        from app import get_app_status
        status = get_app_status()
        
        # Verificar estructura de nivel superior
        required_top_level = {'status', 'uptime_seconds', 'metrics', 'multimodal', 'reranker', 'services', 'timestamp'}
        assert set(status.keys()).issuperset(required_top_level)
        
        # Verificar estructura de métricas multimodales
        multimodal = status['multimodal']
        required_multimodal = {'image_extraction', 'image_embeddings', 'image_storage', 'multimodal_enabled'}
        assert set(multimodal.keys()).issuperset(required_multimodal)
        
        # Verificar que cada sección de métricas tiene subsecciones requeridas
        for section_name in ['image_extraction', 'image_embeddings', 'image_storage']:
            section = multimodal[section_name]
            assert 'performance_metrics' in section
            assert 'error_breakdown' in section
            
            # Verificar que las métricas de rendimiento tienen valores numéricos válidos
            perf_metrics = section['performance_metrics']
            for key, value in perf_metrics.items():
                assert isinstance(value, (int, float)), f"Metric {key} should be numeric, got {type(value)}"
                assert value >= 0, f"Metric {key} should be non-negative, got {value}"


class TestMetricsErrorHandling:
    """Tests para manejo de errores en el sistema de métricas"""
    
    def test_metrics_with_invalid_data(self):
        """Test que las métricas manejan datos inválidos gracefully"""
        metrics = ImageExtractionMetrics()
        
        # Test con valores negativos (no debería fallar)
        metrics.record_pdf_processing(-1, -1, -0.1)
        
        # Las métricas deberían registrar los valores tal como se proporcionan
        # (la validación de valores sensatos debe hacerse en el código que las llama)
        summary = metrics.get_metrics_summary()
        assert summary['extraction_summary']['pdfs_processed'] == 1
    
    def test_status_endpoint_error_resilience(self, reset_all_metrics):
        """Test que el endpoint /status es resiliente a errores en métricas individuales"""
        
        # Simular error en una de las métricas
        with patch('image_extraction_metrics.get_metrics_summary') as mock_extract:
            mock_extract.side_effect = Exception("Simulated error")
            
            from app import get_app_status
            status = get_app_status()
            
            # El endpoint debería seguir funcionando y reportar el error
            assert 'multimodal' in status
            assert 'error' in status['multimodal']
            assert status['multimodal']['multimodal_enabled'] is False
    
    def test_metrics_json_serialization(self, reset_all_metrics):
        """Test que las métricas se serializan correctamente a JSON"""
        
        # Agregar datos que podrían causar problemas de serialización
        image_extraction_metrics.record_pdf_processing(10, 8, 1.234567890123)
        embedding_metrics.record_image_embedding_batch(5, 0.123456789, True)
        
        from app import get_app_status
        status = get_app_status()
        
        # Verificar que se puede serializar a JSON sin errores
        json_str = json.dumps(status)
        
        # Verificar que se puede deserializar de vuelta
        parsed_status = json.loads(json_str)
        
        # Verificar que los datos numéricos se serializaron correctamente
        assert 'multimodal' in parsed_status
        assert isinstance(parsed_status['multimodal']['image_extraction'], dict)
        
        # Verificar que los valores flotantes se redondean apropiadamente
        perf_metrics = parsed_status['multimodal']['image_extraction']['performance_metrics']
        assert isinstance(perf_metrics['total_processing_time_seconds'], (int, float))
        # Debe estar redondeado a 2 decimales
        assert perf_metrics['total_processing_time_seconds'] == 1.23