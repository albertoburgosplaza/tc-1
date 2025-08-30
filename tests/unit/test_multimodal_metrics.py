"""
Tests unitarios para las métricas de funcionalidad multimodal
"""

import pytest
import time
from unittest.mock import Mock, patch

from image_extractor import ImageExtractionMetrics, image_extraction_metrics
from embedding_factory import EmbeddingMetrics, embedding_metrics  
from image_storage import ImageStorageMetrics, image_storage_metrics


class TestImageExtractionMetrics:
    """Tests para métricas de extracción de imágenes"""
    
    def test_metrics_initialization(self):
        """Test que las métricas se inicializan correctamente"""
        metrics = ImageExtractionMetrics()
        
        assert metrics.pdfs_processed == 0
        assert metrics.total_images_found == 0
        assert metrics.total_images_saved == 0
        assert metrics.total_extraction_errors == 0
        assert metrics.total_processing_time == 0.0
        assert len(metrics.extraction_times_by_pdf) == 0
        assert len(metrics.images_per_pdf) == 0
        assert len(metrics.errors_by_type) == 0
        
        # Verificar que filter_stats tiene todas las claves esperadas
        expected_filter_keys = {
            'too_small', 'extreme_aspect_ratio', 'too_small_file', 
            'header_footer', 'low_complexity', 'duplicate_hash',
            'total_filtered', 'total_kept'
        }
        assert set(metrics.filter_stats.keys()) == expected_filter_keys
        assert all(metrics.filter_stats[key] == 0 for key in expected_filter_keys)
    
    def test_record_pdf_processing(self):
        """Test que se registran correctamente las métricas de procesamiento de PDF"""
        metrics = ImageExtractionMetrics()
        
        # Simular filter stats
        filter_stats = {
            'too_small': 2,
            'header_footer': 1,
            'total_filtered': 3,
            'total_kept': 5
        }
        
        # Registrar procesamiento
        metrics.record_pdf_processing(
            images_found=8,
            images_saved=5,
            processing_time=2.5,
            filter_stats=filter_stats
        )
        
        # Verificar métricas básicas
        assert metrics.pdfs_processed == 1
        assert metrics.total_images_found == 8
        assert metrics.total_images_saved == 5
        assert metrics.total_processing_time == 2.5
        assert len(metrics.extraction_times_by_pdf) == 1
        assert metrics.extraction_times_by_pdf[0] == 2.5
        assert len(metrics.images_per_pdf) == 1
        assert metrics.images_per_pdf[0] == 8
        
        # Verificar que filter stats se agregaron
        assert metrics.filter_stats['too_small'] == 2
        assert metrics.filter_stats['header_footer'] == 1
        assert metrics.filter_stats['total_filtered'] == 3
        assert metrics.filter_stats['total_kept'] == 5
    
    def test_record_extraction_error(self):
        """Test que se registran correctamente los errores de extracción"""
        metrics = ImageExtractionMetrics()
        
        metrics.record_extraction_error("FileNotFoundError")
        metrics.record_extraction_error("PermissionError")
        metrics.record_extraction_error("FileNotFoundError")  # Segundo error del mismo tipo
        
        assert metrics.total_extraction_errors == 3
        assert metrics.errors_by_type["FileNotFoundError"] == 2
        assert metrics.errors_by_type["PermissionError"] == 1
    
    def test_get_metrics_summary(self):
        """Test que el resumen de métricas se genera correctamente"""
        metrics = ImageExtractionMetrics()
        
        # Agregar datos de prueba
        metrics.record_pdf_processing(10, 8, 1.5, {'total_filtered': 2, 'total_kept': 8})
        metrics.record_pdf_processing(5, 5, 1.0, {'total_filtered': 0, 'total_kept': 5})
        metrics.record_extraction_error("TestError")
        
        summary = metrics.get_metrics_summary()
        
        # Verificar estructura del resumen
        assert 'extraction_summary' in summary
        assert 'performance_metrics' in summary
        assert 'filtering_metrics' in summary
        assert 'error_breakdown' in summary
        
        # Verificar cálculos
        assert summary['extraction_summary']['pdfs_processed'] == 2
        assert summary['extraction_summary']['total_images_found'] == 15
        assert summary['extraction_summary']['total_images_saved'] == 13
        assert summary['extraction_summary']['total_extraction_errors'] == 1
        assert summary['extraction_summary']['success_rate_percent'] == 50.0  # 1 success out of 2 attempts
        
        # Verificar métricas de rendimiento
        assert summary['performance_metrics']['total_processing_time_seconds'] == 2.5
        assert summary['performance_metrics']['avg_processing_time_seconds'] == 1.25
        assert summary['performance_metrics']['max_processing_time_seconds'] == 1.5
        assert summary['performance_metrics']['min_processing_time_seconds'] == 1.0


class TestEmbeddingMetrics:
    """Tests para métricas de embeddings"""
    
    def test_metrics_initialization(self):
        """Test que las métricas de embeddings se inicializan correctamente"""
        metrics = EmbeddingMetrics()
        
        assert metrics.image_embeddings_generated == 0
        assert metrics.text_embeddings_generated == 0
        assert metrics.total_embedding_time == 0.0
        assert metrics.image_embedding_errors == 0
        assert metrics.text_embedding_errors == 0
        assert len(metrics.embedding_times) == 0
        assert len(metrics.errors_by_type) == 0
        assert len(metrics.images_processed_by_batch) == 0
        assert len(metrics.avg_latency_per_image) == 0
    
    def test_record_image_embedding_batch_success(self):
        """Test que se registran correctamente los embeddings de imagen exitosos"""
        metrics = EmbeddingMetrics()
        
        metrics.record_image_embedding_batch(batch_size=5, processing_time=2.0, success=True)
        
        assert metrics.image_embeddings_generated == 5
        assert metrics.total_embedding_time == 2.0
        assert len(metrics.embedding_times) == 1
        assert metrics.embedding_times[0] == 2.0
        assert len(metrics.images_processed_by_batch) == 1
        assert metrics.images_processed_by_batch[0] == 5
        assert len(metrics.avg_latency_per_image) == 1
        assert metrics.avg_latency_per_image[0] == 0.4  # 2.0 / 5
        assert metrics.image_embedding_errors == 0
    
    def test_record_image_embedding_batch_error(self):
        """Test que se registran correctamente los errores de embeddings de imagen"""
        metrics = EmbeddingMetrics()
        
        metrics.record_image_embedding_batch(batch_size=3, processing_time=1.0, success=False)
        
        assert metrics.image_embeddings_generated == 0
        assert metrics.image_embedding_errors == 3
        assert metrics.total_embedding_time == 1.0  # Tiempo se registra incluso en errores
        assert len(metrics.embedding_times) == 0  # No se agregan tiempos en errores
    
    def test_record_text_embedding_batch(self):
        """Test que se registran correctamente los embeddings de texto"""
        metrics = EmbeddingMetrics()
        
        metrics.record_text_embedding_batch(batch_size=10, processing_time=1.5, success=True)
        
        assert metrics.text_embeddings_generated == 10
        assert metrics.total_embedding_time == 1.5
        assert metrics.text_embedding_errors == 0
        
        # Text embeddings no tienen latencia por imagen
        assert len(metrics.avg_latency_per_image) == 0
    
    def test_record_embedding_error(self):
        """Test que se registran correctamente los errores de embedding"""
        metrics = EmbeddingMetrics()
        
        metrics.record_embedding_error("APIError", batch_size=2)
        metrics.record_embedding_error("TimeoutError", batch_size=1)
        
        assert metrics.errors_by_type["APIError"] == 2
        assert metrics.errors_by_type["TimeoutError"] == 1
    
    def test_get_metrics_summary(self):
        """Test que el resumen de métricas de embeddings se genera correctamente"""
        metrics = EmbeddingMetrics()
        
        # Agregar datos de prueba
        metrics.record_image_embedding_batch(5, 2.0, True)
        metrics.record_text_embedding_batch(10, 1.0, True)
        metrics.record_image_embedding_batch(2, 0.5, False)
        metrics.record_embedding_error("TestError", 2)
        
        summary = metrics.get_metrics_summary()
        
        # Verificar estructura
        assert 'embedding_summary' in summary
        assert 'performance_metrics' in summary
        assert 'error_breakdown' in summary
        
        # Verificar cálculos
        assert summary['embedding_summary']['total_image_embeddings'] == 5
        assert summary['embedding_summary']['total_text_embeddings'] == 10
        assert summary['embedding_summary']['image_embedding_errors'] == 2
        assert summary['embedding_summary']['success_rate_percent'] == 88.24  # 15/(15+2)*100


class TestImageStorageMetrics:
    """Tests para métricas de almacenamiento de imágenes"""
    
    def test_metrics_initialization(self):
        """Test que las métricas de almacenamiento se inicializan correctamente"""
        metrics = ImageStorageMetrics()
        
        assert metrics.images_stored == 0
        assert metrics.thumbnails_stored == 0
        assert metrics.storage_operations == 0
        assert metrics.storage_errors == 0
        assert metrics.total_storage_time == 0.0
        assert metrics.bytes_stored == 0
        assert metrics.duplicate_images_avoided == 0
        assert metrics.documents_processed == 0
        assert metrics.cleanup_operations == 0
        assert metrics.orphaned_files_cleaned == 0
        assert len(metrics.storage_times) == 0
        assert len(metrics.errors_by_type) == 0
    
    def test_record_image_storage_success(self):
        """Test que se registran correctamente los almacenamientos exitosos"""
        metrics = ImageStorageMetrics()
        
        metrics.record_image_storage(
            success=True,
            processing_time=0.5,
            bytes_stored=1024,
            was_duplicate=False
        )
        
        assert metrics.storage_operations == 1
        assert metrics.images_stored == 1
        assert metrics.total_storage_time == 0.5
        assert metrics.bytes_stored == 1024
        assert metrics.duplicate_images_avoided == 0
        assert metrics.storage_errors == 0
        assert len(metrics.storage_times) == 1
        assert metrics.storage_times[0] == 0.5
    
    def test_record_image_storage_duplicate(self):
        """Test que se registran correctamente los duplicados evitados"""
        metrics = ImageStorageMetrics()
        
        metrics.record_image_storage(
            success=True,
            processing_time=0.1,
            bytes_stored=1024,
            was_duplicate=True
        )
        
        assert metrics.storage_operations == 1
        assert metrics.images_stored == 0  # No se cuenta como nueva imagen
        assert metrics.duplicate_images_avoided == 1
        assert metrics.bytes_stored == 0  # No se cuenta en bytes totales
        assert metrics.total_storage_time == 0.1
        assert len(metrics.storage_times) == 1
    
    def test_record_image_storage_error(self):
        """Test que se registran correctamente los errores de almacenamiento"""
        metrics = ImageStorageMetrics()
        
        metrics.record_image_storage(
            success=False,
            processing_time=0.2,
            bytes_stored=0
        )
        
        assert metrics.storage_operations == 1
        assert metrics.images_stored == 0
        assert metrics.storage_errors == 1
        assert metrics.total_storage_time == 0.0  # No se cuenta tiempo en errores
        assert len(metrics.storage_times) == 0
    
    def test_record_thumbnail_storage(self):
        """Test que se registran correctamente los thumbnails almacenados"""
        metrics = ImageStorageMetrics()
        
        metrics.record_thumbnail_storage(success=True, bytes_stored=256)
        
        assert metrics.thumbnails_stored == 1
        assert metrics.bytes_stored == 256
    
    def test_record_document_processed(self):
        """Test que se registra correctamente el procesamiento de documentos"""
        metrics = ImageStorageMetrics()
        
        metrics.record_document_processed()
        metrics.record_document_processed()
        
        assert metrics.documents_processed == 2
    
    def test_record_cleanup_operation(self):
        """Test que se registran correctamente las operaciones de limpieza"""
        metrics = ImageStorageMetrics()
        
        metrics.record_cleanup_operation(orphaned_files=5)
        
        assert metrics.cleanup_operations == 1
        assert metrics.orphaned_files_cleaned == 5
    
    def test_record_storage_error(self):
        """Test que se registran correctamente los errores por tipo"""
        metrics = ImageStorageMetrics()
        
        metrics.record_storage_error("PermissionError")
        metrics.record_storage_error("OSError")
        metrics.record_storage_error("PermissionError")
        
        assert metrics.errors_by_type["PermissionError"] == 2
        assert metrics.errors_by_type["OSError"] == 1
    
    def test_get_metrics_summary(self):
        """Test que el resumen de métricas de almacenamiento se genera correctamente"""
        metrics = ImageStorageMetrics()
        
        # Agregar datos de prueba
        metrics.record_image_storage(True, 0.5, 1024, False)
        metrics.record_image_storage(True, 0.3, 512, False)
        metrics.record_image_storage(True, 0.1, 256, True)  # Duplicado
        metrics.record_thumbnail_storage(True, 128)
        metrics.record_document_processed()
        metrics.record_cleanup_operation(3)
        metrics.record_storage_error("TestError")
        
        summary = metrics.get_metrics_summary()
        
        # Verificar estructura
        assert 'storage_summary' in summary
        assert 'performance_metrics' in summary
        assert 'maintenance_metrics' in summary
        assert 'error_breakdown' in summary
        
        # Verificar cálculos
        assert summary['storage_summary']['images_stored'] == 2  # No cuenta duplicados
        assert summary['storage_summary']['thumbnails_stored'] == 1
        assert summary['storage_summary']['total_storage_operations'] == 3
        assert summary['storage_summary']['duplicate_images_avoided'] == 1
        assert summary['storage_summary']['documents_processed'] == 1
        
        # Verificar métricas de rendimiento
        assert summary['performance_metrics']['total_storage_time_seconds'] == 0.9  # 0.5 + 0.3 + 0.1
        assert summary['performance_metrics']['total_bytes_stored'] == 1664  # 1024 + 512 + 128
        assert summary['maintenance_metrics']['cleanup_operations'] == 1
        assert summary['maintenance_metrics']['orphaned_files_cleaned'] == 3


class TestGlobalMetricsInstances:
    """Tests para las instancias globales de métricas"""
    
    def test_global_image_extraction_metrics_exists(self):
        """Test que la instancia global de métricas de extracción existe"""
        assert image_extraction_metrics is not None
        assert isinstance(image_extraction_metrics, ImageExtractionMetrics)
    
    def test_global_embedding_metrics_exists(self):
        """Test que la instancia global de métricas de embeddings existe"""
        assert embedding_metrics is not None
        assert isinstance(embedding_metrics, EmbeddingMetrics)
    
    def test_global_storage_metrics_exists(self):
        """Test que la instancia global de métricas de almacenamiento existe"""
        assert image_storage_metrics is not None
        assert isinstance(image_storage_metrics, ImageStorageMetrics)
    
    def test_metrics_reset_functionality(self):
        """Test que las métricas se pueden resetear correctamente"""
        # Test con extracción
        image_extraction_metrics.record_pdf_processing(5, 3, 1.0)
        assert image_extraction_metrics.pdfs_processed > 0
        
        image_extraction_metrics.reset()
        assert image_extraction_metrics.pdfs_processed == 0
        assert image_extraction_metrics.total_images_found == 0
        
        # Test con embeddings
        embedding_metrics.record_image_embedding_batch(3, 1.0, True)
        assert embedding_metrics.image_embeddings_generated > 0
        
        embedding_metrics.reset()
        assert embedding_metrics.image_embeddings_generated == 0
        
        # Test con almacenamiento
        image_storage_metrics.record_image_storage(True, 0.5, 1024, False)
        assert image_storage_metrics.images_stored > 0
        
        image_storage_metrics.reset()
        assert image_storage_metrics.images_stored == 0


class TestMetricsIntegration:
    """Tests de integración entre diferentes métricas"""
    
    def test_metrics_work_independently(self):
        """Test que las diferentes métricas trabajan independientemente"""
        # Resetear todas las métricas
        image_extraction_metrics.reset()
        embedding_metrics.reset()
        image_storage_metrics.reset()
        
        # Agregar datos a cada métrica
        image_extraction_metrics.record_pdf_processing(5, 4, 1.0)
        embedding_metrics.record_image_embedding_batch(4, 0.8, True)
        image_storage_metrics.record_image_storage(True, 0.3, 1024, False)
        
        # Verificar que cada una mantiene sus propios datos
        extraction_summary = image_extraction_metrics.get_metrics_summary()
        embedding_summary = embedding_metrics.get_metrics_summary()
        storage_summary = image_storage_metrics.get_metrics_summary()
        
        assert extraction_summary['extraction_summary']['pdfs_processed'] == 1
        assert embedding_summary['embedding_summary']['total_image_embeddings'] == 4
        assert storage_summary['storage_summary']['images_stored'] == 1
        
        # Verificar que no se interfieren entre sí
        assert extraction_summary['extraction_summary']['total_images_found'] == 5
        assert embedding_summary['embedding_summary']['total_text_embeddings'] == 0
        assert storage_summary['storage_summary']['thumbnails_stored'] == 0
    
    def test_metrics_edge_cases(self):
        """Test casos edge en las métricas"""
        metrics = ImageExtractionMetrics()
        
        # Test con division por cero
        summary = metrics.get_metrics_summary()
        assert summary['extraction_summary']['success_rate_percent'] == 100.0
        assert summary['performance_metrics']['avg_processing_time_seconds'] == 0.0
        
        # Test con datos mínimos
        metrics.record_pdf_processing(0, 0, 0.001)  # PDF sin imágenes
        summary = metrics.get_metrics_summary()
        
        assert summary['extraction_summary']['pdfs_processed'] == 1
        assert summary['extraction_summary']['total_images_found'] == 0
        assert summary['performance_metrics']['min_processing_time_seconds'] == 0.0


class TestMetricsPerformance:
    """Tests de rendimiento para las métricas"""
    
    def test_metrics_performance_with_large_datasets(self):
        """Test que las métricas manejan eficientemente grandes volúmenes de datos"""
        metrics = ImageExtractionMetrics()
        
        # Simular procesamiento de muchos PDFs
        start_time = time.time()
        for i in range(100):
            metrics.record_pdf_processing(
                images_found=i % 10,
                images_saved=(i % 10) - (i % 3), 
                processing_time=0.1 + (i % 5) * 0.01
            )
        
        processing_time = time.time() - start_time
        
        # Verificar que el procesamiento fue eficiente (< 0.1 segundos)
        assert processing_time < 0.1
        
        # Verificar que los datos se registraron correctamente
        summary = metrics.get_metrics_summary()
        assert summary['extraction_summary']['pdfs_processed'] == 100
        assert len(metrics.extraction_times_by_pdf) == 100
        assert len(metrics.images_per_pdf) == 100
    
    def test_metrics_memory_efficiency(self):
        """Test que las métricas no consumen memoria excesiva"""
        metrics = ImageStorageMetrics()
        
        # Agregar muchas operaciones
        for i in range(1000):
            metrics.record_image_storage(
                success=True,
                processing_time=0.001,
                bytes_stored=1024,
                was_duplicate=i % 10 == 0  # 10% duplicados
            )
        
        # Verificar que las listas no crecen descontroladamente
        assert len(metrics.storage_times) == 1000
        
        # Las métricas deberían ser agregados simples, no listas gigantes
        summary = metrics.get_metrics_summary()
        assert isinstance(summary['storage_summary']['images_stored'], int)
        assert isinstance(summary['performance_metrics']['avg_storage_time_ms'], float)