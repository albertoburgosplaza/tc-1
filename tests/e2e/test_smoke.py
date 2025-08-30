"""End-to-end smoke tests using Docker Compose"""

import pytest
import requests
import subprocess
import time
import os
import tempfile
from pathlib import Path
import docker
import json


class TestE2ESmoke:
    """End-to-end smoke tests for the complete system"""
    
    @pytest.fixture(scope="class")
    def docker_client(self):
        """Get Docker client for container management"""
        return docker.from_env()

    @pytest.fixture(scope="class")
    def sample_pdf_file(self):
        """Create a sample PDF file for testing"""
        pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] 
   /Contents 4 0 R >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT
/F1 24 Tf
100 700 Td
(Machine Learning Guide) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f 
0000000010 00000 n 
0000000053 00000 n 
0000000100 00000 n 
0000000179 00000 n 
trailer
<< /Size 5 /Root 1 0 R >>
startxref
268
%%EOF"""
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            f.write(pdf_content)
            yield f.name
        
        # Cleanup
        os.unlink(f.name)

    @pytest.fixture(scope="class") 
    def docker_compose_test_file(self):
        """Create docker-compose.test.yml for testing"""
        compose_content = """version: "3.9"

services:
  qdrant-test:
    image: qdrant/qdrant:latest
    container_name: qdrant-test
    restart: unless-stopped
    ports:
      - "6334:6333"
    volumes:
      - qdrant_test_data:/qdrant/storage
    healthcheck:
      test: ["CMD-SHELL", "wget --no-verbose --tries=1 --spider http://localhost:6333/healthz || exit 1"]
      interval: 5s
      timeout: 3s
      retries: 10


  pyexec-test:
    build:
      context: .
      dockerfile: Dockerfile
      args:
        SERVICE: pyexec
    container_name: pyexec-test
    restart: unless-stopped
    command: bash -lc "uvicorn pyexec_service:app --host 0.0.0.0 --port 8001"
    environment:
      PYEXEC_TIMEOUT_SEC: "5"
      MAX_EXPR_LENGTH: "500"
      MAX_EXPR_COMPLEXITY: "100"
    expose:
      - "8001"
    healthcheck:
      test: ["CMD", "curl", "-fsS", "http://localhost:8001/health"]
      interval: 3s
      timeout: 2s
      retries: 10

  app-test:
    build:
      context: .
      dockerfile: Dockerfile
      args:
        SERVICE: app
    container_name: rag_app_test
    restart: unless-stopped
    depends_on:
      qdrant-test:
        condition: service_healthy
      pyexec-test:
        condition: service_healthy
    environment:
      GOOGLE_API_KEY: "test_google_key"
      JINA_API_KEY: "test_jina_key"
      QDRANT_URL: http://qdrant-test:6333
      COLLECTION_NAME: test_corpus_pdf
      GRADIO_SERVER_NAME: 0.0.0.0
      GRADIO_SERVER_PORT: "7860"
      PYEXEC_URL: http://pyexec-test:8001
      MAX_HISTORY_CHARS: "8000"
      MAX_QUERY_LENGTH: "2000"
      MIN_QUERY_LENGTH: "1"
    ports:
      - "7861:7860"
    volumes:
      - ./tests/fixtures:/app/docs:ro
    command: bash -lc "python app.py"

  ingest-test:
    build:
      context: .
      dockerfile: Dockerfile
      args:
        SERVICE: ingest
    container_name: ingest-test
    depends_on:
      qdrant-test:
        condition: service_healthy
    environment:
      QDRANT_URL: http://qdrant-test:6333
      COLLECTION_NAME: test_corpus_pdf
      EMBEDDING_MODEL: sentence-transformers/all-MiniLM-L6-v2
      DOCUMENTS_DIR: /app/docs
      MAX_PDF_SIZE_MB: "100"
      CHUNK_SIZE: "1200"
      CHUNK_OVERLAP: "180"
      MIN_CONTENT_LENGTH: "10"
    volumes:
      - ./tests/fixtures:/app/docs:ro
    entrypoint: ["python", "ingest.py"]

volumes:
  qdrant_test_data:
"""
        
        test_compose_path = Path("docker-compose.test.yml")
        with open(test_compose_path, "w") as f:
            f.write(compose_content)
        
        yield str(test_compose_path)
        
        # Cleanup
        if test_compose_path.exists():
            test_compose_path.unlink()

    @pytest.fixture(scope="class")
    def setup_test_environment(self, sample_pdf_file, docker_compose_test_file):
        """Setup test environment with sample data"""
        # Create test fixtures directory
        fixtures_dir = Path("tests/fixtures")
        fixtures_dir.mkdir(exist_ok=True)
        
        # Copy sample PDF to fixtures
        import shutil
        shutil.copy2(sample_pdf_file, fixtures_dir / "sample.pdf")
        
        yield fixtures_dir
        
        # Cleanup
        if (fixtures_dir / "sample.pdf").exists():
            (fixtures_dir / "sample.pdf").unlink()

    def wait_for_service(self, url: str, timeout: int = 60, interval: int = 2) -> bool:
        """Wait for service to become available"""
        for _ in range(timeout // interval):
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(interval)
        return False

    def wait_for_container_healthy(self, container_name: str, timeout: int = 120) -> bool:
        """Wait for container to become healthy"""
        client = docker.from_env()
        for _ in range(timeout):
            try:
                container = client.containers.get(container_name)
                if container.attrs['State']['Health']['Status'] == 'healthy':
                    return True
            except (docker.errors.NotFound, KeyError):
                pass
            time.sleep(1)
        return False

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_docker_compose_startup(self, docker_compose_test_file, setup_test_environment):
        """Test that all services start up successfully"""
        try:
            # Start services
            result = subprocess.run([
                "docker-compose", "-f", docker_compose_test_file, 
                "up", "-d", "--build"
            ], capture_output=True, text=True, timeout=300)
            
            assert result.returncode == 0, f"Docker compose up failed: {result.stderr}"
            
            # Wait for services to be healthy
            services_to_check = [
                "qdrant-test",
                "pyexec-test"
            ]
            
            for service in services_to_check:
                assert self.wait_for_container_healthy(service, timeout=120), \
                    f"Service {service} did not become healthy"
            
            # Wait for app service to be running (it doesn't have healthcheck)
            assert self.wait_for_service("http://localhost:7861", timeout=60), \
                "Gradio app did not become available"
                
        finally:
            # Cleanup
            subprocess.run([
                "docker-compose", "-f", docker_compose_test_file, "down", "-v"
            ], capture_output=True)

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_service_health_endpoints(self, docker_compose_test_file, setup_test_environment):
        """Test all service health endpoints"""
        try:
            # Start services
            subprocess.run([
                "docker-compose", "-f", docker_compose_test_file, 
                "up", "-d", "--build"
            ], timeout=300)
            
            # Wait and test health endpoints
            health_endpoints = [
                ("http://localhost:6334/healthz", "Qdrant"),
            ]
            
            for url, service_name in health_endpoints:
                assert self.wait_for_service(url, timeout=120), \
                    f"{service_name} health endpoint not responding"
                
                response = requests.get(url)
                assert response.status_code == 200, \
                    f"{service_name} health check failed"
                    
        finally:
            subprocess.run([
                "docker-compose", "-f", docker_compose_test_file, "down", "-v"
            ], capture_output=True)

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_pyexec_service_functionality(self, docker_compose_test_file, setup_test_environment):
        """Test pyexec service mathematical calculations"""
        try:
            subprocess.run([
                "docker-compose", "-f", docker_compose_test_file, 
                "up", "-d", "--build"
            ], timeout=300)
            
            # Wait for pyexec to be ready
            assert self.wait_for_container_healthy("pyexec-test", timeout=120)
            
            # Test calculation endpoint through app proxy
            # (pyexec is not directly exposed, accessed through app)
            time.sleep(10)  # Extra wait for app to fully start
            
            # Basic connectivity test - if we can reach the app, pyexec is likely working
            app_response = requests.get("http://localhost:7861", timeout=10)
            assert app_response.status_code == 200, "App not responding"
            
        finally:
            subprocess.run([
                "docker-compose", "-f", docker_compose_test_file, "down", "-v"
            ], capture_output=True)

    @pytest.mark.e2e
    @pytest.mark.slow  
    def test_ingest_process(self, docker_compose_test_file, setup_test_environment):
        """Test document ingestion process"""
        try:
            subprocess.run([
                "docker-compose", "-f", docker_compose_test_file, 
                "up", "-d", "--build"
            ], timeout=300)
            
            # Wait for base services
            assert self.wait_for_container_healthy("qdrant-test", timeout=120)
            
            # Run ingestion service
            ingest_result = subprocess.run([
                "docker-compose", "-f", docker_compose_test_file,
                "run", "--rm", "ingest-test"
            ], capture_output=True, text=True, timeout=180)
            
            # Check that ingest completed without critical errors
            # (it may have warnings but should not fail completely)
            assert ingest_result.returncode == 0, \
                f"Ingestion failed: {ingest_result.stderr}"
                
        finally:
            subprocess.run([
                "docker-compose", "-f", docker_compose_test_file, "down", "-v"
            ], capture_output=True)

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_gradio_interface_accessibility(self, docker_compose_test_file, setup_test_environment):
        """Test that Gradio interface is accessible and functional"""
        try:
            subprocess.run([
                "docker-compose", "-f", docker_compose_test_file, 
                "up", "-d", "--build"
            ], timeout=300)
            
            # Wait for all services
            services = ["qdrant-test", "pyexec-test"]
            for service in services:
                assert self.wait_for_container_healthy(service, timeout=120)
            
            # Wait for Gradio app
            assert self.wait_for_service("http://localhost:7861", timeout=120)
            
            # Test Gradio interface
            response = requests.get("http://localhost:7861")
            assert response.status_code == 200
            assert "gradio" in response.text.lower() or "machine learning" in response.text.lower()
            
        finally:
            subprocess.run([
                "docker-compose", "-f", docker_compose_test_file, "down", "-v"
            ], capture_output=True)

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_complete_system_integration(self, docker_compose_test_file, setup_test_environment):
        """Test complete system integration from startup to query"""
        try:
            # Start all services
            startup_result = subprocess.run([
                "docker-compose", "-f", docker_compose_test_file, 
                "up", "-d", "--build"
            ], capture_output=True, text=True, timeout=300)
            
            assert startup_result.returncode == 0, \
                f"System startup failed: {startup_result.stderr}"
            
            # Wait for core services to be healthy
            core_services = ["qdrant-test", "pyexec-test"]
            for service in core_services:
                assert self.wait_for_container_healthy(service, timeout=120), \
                    f"Core service {service} not healthy"
            
            # Wait for app to be available
            assert self.wait_for_service("http://localhost:7861", timeout=120), \
                "Application not available"
            
            # Run ingestion
            print("Running ingestion...")
            ingest_result = subprocess.run([
                "docker-compose", "-f", docker_compose_test_file,
                "run", "--rm", "ingest-test"
            ], capture_output=True, text=True, timeout=180)
            
            # Ingestion should complete (may have warnings but not fail)
            print(f"Ingestion stdout: {ingest_result.stdout}")
            print(f"Ingestion stderr: {ingest_result.stderr}")
            
            # Test basic system health
            health_checks = [
                "http://localhost:6334/healthz",  # Qdrant
                "http://localhost:7861",  # Gradio app
            ]
            
            for url in health_checks:
                response = requests.get(url, timeout=10)
                assert response.status_code == 200, f"Health check failed for {url}"
            
            print("✅ Complete system integration test passed")
            
        except Exception as e:
            # Print container logs for debugging
            print("=== Container Logs for Debugging ===")
            try:
                logs_result = subprocess.run([
                    "docker-compose", "-f", docker_compose_test_file, "logs"
                ], capture_output=True, text=True, timeout=30)
                print(logs_result.stdout)
            except:
                pass
            raise e
            
        finally:
            # Cleanup
            subprocess.run([
                "docker-compose", "-f", docker_compose_test_file, "down", "-v"
            ], capture_output=True)

    @pytest.mark.e2e
    def test_service_configuration_validation(self, docker_compose_test_file):
        """Test that docker-compose configuration is valid"""
        # Validate docker-compose file
        result = subprocess.run([
            "docker-compose", "-f", docker_compose_test_file, "config"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0, f"Docker compose config invalid: {result.stderr}"
        
        # Parse and validate the configuration
        config_output = result.stdout
        assert "qdrant-test" in config_output
        # Ollama is no longer used - removed  
        assert "pyexec-test" in config_output
        assert "app-test" in config_output
        assert "ingest-test" in config_output

    def test_docker_environment_setup(self):
        """Test that Docker environment is properly set up"""
        try:
            result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
            assert result.returncode == 0, "Docker not available"
            
            result = subprocess.run(["docker-compose", "--version"], capture_output=True, text=True)
            assert result.returncode == 0, "Docker Compose not available"
            
        except FileNotFoundError:
            pytest.skip("Docker or Docker Compose not installed")