# Contributing to RAG Chatbot

Welcome to the RAG Chatbot project! This guide will help you understand the project architecture, set up your development environment, and contribute effectively.

## Table of Contents
- [Architecture Overview](#architecture-overview)
- [Development Setup](#development-setup)
- [Code Structure](#code-structure)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Code Standards](#code-standards)
- [Adding New Features](#adding-new-features)
- [Security Considerations](#security-considerations)
- [Troubleshooting](#troubleshooting)

## Architecture Overview

### System Components

The RAG Chatbot uses a microservices architecture with the following components:

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG CHATBOT ARCHITECTURE                 │
├─────────────────────────────────────────────────────────────┤
│  [User] → [Gradio UI] → [App Service] → [LLM/RAG/PyExec]    │
│             │              │                               │
│             │              ├─ [Ollama LLM]                 │
│             │              ├─ [Qdrant Vector DB]           │
│             │              └─ [PyExec Service]             │
│             │                                              │
│             └─ [Ingest Job] → [PDF Processing]             │
└─────────────────────────────────────────────────────────────┘
```

#### Core Services

1. **App Service** (`app.py`)
   - **Role**: Main orchestrator and user interface
   - **Technology**: Python + Gradio + LangChain
   - **Port**: 7860 (UI), 8080 (health check)
   - **Responsibilities**:
     - Serve web interface
     - Handle user queries
     - Orchestrate RAG workflow
     - Manage conversation memory
     - Route Python execution requests
     - Input validation and sanitization

2. **Qdrant Vector Database**
   - **Role**: Vector storage and semantic search
   - **Technology**: Qdrant (Rust-based vector DB)
   - **Port**: 6333
   - **Responsibilities**:
     - Store document embeddings
     - Perform similarity search
     - Provide web dashboard

3. **Ollama LLM Service**
   - **Role**: Large Language Model inference
   - **Technology**: Ollama (local LLM runner)
   - **Port**: 11434
   - **Responsibilities**:
     - Run local LLM models
     - Generate text responses
     - Support multiple model formats

4. **PyExec Service** (`pyexec_service.py`)
   - **Role**: Secure Python code execution
   - **Technology**: FastAPI + AST validation
   - **Port**: 8001
   - **Responsibilities**:
     - Execute Python expressions safely
     - Validate code complexity
     - Enforce execution timeouts
     - Block dangerous operations

5. **Ingest Job** (`ingest.py`)
   - **Role**: Document processing pipeline
   - **Technology**: LangChain + PyPDF
   - **Run mode**: One-time job
   - **Responsibilities**:
     - Process PDF documents
     - Create text chunks
     - Generate embeddings
     - Store vectors in Qdrant

### Data Flow

#### RAG Query Flow
1. User submits query via Gradio interface
2. App validates input and checks conversation history
3. Query is embedded using HuggingFace transformer
4. Vector search retrieves relevant document chunks from Qdrant
5. Context + query sent to Ollama LLM
6. Response generated and returned to user

#### Python Execution Flow
1. App detects Python expression (starts with `python:` or natural language)
2. Expression sent to PyExec service via HTTP
3. PyExec validates AST complexity and safety
4. Expression executed in sandboxed environment
5. Result returned to App and included in response

### Communication Patterns

- **Synchronous HTTP**: App ↔ PyExec, App ↔ Qdrant, App ↔ Ollama
- **Database Access**: App → Qdrant (read), Ingest → Qdrant (write)
- **Model Access**: App → Ollama (inference)
- **Health Checks**: All services expose health endpoints

## Development Setup

### Prerequisites

- Docker 20.10+ and Docker Compose 2.0+
- Python 3.9+ (for local development)
- Git
- 8GB RAM recommended
- 15GB free disk space

### Quick Start

1. **Clone repository:**
   ```bash
   git clone <repository-url>
   cd turingchallenge-reto-1
   ```

2. **Run automated setup:**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. **For development with code changes:**
   ```bash
   docker compose down
   docker compose up --build -d
   ```

### Development Environment

#### Local Python Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.app.txt
pip install -r requirements.ingest.txt
pip install -r requirements.pyexec.txt

# Install development dependencies
pip install pytest black isort flake8 mypy
```

#### Environment Variables

Create a `.env.local` file for development:

```bash
# LLM Configuration
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=mistral:7b-instruct

# Vector Database
QDRANT_URL=http://localhost:6333
COLLECTION_NAME=corpus_pdf_dev

# Services
PYEXEC_URL=http://localhost:8001
GRADIO_SERVER_PORT=7860

# Development specific
LOG_LEVEL=DEBUG
MAX_QUERY_LENGTH=5000
PYEXEC_TIMEOUT_SEC=10
```

## Code Structure

### Project Layout

```
├── app.py                    # Main Gradio application
├── pyexec_service.py         # Python execution microservice
├── ingest.py                 # Document processing pipeline
├── docker-compose.yml        # Service orchestration
├── Dockerfile                # Multi-stage container build
├── setup.sh                  # Automated setup script
│
├── requirements.*.txt        # Python dependencies by service
├── pytest.ini              # Test configuration
├── .dockerignore            # Docker build context exclusions
│
├── tests/                   # Test suite
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   ├── e2e/               # End-to-end tests
│   └── fixtures/          # Test data
│
├── docs/                   # Document storage for ingestion
└── README.md              # User documentation
```

### Key Modules

#### app.py - Main Application
```python
# Core components
- RAG orchestration (retrieval + generation)
- Conversation memory management
- Python expression detection and routing
- Input validation and sanitization
- Gradio interface setup
- Health check endpoint
```

#### pyexec_service.py - Secure Execution
```python
# Security features
- AST-based code validation
- Execution timeout enforcement
- Complexity analysis
- Whitelist-based function filtering
- Error categorization and logging
```

#### ingest.py - Document Processing
```python
# Processing pipeline
- PDF parsing and text extraction
- Text chunking with overlap
- Embedding generation
- Vector database storage
- Error handling and recovery
```

### Configuration Management

All services use environment variables for configuration:

- **Centralized**: docker-compose.yml defines all variables
- **Overridable**: Use `.env` file for local overrides
- **Validated**: Each service validates required variables on startup
- **Typed**: All environment variables are cast to appropriate types

## Development Workflow

### Branch Strategy

- `main`: Production-ready code
- `develop`: Integration branch for features
- `feature/feature-name`: Individual feature development
- `bugfix/issue-description`: Bug fixes
- `hotfix/critical-fix`: Emergency production fixes

### Standard Workflow

1. **Create feature branch:**
   ```bash
   git checkout -b feature/new-awesome-feature
   ```

2. **Make changes with tests:**
   - Write failing tests first (TDD approach)
   - Implement feature
   - Ensure all tests pass
   - Update documentation

3. **Code quality checks:**
   ```bash
   # Format code
   black .
   isort .
   
   # Lint code
   flake8 .
   mypy .
   
   # Run tests
   python run_tests.py
   ```

4. **Commit with conventional format:**
   ```bash
   git commit -m "feat: add vector search optimization"
   git commit -m "fix: resolve memory leak in conversation history"
   git commit -m "docs: update API documentation"
   ```

5. **Create Pull Request:**
   - Use PR template
   - Include tests and documentation
   - Request review from maintainers

### Docker Development

For services that need rebuilding during development:

```bash
# Rebuild specific service
docker compose up --build app

# Rebuild all custom services
docker compose up --build pyexec app ingest

# Development with live reload
docker compose -f docker-compose.dev.yml up
```

### Hot Reload Development

1. **Run external services:**
   ```bash
   docker compose up -d qdrant ollama
   ```

2. **Run app locally:**
   ```bash
   export QDRANT_URL=http://localhost:6333
   export OLLAMA_BASE_URL=http://localhost:11434
   python app.py
   ```

## Testing

### Test Structure

- **Unit Tests**: Individual function/class testing
- **Integration Tests**: Service interaction testing
- **E2E Tests**: Complete workflow testing
- **Performance Tests**: Load and response time testing

### Running Tests

```bash
# All tests
python run_tests.py

# Specific test categories
pytest tests/unit/                    # Fast unit tests
pytest tests/integration/             # Service integration
pytest tests/e2e/                     # End-to-end workflows

# With coverage
pytest --cov=. --cov-report=html

# Specific test file
pytest tests/unit/test_embeddings.py -v
```

### Test Fixtures

Located in `tests/fixtures/`:
- `qa_dataset.json`: Question-answer pairs for RAG testing
- Sample PDFs for ingestion testing
- Mock service responses

### Writing Tests

#### Unit Test Example
```python
import pytest
from app import validate_query

def test_validate_query_length():
    # Test normal query
    result = validate_query("What is AI?")
    assert result.is_valid == True
    
    # Test empty query
    result = validate_query("")
    assert result.is_valid == False
    assert "too short" in result.error_message.lower()
```

#### Integration Test Example
```python
import requests
import pytest

def test_pyexec_service_integration():
    response = requests.post(
        "http://localhost:8001/eval",
        json={"expression": "2 + 2"}
    )
    assert response.status_code == 200
    assert response.json()["result"] == 4
```

## Code Standards

### Python Style

- **Formatter**: Black (line length: 88)
- **Import sorting**: isort
- **Linting**: flake8
- **Type checking**: mypy (when possible)

### Code Organization

- **Functions**: Max 50 lines, single responsibility
- **Classes**: Cohesive, well-documented
- **Modules**: Logical grouping, clear interfaces
- **Error handling**: Explicit, with logging

### Documentation

- **Docstrings**: Google style for all public functions
- **Comments**: Explain complex logic and business decisions
- **Type hints**: Use where helpful for clarity
- **README updates**: Keep user documentation current

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): description

feat: add new RAG optimization
fix: resolve embedding dimension mismatch
docs: update architecture diagrams
test: add integration tests for pyexec
refactor: extract conversation memory class
```

## Adding New Features

### New Service Integration

1. **Add service to docker-compose.yml:**
   ```yaml
   new-service:
     build:
       context: .
       dockerfile: Dockerfile.newservice
     environment:
       - CONFIG_VAR=value
     healthcheck:
       test: ["CMD", "curl", "-f", "http://localhost:PORT/health"]
   ```

2. **Implement health check endpoint**
3. **Add service dependencies in app service**
4. **Update documentation and setup script**

### New LLM Model Support

1. **Add model to Ollama supported models**
2. **Update docker-compose.yml environment variables**
3. **Test model compatibility with existing prompts**
4. **Update setup.sh model options**

### New Document Format Support

1. **Add loader to ingest.py:**
   ```python
   from langchain_community.document_loaders import NewFormatLoader
   
   def process_new_format(file_path):
       loader = NewFormatLoader(file_path)
       return loader.load()
   ```

2. **Update file validation in ingest.py**
3. **Add tests for new format**
4. **Update documentation**

### RAG Enhancement

1. **Retrieval improvements**: Modify search parameters, add re-ranking
2. **Generation improvements**: Adjust prompts, add post-processing
3. **Memory enhancements**: Extend conversation tracking
4. **Evaluation**: Add metrics and benchmarks

## Security Considerations

### Input Validation

All user input is validated at multiple layers:

1. **Gradio interface**: Length limits, basic sanitization
2. **App service**: XSS prevention, content filtering
3. **PyExec service**: AST validation, execution sandboxing

### Code Execution Security

The PyExec service implements multiple security measures:

- **AST Analysis**: Block dangerous node types (imports, file I/O)
- **Timeout Enforcement**: Prevent infinite loops
- **Complexity Limits**: Prevent resource exhaustion
- **Function Whitelist**: Only allow safe built-in functions

### Container Security

- **Non-root users**: All services run as unprivileged users
- **Resource limits**: Memory and CPU constraints
- **Network isolation**: Services communicate only as needed
- **Health checks**: Early detection of service issues

### Security Guidelines for Contributors

1. **Never** add file I/O to PyExec whitelist
2. **Always** validate user input at service boundaries
3. **Use** parameterized queries for any database operations
4. **Test** security measures with adversarial inputs
5. **Log** security events for monitoring

## Troubleshooting

### Common Development Issues

#### Services Not Starting
```bash
# Check service logs
docker compose logs [service-name]

# Verify port availability
netstat -tuln | grep [port]

# Reset environment
docker compose down -v
docker compose up --build
```

#### Model Download Failures
```bash
# Check Ollama connectivity
docker exec ollama curl -I https://ollama.ai

# Try alternative model
docker exec ollama ollama pull mistral:7b-instruct-q4_0

# Check disk space
docker exec ollama df -h
```

#### Vector Database Issues
```bash
# Check Qdrant health
curl http://localhost:6333/healthz

# Reset collections
docker exec qdrant rm -rf /qdrant/storage/*
docker compose restart qdrant
docker compose run --rm ingest
```

#### Python Execution Problems
```bash
# Test PyExec directly
curl -X POST http://localhost:8001/eval \
  -H "Content-Type: application/json" \
  -d '{"expression": "2+2"}'

# Check security restrictions
docker compose logs pyexec
```

### Performance Optimization

#### Memory Usage
- Monitor with `docker stats`
- Adjust model quantization
- Optimize chunk sizes for embeddings

#### Response Time
- Profile with `cProfile` for Python services
- Monitor database query performance
- Optimize vector search parameters

#### Scaling
- Use Docker Compose replicas for stateless services
- Consider Redis for conversation memory persistence
- Implement load balancing for high traffic

### Debugging Tips

1. **Structured Logging**: All services use consistent log formats
2. **Health Endpoints**: Check service status before debugging
3. **Docker Exec**: Access service containers for direct debugging
4. **Environment Variables**: Verify configuration with `env` command
5. **Network Issues**: Use `docker compose logs` and `docker inspect`

### Getting Help

- **Issues**: Search existing GitHub issues first
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check README.md for user-facing info
- **Code Review**: Tag maintainers in PRs for feedback

---

## Quick Reference

### Essential Commands
```bash
# Development
./setup.sh                           # Full automated setup
docker compose up --build            # Rebuild and start
python run_tests.py                  # Run full test suite

# Maintenance  
docker compose down -v               # Stop and remove volumes
docker system prune -f               # Clean Docker resources
git pull && ./setup.sh -f            # Update and reinstall

# Monitoring
docker compose logs -f [service]     # Follow service logs
docker stats                         # Resource monitoring
curl http://localhost:8080/health    # Health check
```

### File Locations
- Configuration: `docker-compose.yml`
- Environment: `.env` (local overrides)
- Tests: `tests/` directory
- Documents: `docs/` directory
- Logs: Container stdout (use `docker compose logs`)

Thank you for contributing to the RAG Chatbot project! 🚀