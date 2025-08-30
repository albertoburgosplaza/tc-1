# RAG Chatbot Testing Suite

Comprehensive testing suite for the RAG (Retrieval-Augmented Generation) Chatbot system.

## Test Structure

```
tests/
├── unit/                   # Unit tests for individual components
│   ├── test_chunking.py           # Text chunking functionality
│   ├── test_embeddings.py         # Embedding generation and handling
│   ├── test_input_validation.py   # Input validation functions
│   └── test_pyexec_validation.py  # Mathematical expression validation
├── integration/            # Integration tests for service interactions
│   ├── test_rag_flow.py           # Complete RAG pipeline
│   └── test_services_integration.py # Service-to-service communication
├── e2e/                   # End-to-end tests
│   └── test_smoke.py              # Docker Compose smoke tests
├── evaluation/            # RAG quality evaluation
│   └── test_rag_quality.py       # Quality metrics and dataset evaluation
├── fixtures/              # Test data and utilities
│   └── qa_dataset.json           # Synthetic Q&A evaluation dataset
└── conftest.py           # Shared pytest configuration
```

## Running Tests

### Prerequisites

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.app.txt
   ```

2. **For Docker Tests** (optional):
   ```bash
   docker --version
   docker-compose --version
   ```

### Quick Test Execution

#### Run All Basic Tests
```bash
python run_tests.py
```

#### Run Specific Test Suites
```bash
# Unit tests only
python run_tests.py --suite unit

# Integration tests
python run_tests.py --suite integration

# Acceptance criteria validation
python run_tests.py --suite acceptance
```

#### Run with Additional Options
```bash
# Include slow tests (evaluation, coverage)
python run_tests.py --slow

# Include Docker-based tests
python run_tests.py --docker

# Full comprehensive test suite
python run_tests.py --slow --docker
```

### Manual Test Execution

#### Unit Tests
```bash
python -m pytest tests/unit/ -v
```

#### Integration Tests
```bash
python -m pytest tests/integration/ -v -m integration
```

#### End-to-End Tests (requires Docker)
```bash
python -m pytest tests/e2e/ -v -m e2e
```

#### Quality Evaluation Tests
```bash
python -m pytest tests/evaluation/ -v -m evaluation
```

#### Acceptance Criteria Validation
```bash
python validate_acceptance_criteria.py
```

## Test Categories and Markers

### Test Markers
- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests  
- `@pytest.mark.e2e` - End-to-end tests
- `@pytest.mark.evaluation` - Quality evaluation tests
- `@pytest.mark.slow` - Tests that take longer to execute
- `@pytest.mark.requires_services` - Tests requiring external services

### Run by Markers
```bash
# Run only fast tests
python -m pytest -m "not slow"

# Run only tests that don't require services
python -m pytest -m "not requires_services"

# Run specific marker
python -m pytest -m "unit or integration"
```

## Test Configuration

### Environment Variables
Set these for consistent test behavior:
```bash
export QDRANT_URL="http://localhost:6333"
export GOOGLE_API_KEY="your_test_google_api_key"
export JINA_API_KEY="your_test_jina_api_key"
export PYEXEC_URL="http://localhost:8001"
export MAX_QUERY_LENGTH="2000"
export PYEXEC_TIMEOUT_SEC="5"
```

### Test Data
- **Q&A Dataset**: `tests/fixtures/qa_dataset.json` contains 20 synthetic questions for RAG evaluation
- **Mock Data**: Fixtures in `conftest.py` provide consistent test data

## Quality Metrics

### Coverage Goals
- Unit test coverage: >80%
- Integration test coverage: >60%
- Critical path coverage: >90%

### Performance Benchmarks
- **Response Latency**: P50 < 3.5 seconds
- **Precision@3**: >0.7 for retrieval
- **Groundedness**: >0.8 for responses

### Evaluation Criteria
From `qa_dataset.json`:
- Precision@K threshold: 0.7
- Recall@K threshold: 0.6  
- Groundedness threshold: 0.8
- Response time threshold: 3.0 seconds

## Docker Testing

### Test Environment
Uses `docker-compose.test.yml` with optimized settings:
- Faster timeouts
- Smaller resource limits
- Test-specific configurations

### Running Docker Tests
```bash
# Validate Docker configuration
docker-compose -f docker-compose.test.yml config

# Run full Docker test suite
python -m pytest tests/e2e/ -v -m e2e --tb=short

# Manual Docker testing
docker-compose -f docker-compose.test.yml up --build -d
# ... run tests ...
docker-compose -f docker-compose.test.yml down -v
```

## Acceptance Criteria Validation

The `validate_acceptance_criteria.py` script checks PRD requirements:

### Validated Criteria
1. **Service Availability**: All services respond to health checks
2. **Response Latency**: P50 response time < 3.5 seconds
3. **Memory Management**: 6-turn conversation limit
4. **Input Validation**: Dangerous inputs blocked, length limits enforced
5. **Security**: Mathematical execution sandboxed and limited
6. **Health Checks**: Docker services have proper health monitoring
7. **Resource Limits**: PDF size, chunk size, processing limits
8. **Configuration**: All required files and settings present

### Running Validation
```bash
# Full validation
python validate_acceptance_criteria.py

# Check specific requirements (modify script)
python validate_acceptance_criteria.py --check-latency
```

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**:
   ```bash
   pip install -r requirements.app.txt
   export PYTHONPATH=/path/to/project
   ```

2. **Docker Connection Errors**:
   ```bash
   docker --version
   docker-compose --version
   # Ensure Docker daemon is running
   ```

3. **Service Connection Timeouts**:
   - Check services are running: `docker-compose ps`
   - Increase timeout values in test configuration
   - Verify port mappings

4. **Test Data Issues**:
   - Ensure `tests/fixtures/qa_dataset.json` exists
   - Check file permissions and encoding

### Test Debugging

#### Verbose Output
```bash
python -m pytest tests/ -v -s --tb=long
```

#### Run Single Test
```bash
python -m pytest tests/unit/test_chunking.py::TestChunking::test_chunking_long_text -v -s
```

#### Debug Mode
```bash
python -m pytest --pdb tests/unit/test_chunking.py
```

## Continuous Integration

### Pre-commit Checks
```bash
# Run before committing
python run_tests.py --suite unit
python run_tests.py --suite integration  
python validate_acceptance_criteria.py
```

### CI Pipeline Suggestions
```yaml
# Example CI steps
- run: python run_tests.py --suite unit
- run: python run_tests.py --suite integration
- run: python run_tests.py --suite acceptance
- run: python run_tests.py --slow  # Optional for full builds
```

## Contributing

### Adding New Tests

1. **Unit Tests**: Add to appropriate file in `tests/unit/`
2. **Integration Tests**: Add to `tests/integration/`
3. **Evaluation Questions**: Add to `tests/fixtures/qa_dataset.json`

### Test Guidelines

- Use descriptive test names: `test_chunking_handles_empty_text`
- Include docstrings explaining test purpose
- Use appropriate markers: `@pytest.mark.unit`
- Mock external dependencies in unit tests
- Use fixtures for consistent test data

### Quality Standards

- All tests should be deterministic
- Clean up resources (temp files, connections)
- Test both happy path and edge cases
- Include performance assertions where relevant

## Reports and Artifacts

### Generated Files
- `test_execution_report.json` - Comprehensive test results
- `acceptance_criteria_report.json` - Validation results
- `htmlcov/` - Coverage report (when using --slow)

### Report Analysis
```bash
# View test results
cat test_execution_report.json | jq '.test_execution_summary'

# View acceptance criteria
cat acceptance_criteria_report.json | jq '.results'
```