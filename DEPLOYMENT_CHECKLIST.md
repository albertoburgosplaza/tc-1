# RAG Chatbot with Google Gemini - Production Deployment Checklist

## Pre-Deployment Requirements

### System Requirements
- [ ] **Docker Engine** >= 20.10.0
- [ ] **Docker Compose** >= 2.0.0
- [ ] **Available RAM**: Minimum 2GB (Recommended 4GB)
- [ ] **Available CPU**: Minimum 2 cores
- [ ] **Available Disk**: Minimum 5GB free space
- [ ] **Network**: Internet access for Google API
- [ ] **Google API Key**: Valid API key for Gemini access

### Security Prerequisites
- [ ] **Firewall Configuration**: Only ports 7860 and 8080 open externally
- [ ] **SSL/TLS Certificates**: Ready for HTTPS (if applicable)
- [ ] **Access Controls**: Define user access policies
- [ ] **Backup Strategy**: Database and volume backup procedures defined

## Pre-Deployment Validation

### 1. Environment Setup (Time: 5-10 minutes)
```bash
# Clone repository
git clone <repository-url>
cd turingchallenge-reto-1

# Verify system requirements
docker --version
docker-compose --version
df -h  # Check disk space
free -h  # Check memory
```

- [ ] Repository cloned successfully
- [ ] Docker versions verified
- [ ] System resources adequate

### 2. Configuration Review (Time: 5 minutes)
```bash
# Review configuration files
cat docker-compose.yml
cat .env  # If exists
```

- [ ] **docker-compose.yml** reviewed and appropriate for environment
- [ ] **Environment variables** configured for production
- [ ] **Resource limits** appropriate for available hardware
- [ ] **Security settings** enabled (non-root users, network isolation)

### 3. Document Preparation (Time: 2-5 minutes)
```bash
# Ensure demo documents are available
ls -la docs/
```

- [ ] **Demo corpus** available in `/docs` directory
- [ ] **Document formats** supported (PDF)
- [ ] **Total size** under configured limits (100MB per file)

## Deployment Process

### 4. Initial System Build (Time: 15-25 minutes)
**Expected Time: < 30 minutes (PRD Requirement)**

```bash
# Start deployment timer
DEPLOY_START_TIME=$(date +%s)

# Build and start services
docker-compose up --build -d

# Wait for all services to be healthy
timeout 1800 bash -c 'until docker-compose ps | grep -q "healthy"; do sleep 10; done'

# Calculate deployment time
DEPLOY_END_TIME=$(date +%s)
echo "Deployment time: $((DEPLOY_END_TIME - DEPLOY_START_TIME)) seconds"
```

- [ ] **All services built** successfully
- [ ] **All containers started** and marked as healthy
- [ ] **Deployment time** < 30 minutes (PRD KPI)
- [ ] **No error logs** in container outputs

### 5. Service Health Verification (Time: 2-3 minutes)
```bash
# Check service status
docker-compose ps

# Check service logs
docker-compose logs qdrant | tail -20
# Google Gemini uses API, check app logs for API status
docker-compose logs pyexec | tail -20
docker-compose logs rag_app | tail -20
```

- [ ] **Qdrant** status: healthy, port 6333 accessible
- [ ] **Google Gemini API** configured with valid API key  
- [ ] **PyExec** status: healthy, running as non-root user
- [ ] **RAG App** status: healthy, ports 7860 & 8080 accessible
- [ ] **No critical errors** in service logs

### 6. Document Ingestion Test (Time: 1-3 minutes)
**Expected Time: < 2 minutes (PRD Requirement)**

```bash
# Start ingestion timer
INGEST_START_TIME=$(date +%s)

# Run document ingestion
docker-compose run --rm ingest

# Calculate ingestion time
INGEST_END_TIME=$(date +%s)
echo "Ingestion time: $((INGEST_END_TIME - INGEST_START_TIME)) seconds"

# Verify documents were ingested
docker-compose exec qdrant curl http://localhost:6333/collections/rag_multimodal
```

- [ ] **Ingestion completed** without errors
- [ ] **Ingestion time** < 2 minutes (PRD KPI)
- [ ] **Document count** matches expected corpus size
- [ ] **Vector collection** created successfully in Qdrant

## Post-Deployment Validation

### 7. Functional Testing (Time: 5-10 minutes)
```bash
# Test basic functionality
curl -f http://localhost:8080/health
curl -f http://localhost:7860/  # Should return Gradio interface

# Test pyexec service
curl -X POST http://localhost:8001/execute \
     -H "Content-Type: application/json" \
     -d '{"expression": "2 + 2"}'
```

- [ ] **Health endpoints** responding correctly
- [ ] **Gradio interface** accessible and loading
- [ ] **Mathematical execution** working securely
- [ ] **Basic RAG queries** returning results

### 8. Performance Validation (Time: 5-10 minutes)
**Expected Latency: P50 < 3.5s (PRD Requirement)**

```bash
# Run acceptance criteria validation
python3 validate_acceptance_criteria.py

# Check validation report
cat acceptance_criteria_report.json
```

- [ ] **Response latency P50** < 3.5 seconds
- [ ] **Memory management** within limits (6 turns, 8000 chars)
- [ ] **Input validation** properly configured
- [ ] **Security measures** functioning correctly

### 9. Security Validation (Time: 3-5 minutes)
```bash
# Verify security configurations
docker inspect pyexec | grep -E '"User"|ReadonlyRootfs|CapDrop|CapAdd'

# Test network isolation
docker-compose exec pyexec ping -c 1 8.8.8.8  # Should fail for isolated services
```

- [ ] **PyExec user** is non-root (pyexec)
- [ ] **Readonly filesystem** enabled for pyexec
- [ ] **Capabilities** properly dropped/added
- [ ] **Network isolation** functioning (pyexec cannot reach external)
- [ ] **Resource limits** enforced

### 10. Quality Assurance (Time: 10-15 minutes)
```bash
# Run comprehensive test suite (if available)
python3 run_tests.py --suite unit
python3 run_tests.py --suite integration
```

- [ ] **Unit tests** passing
- [ ] **Integration tests** passing
- [ ] **RAG quality** meets requirements (90% valid citations)
- [ ] **End-to-end flows** functioning correctly

## Production Monitoring Setup

### 11. Monitoring Configuration (Time: 5 minutes)
```bash
# Set up log monitoring
docker-compose logs -f &

# Monitor resource usage
docker stats

# Set up alerts (implementation specific)
```

- [ ] **Log aggregation** configured
- [ ] **Resource monitoring** active
- [ ] **Health check alerts** configured
- [ ] **Performance monitoring** enabled

## Post-Deployment Documentation

### 12. Final Documentation (Time: 5 minutes)
```bash
# Document deployment details
echo "Deployment completed at: $(date)" >> deployment.log
echo "System configuration:" >> deployment.log
docker-compose config >> deployment.log
```

- [ ] **Deployment timestamp** recorded
- [ ] **Configuration snapshot** saved
- [ ] **Performance baseline** documented
- [ ] **Access credentials** securely stored

## Rollback Procedures

### Emergency Rollback (Time: 2-5 minutes)
```bash
# Stop services
docker-compose down

# Remove volumes if needed (CAUTION: Data loss)
docker-compose down -v

# Restore from backup (if available)
# docker-compose up -d
```

- [ ] **Rollback procedure** tested and documented
- [ ] **Backup restoration** procedure validated
- [ ] **Emergency contacts** notified if needed

## Success Criteria Summary

### Performance KPIs (PRD Requirements)
- [ ] ✅ **Setup Time**: < 30 minutes
- [ ] ✅ **Ingestion Time**: < 2 minutes for demo corpus
- [ ] ✅ **Response Latency**: P50 < 3.5 seconds
- [ ] ✅ **Citation Quality**: 90% responses with valid citations

### Security Requirements
- [ ] ✅ **Non-root execution**: PyExec service
- [ ] ✅ **Network isolation**: Internal communication only
- [ ] ✅ **Input validation**: Dangerous inputs blocked
- [ ] ✅ **Resource limits**: CPU/Memory/PID controls active

### Operational Requirements
- [ ] ✅ **Service health**: All services healthy and responding
- [ ] ✅ **Data persistence**: Qdrant and Ollama data persisted
- [ ] ✅ **Log collection**: Centralized logging functional
- [ ] ✅ **Monitoring**: Resource and performance monitoring active

## Troubleshooting Guide

### Common Issues

1. **Services not starting**
   ```bash
   # Check resource availability
   docker system df
   docker system prune  # Clean up if needed
   
   # Check logs for specific errors
   docker-compose logs [service_name]
   ```

2. **Slow performance**
   ```bash
   # Check resource usage
   docker stats
   
   # Verify CPU/Memory limits
   docker-compose config
   ```

3. **Connection issues**
   ```bash
   # Verify network connectivity
   docker network ls
   docker-compose exec app ping qdrant
   ```

4. **PyExec security errors**
   ```bash
   # Verify user configuration
   docker-compose exec pyexec id
   
   # Check filesystem permissions
   docker-compose exec pyexec ls -la /app
   ```

### Emergency Contacts
- **System Administrator**: [Contact Info]
- **Technical Lead**: [Contact Info]
- **On-Call Engineer**: [Contact Info]

---

**Deployment Checklist Version**: 1.0  
**Last Updated**: $(date)  
**Next Review**: [Schedule quarterly review]