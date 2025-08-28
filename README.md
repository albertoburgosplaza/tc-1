
# Chatbot RAG (OSS/local) — Microservicios con Docker Compose

Sistema de chatbot local con Retrieval Augmented Generation (RAG) que permite consultar documentos PDF y ejecutar código Python de forma segura. El sistema utiliza una arquitectura de microservicios con Docker Compose para facilitar el despliegue y escalabilidad.

## 🏗️ Arquitectura

**Servicios principales:**
- **qdrant**: Base de datos vectorial para almacenamiento de embeddings
- **ollama**: Motor de LLM local (soporta múltiples modelos)
- **pyexec**: Microservicio para ejecutar expresiones Python de forma segura
- **app**: Interfaz web con Gradio + LangChain que orquesta RAG y ejecuta Python
- **ingest**: Job de procesamiento para ingestar PDFs en Qdrant

## ⚡ Inicio Rápido

### Requisitos del Sistema

**Obligatorios:**
- Docker (versión 20.10 o superior)
- Docker Compose (versión 2.0 o superior)
- Mínimo 4GB RAM disponible (2GB para llama3.2:1b + servicios)
- Al menos 8GB de espacio en disco (reducido por llama3.2:1b)

**Verificar instalación:**
```bash
docker --version && docker compose version
```

### Instalación Automática

Para un setup completo en menos de 30 minutos:

1. **Clonar y preparar:**
   ```bash
   git clone <repository-url>
   cd turingchallenge-reto-1
   chmod +x setup.sh
   ./setup.sh
   ```

### Instalación Manual

#### 1. Preparar Documentos

Crear carpeta de documentos y añadir PDFs:
```bash
# Crear carpeta
mkdir -p docs

# Añadir tus PDFs (ejemplos)
cp /ruta/a/tus/*.pdf docs/
# O descargar documentos de ejemplo
wget -P docs/ "https://example.com/sample.pdf"
```

**Limitaciones:**
- Tamaño máximo por PDF: 100MB
- Formatos soportados: PDF únicamente
- Mínimo contenido por documento: 10 caracteres

#### 2. Levantar Servicios Base

```bash
# Construir y levantar servicios (orden importante)
docker compose up -d --build qdrant ollama pyexec app

# Verificar que los servicios están saludables
docker compose ps
```

**Tiempo estimado:** 5-10 minutos (primera vez)

#### 3. Descargar Modelo LLM

```bash
# Modelo configurado por defecto (1B parámetros, ~1.3GB)
docker exec -it ollama ollama pull llama3.2:1b

# Modelos alternativos más potentes
docker exec -it ollama ollama pull mistral:7b-instruct      # ~4GB
docker exec -it ollama ollama pull llama3.1:8b-instruct     # ~4.7GB

# Verificar modelo descargado
docker exec -it ollama ollama list
```

**Tiempo estimado:** 2-5 minutos para llama3.2:1b (según conexión)

#### 📋 Sobre el modelo llama3.2:1b

**Ventajas:**
- 🚀 **Velocidad:** Respuestas ultra-rápidas (<2 segundos)
- 💾 **Eficiencia:** Solo ~1.3GB de almacenamiento
- ⚡ **Recursos:** Funciona con 2GB RAM
- 🔧 **Optimización:** Ideal para RAG y consultas directas

**Limitaciones:**
- Capacidades de razonamiento reducidas vs modelos 7B+
- Mejor para consultas factuales que para análisis complejos
- Recomendado cambiar a mistral:7b-instruct para tareas avanzadas

#### 4. Procesar Documentos

```bash
# Ejecutar ingesta de PDFs
docker compose run --rm ingest

# Verificar ingesta exitosa
docker logs ingest
```

**Tiempo estimado:** 1-5 minutos (según cantidad de PDFs)

#### 5. Acceder a la Interfaz

- **Interfaz principal:** http://localhost:7860
- **API health check:** http://localhost:8080/health
- **Qdrant dashboard:** http://localhost:6333/dashboard

## 🎯 Guía de Uso

### Consultas RAG (Documentos)

**Ejemplos de preguntas:**
```
- "¿Qué dice el documento sobre inteligencia artificial?"
- "Resume los puntos principales del capítulo 3"
- "¿Cuáles son las conclusiones del estudio?"
- "Compara las ventajas de Random Forest vs Redes Neuronales"
```

**Características:**
- Respuestas incluyen citas con documento y página
- Búsqueda semántica en el contenido
- Soporte para consultas en múltiples idiomas
- Análisis comparativo entre documentos

### Ejecución de Código Python

**Sintaxis soportada:**
```python
# Operaciones matemáticas
python: (2+3)*10
python: import math; math.sqrt(16)

# Análisis estadístico
python: import statistics; statistics.mean([1,2,3,4,5])

# Análisis de datos
python: [1,2,3,4,5]; sum(_)/len(_)

# Consultas en lenguaje natural
"Calcula la media de 1, 5, 10"
"¿Cuánto es 15% de 200?"
"Calcula el interés compuesto de $5000 al 3.5% por 7 años"
```

**Limitaciones de seguridad:**
- Timeout: 5 segundos
- Máximo 500 caracteres por expresión
- Sin acceso a archivos del sistema
- Bibliotecas limitadas (math, statistics, básicas)

### Consultas Mixtas

Puedes combinar ambas funcionalidades:
```
"Según el documento financiero, calcula el ROI si la inversión es de $100,000 y los ingresos son python: 150000 * 1.2"

"El paper menciona una muestra de 200 participantes. ¿Es estadísticamente significativo con un margen de error del 3.5%? python: 1.96/0.035**2 * 0.25"
```

### Ejemplos Completos

Para ejemplos detallados con respuestas esperadas, consulta **[EXAMPLES.md](EXAMPLES.md)** que incluye:
- 📊 Casos de análisis financiero con cálculos
- 🔬 Validación de datos de investigación
- 💬 Conversaciones multi-turno con memoria contextual
- 🧮 Análisis estadístico de datasets
- ⚠️ Manejo de errores y casos límite

## ⚙️ Configuración Avanzada

### Variables de Entorno Principales

#### Servicio App
```yaml
# Configuración del LLM
LLM_MODEL: llama3.2:1b                   # Modelo de Ollama (1B parámetros, optimizado para velocidad)
LLM_PROVIDER: ollama                     # Proveedor: ollama o google
OLLAMA_BASE_URL: http://ollama:11434     # URL interna de Ollama
GOOGLE_API_KEY: "${GOOGLE_API_KEY:-}"    # API Key de Google AI (para Gemini)

# Configuración de Qdrant
QDRANT_URL: http://qdrant:6333           # URL interna de Qdrant
COLLECTION_NAME: corpus_pdf              # Nombre de la colección vectorial

# Configuración de la interfaz
GRADIO_SERVER_NAME: 0.0.0.0             # Host de Gradio
GRADIO_SERVER_PORT: 7860                 # Puerto de Gradio

# Validación de entrada
MAX_HISTORY_CHARS: 8000                  # Máximo caracteres en historial
MAX_QUERY_LENGTH: 2000                   # Máximo caracteres por consulta
MIN_QUERY_LENGTH: 1                      # Mínimo caracteres por consulta

# Integración Python
PYEXEC_URL: http://pyexec:8001           # URL del servicio Python
```

#### Servicio PyExec
```yaml
PYEXEC_TIMEOUT_SEC: 5                    # Timeout para ejecución Python
MAX_EXPR_LENGTH: 500                     # Máximo caracteres en expresión
MAX_EXPR_COMPLEXITY: 100                 # Límite de complejidad AST
```

#### Servicio Ingest
```yaml
# Configuración de embeddings
EMBEDDING_MODEL: sentence-transformers/all-MiniLM-L6-v2
DOCUMENTS_DIR: /app/docs                 # Carpeta interna de documentos

# Procesamiento de PDFs
MAX_PDF_SIZE_MB: 100                     # Tamaño máximo por PDF
CHUNK_SIZE: 1200                         # Tamaño de chunks en caracteres
CHUNK_OVERLAP: 180                       # Solapamiento entre chunks
MIN_CONTENT_LENGTH: 10                   # Contenido mínimo por chunk
```

### Cambiar Modelo LLM

#### Opción 1: Usar Gemini 2.5 Flash Lite (Google AI)

1. **Configurar API Key:**
   ```bash
   # Crear archivo .env
   echo "GOOGLE_API_KEY=your_google_api_key_here" >> .env
   ```

2. **Usar desde la interfaz web:**
   - Accede a http://localhost:7860
   - Selecciona "google" en el selector "Proveedor LLM"
   - El sistema cambiará automáticamente a Gemini 2.5 Flash Lite

**Ventajas de Gemini:**
- ⚡ **Ultra velocidad:** Modelo optimizado para respuestas rápidas
- 🌐 **Sin instalación local:** No requiere descargar modelos
- 🔧 **Auto-configurado:** Listo para usar con tu API key

#### Opción 2: Usar modelos Ollama locales

1. **Descargar nuevo modelo:**
   ```bash
   # Modelo más potente (recomendado para tareas complejas)
   docker exec -it ollama ollama pull mistral:7b-instruct
   # o
   docker exec -it ollama ollama pull llama3.1:8b-instruct
   ```

2. **Actualizar configuración:**
   ```bash
   # Editar docker-compose.yml
   # Cambiar LLM_MODEL: mistral:7b-instruct
   # (por defecto está configurado llama3.2:1b para velocidad)
   ```

3. **Reiniciar aplicación:**
   ```bash
   docker compose restart app
   ```

### Cambiar Modelo de Embeddings

⚠️ **Importante:** Cambiar embeddings requiere re-ingestión completa

1. **Editar configuración:**
   ```yaml
   # En docker-compose.yml servicio ingest
   EMBEDDING_MODEL: sentence-transformers/all-mpnet-base-v2
   ```

2. **Actualizar dimensiones en código:**
   ```python
   # En ingest.py, ajustar:
   # all-MiniLM-L6-v2: 384 dimensiones
   # all-mpnet-base-v2: 768 dimensiones
   ```

3. **Limpiar y re-ingestar:**
   ```bash
   docker compose down -v  # Elimina volúmenes
   docker compose up -d qdrant
   docker compose run --rm ingest
   ```

## 🔧 Troubleshooting

### Problemas Comunes

#### 1. Servicios no inician

**Error:** `docker compose up` falla
```bash
# Verificar logs
docker compose logs [servicio]

# Soluciones comunes
docker system prune -f              # Limpiar Docker
docker compose down -v              # Eliminar volúmenes
docker compose pull                 # Actualizar imágenes
```

#### 2. Modelo no descarga

**Error:** `ollama pull` falla o es muy lento
```bash
# Verificar espacio en disco
df -h

# El modelo configurado por defecto ya es muy ligero
docker exec -it ollama ollama pull llama3.2:1b

# Verificar conectividad
docker exec -it ollama curl -I https://ollama.ai
```

#### 3. Ingesta falla

**Error:** PDFs no se procesan
```bash
# Verificar logs detallados
docker compose logs ingest

# Verificar PDFs
ls -la docs/                        # Comprobar archivos
file docs/*.pdf                     # Verificar formato

# Reintentar ingesta
docker compose run --rm ingest
```

#### 4. Interfaz web no carga

**Error:** http://localhost:7860 no responde
```bash
# Verificar estado de servicios
docker compose ps

# Verificar salud de app
curl -f http://localhost:8080/health

# Revisar logs
docker compose logs app

# Reiniciar servicio
docker compose restart app
```

#### 5. Errores de permisos

**Error:** Permission denied al crear carpetas
```bash
# En Linux/WSL
sudo chown -R $USER:$USER docs/
chmod -R 755 docs/

# Alternativa con Docker
docker run --rm -v $(pwd)/docs:/docs alpine chown -R 1000:1000 /docs
```

### Logs y Monitoreo

**Ver logs en tiempo real:**
```bash
# Todos los servicios
docker compose logs -f

# Servicio específico
docker compose logs -f app

# Filtrar por nivel
docker compose logs app | grep ERROR
```

**Verificar salud de servicios:**
```bash
# Estado general
docker compose ps

# Health checks específicos
docker inspect --format='{{.State.Health.Status}}' qdrant
docker inspect --format='{{.State.Health.Status}}' ollama
```

### Rendimiento

**Optimizar memoria:**
```bash
# Limitar memoria de Ollama
docker update --memory=2g ollama

# Monitorear uso
docker stats
```

**Optimizar velocidad:**
- llama3.2:1b ya está optimizado para velocidad por defecto
- Para más potencia: cambiar a mistral:7b-instruct
- Ajustar CHUNK_SIZE según hardware (default: 1200)
- Considerar SSD para volúmenes Docker

## 🧪 Testing

```bash
# Ejecutar tests completos
python run_tests.py

# Tests específicos
pytest tests/unit/                     # Tests unitarios
pytest tests/integration/              # Tests de integración
pytest tests/e2e/                      # Tests end-to-end

# Test de aceptación
python validate_acceptance_criteria.py
```

## 📊 Monitoreo y Métricas

**Endpoints de salud:**
- App: http://localhost:8080/health
- Qdrant: http://localhost:6333/healthz
- Ollama: http://localhost:11434/api/tags
- PyExec: http://localhost:8001/health

**Métricas disponibles:**
- Qdrant: Dashboard en http://localhost:6333/dashboard
- Docker: `docker compose top` y `docker stats`

## 🚀 Despliegue en Producción

### Consideraciones de Seguridad

1. **Cambiar puertos por defecto:**
   ```yaml
   ports:
     - "7861:7860"  # En lugar de 7860:7860
   ```

2. **Variables de entorno sensibles:**
   ```bash
   # Crear .env file
   echo "GRADIO_AUTH=admin:password" > .env
   ```

3. **Firewall y reverse proxy:**
   ```bash
   # Solo exponer puertos necesarios
   # Usar nginx/traefik como proxy
   ```

### Escalabilidad

```yaml
# En docker-compose.yml
deploy:
  replicas: 2
  resources:
    limits:
      memory: 2G
    reservations:
      memory: 1G
```

## 🤝 Desarrollo y Contribución

**Estructura del proyecto:**
```
├── app.py              # Aplicación principal Gradio
├── ingest.py           # Procesamiento de PDFs
├── pyexec_service.py   # Servicio de ejecución Python
├── docker-compose.yml  # Configuración de servicios
├── Dockerfile          # Imagen multi-etapa
├── requirements.*.txt  # Dependencias por servicio
├── tests/              # Suite de testing
└── docs/              # Documentos para ingestar
```

**Flujo de desarrollo:**
1. Fork del repositorio
2. Crear rama feature/nombre-feature
3. Desarrollar con tests
4. Ejecutar suite completa: `python run_tests.py`
5. Crear Pull Request

**Estándares de código:**
- Python: Black, isort, flake8
- Dockerfiles: Hadolint
- Documentación: Markdown lint

## 📄 Licencia

[Especificar licencia del proyecto]

## 🆘 Soporte

- **Issues:** [GitHub Issues](link-to-issues)
- **Documentación:** [Wiki](link-to-wiki)
- **Chat:** [Discord/Slack](link-to-chat)

---

**Última actualización:** Agosto 2024 (actualizado modelo llama3.2:1b)
**Versión:** 1.0.0
