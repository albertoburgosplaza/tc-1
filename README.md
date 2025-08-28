
# Chatbot RAG con Google Gemini — Microservicios con Docker Compose

Sistema de chatbot con Retrieval Augmented Generation (RAG) que permite consultar documentos PDF y ejecutar código Python de forma segura. Utiliza Google Gemini como modelo de lenguaje y una arquitectura de microservicios con Docker Compose para facilitar el despliegue y escalabilidad.

## 🏗️ Arquitectura

**Servicios principales:**
- **qdrant**: Base de datos vectorial para almacenamiento de embeddings
- **pyexec**: Microservicio para ejecutar expresiones Python de forma segura
- **app**: Interfaz web con Gradio + LangChain + Google Gemini que orquesta RAG
- **ingest**: Job de procesamiento para ingestar PDFs en Qdrant con embeddings de Google

## ⚡ Inicio Rápido

### Requisitos del Sistema

**Obligatorios:**
- Docker (versión 20.10 o superior)
- Docker Compose (versión 2.0 o superior)
- **Google API Key** (para acceso a Gemini)
- Mínimo 2GB RAM disponible
- Al menos 2GB de espacio en disco

**Verificar instalación:**
```bash
docker --version && docker compose version
```

### Instalación Automática

Para un setup completo en menos de 5 minutos:

1. **Configurar API Key:**
   ```bash
   export GOOGLE_API_KEY="tu_api_key_aqui"
   ```

2. **Clonar y preparar:**
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

#### 2. Configurar Google API Key

```bash
# Configurar variable de entorno
export GOOGLE_API_KEY="tu_google_api_key_aqui"

# O crear archivo .env
echo "GOOGLE_API_KEY=tu_google_api_key_aqui" > .env
```

**Cómo obtener Google API Key:**
1. Ir a [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Crear una nueva API key
3. Habilitar Generative AI API

#### 3. Levantar Servicios

```bash
# Construir y levantar servicios
docker compose up -d --build qdrant pyexec app

# Verificar que los servicios están saludables
docker compose ps
```

**Tiempo estimado:** 3-5 minutos (primera vez)

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
# Configuración del LLM (solo Google Gemini)
LLM_PROVIDER: google                     # Único proveedor soportado
GOOGLE_API_KEY: "${GOOGLE_API_KEY}"      # API Key de Google AI (REQUERIDA)

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
# Configuración de embeddings (Google)
EMBEDDING_MODEL: models/embedding-001    # Modelo de Google para embeddings
EMBEDDING_PROVIDER: google              # Proveedor de embeddings
DOCUMENTS_DIR: /app/docs                # Carpeta interna de documentos

# Procesamiento de PDFs
MAX_PDF_SIZE_MB: 100                     # Tamaño máximo por PDF
CHUNK_SIZE: 1200                         # Tamaño de chunks en caracteres
CHUNK_OVERLAP: 180                       # Solapamiento entre chunks
MIN_CONTENT_LENGTH: 10                   # Contenido mínimo por chunk
```

### Modelo LLM Configurado

Este sistema utiliza únicamente **Google Gemini 2.5 Flash Lite** como modelo de lenguaje:

**Características:**
- ⚡ **Ultra velocidad:** Respuestas en milisegundos
- 🌐 **API-based:** Sin modelos locales, menor uso de recursos
- 🔧 **Pre-configurado:** Listo para usar con tu Google API Key
- 🌍 **Multilengüe:** Soporte nativo para múltiples idiomas

**Configuración requerida:**
```bash
# Solo necesitas tu Google API Key
export GOOGLE_API_KEY="tu_api_key_aqui"
```

### Modelo de Embeddings Configurado

El sistema utiliza **Google `models/embedding-001`** para generar embeddings:

**Características:**
- 📊 **Alta calidad:** 768 dimensiones optimizadas para RAG
- 🚀 **Rápido:** Generación via API sin procesamiento local
- 🔄 **Consistente:** Embeddings estables entre sesiones

**Configuración actual:**
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
