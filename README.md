
# Chatbot RAG con Google Gemini — Microservicios con Docker Compose

Sistema de chatbot con Retrieval Augmented Generation (RAG) **multimodal** que permite consultar documentos PDF con texto e imágenes, y ejecutar código Python de forma segura. Utiliza Google Gemini como modelo de lenguaje, Jina embeddings para procesamiento multimodal y una arquitectura de microservicios con Docker Compose para facilitar el despliegue y escalabilidad.

## 🏗️ Arquitectura

**Servicios principales:**
- **qdrant**: Base de datos vectorial para almacenamiento de embeddings
- **pyexec**: Microservicio para ejecutar expresiones Python de forma segura
- **app**: Interfaz web con Gradio + LangChain + Google Gemini que orquesta RAG
- **ingest**: Job de procesamiento para ingestar PDFs en Qdrant con embeddings multimodales de Jina (texto + imágenes)

## ⚡ Inicio Rápido

### Requisitos del Sistema

**Obligatorios:**
- Docker (versión 20.10 o superior)
- Docker Compose (versión 2.0 o superior)
- **Google API Key** (para acceso a Gemini LLM)
- **Jina API Key** (para embeddings y reranking)
- Mínimo 2GB RAM disponible
- Al menos 2GB de espacio en disco

**Verificar instalación:**
```bash
docker --version && docker compose version
```

### Instalación Automática

Para un setup completo en menos de 5 minutos:

1. **Configurar API Keys:**
   ```bash
   export GOOGLE_API_KEY="tu_google_api_key_aqui"
   export JINA_API_KEY="tu_jina_api_key_aqui"
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
- **Imágenes:** Máximo 1024x1024px, 5MB por imagen
- **Procesamiento:** Las imágenes se extraen automáticamente durante la ingesta

#### 2. Configurar API Keys

```bash
# Configurar variables de entorno
export GOOGLE_API_KEY="tu_google_api_key_aqui"
export JINA_API_KEY="tu_jina_api_key_aqui"

# O crear archivo .env
echo "GOOGLE_API_KEY=tu_google_api_key_aqui" > .env
echo "JINA_API_KEY=tu_jina_api_key_aqui" >> .env
```

**Cómo obtener las API Keys:**

**Google API Key (para Gemini LLM):**
1. Ir a [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Crear una nueva API key
3. Habilitar Generative AI API

**Jina API Key (para embeddings y reranking):**
1. Ir a [Jina AI](https://jina.ai/)
2. Crear una cuenta gratuita
3. Obtener tu API key desde el dashboard
4. Verificar los límites en [Jina Pricing](https://jina.ai/pricing)

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
# Ejecutar ingesta completa de todos los PDFs
docker compose run --rm ingest

# O procesar un solo PDF sin rehacer toda la colección
python add_single_pdf.py docs/nuevo_documento.pdf

# Verificar ingesta exitosa
docker logs ingest
```

**Tiempo estimado:** 1-5 minutos (según cantidad de PDFs)

**📝 Procesamiento individual:**
Para añadir documentos nuevos sin reprocesar toda la colección, utiliza el script `add_single_pdf.py`:
- ✅ Más rápido para documentos individuales
- ✅ No requiere Docker (pero sí las variables de entorno)
- ✅ Validaciones automáticas de formato y conexión
- ✅ Preserva documentos ya procesados

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
- 📝 **Citas mejoradas:** Referencias inline con documento, página y contexto detallado
- 🔍 **Búsqueda semántica:** Vectores multimodales de Jina para alta precisión
- 🌍 **Soporte multilingüe:** Consultas en múltiples idiomas nativamente
- 🔄 **Análisis comparativo:** Entre documentos y secciónes
- 🎯 **Reranking inteligente:** Jina Rerank optimiza relevancia de resultados
- 📊 **Mapeo de fuentes:** Identificación precisa de documentos citados

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

### Consultas Multimodales (Texto + Imágenes)

El sistema ahora soporta **RAG multimodal** que procesa tanto texto como imágenes extraídas de los PDFs:

**Características multimodales:**
- 🖼️ **Extracción de imágenes:** Automática desde PDFs durante la ingesta
- 🔍 **Búsqueda visual:** Encuentra imágenes relevantes según consultas de texto
- 🤝 **Contexto híbrido:** Combina información textual y visual en las respuestas
- 🎯 **Reranking inteligente:** Jina Rerank mejora la relevancia de resultados mixtos

**Ejemplos de consultas multimodales:**
```
- "¿Qué gráficos muestran la tendencia de ventas?"
- "Encuentra diagramas sobre arquitectura de microservicios"
- "Muestra las imágenes relacionadas con el proceso de manufactura"
- "Compara las tablas de resultados experimentales"
- "¿Hay capturas de pantalla de la interfaz de usuario?"
```

**Configuración multimodal:**
```yaml
ENABLE_IMAGE_INGEST: true              # Habilitar procesamiento de imágenes
IMAGE_MAX_SIDE_PX: 1024                # Tamaño máximo de imagen (píxeles)
IMAGE_MAX_BYTES: 5242880               # Tamaño máximo (5MB)
TOP_K_SEARCH: 50                       # Búsqueda inicial
TOP_N_RERANK: 10                       # Resultados finales tras reranking
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
COLLECTION_NAME: rag_multimodal          # Nombre de la colección vectorial multimodal

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
# Configuración de embeddings (Jina)
EMBEDDING_MODEL: jina-embeddings-v4     # Modelo de Jina para embeddings
EMBEDDING_PROVIDER: jina                # Proveedor de embeddings
JINA_API_KEY: "${JINA_API_KEY}"          # API Key de Jina (REQUERIDA)
DOCUMENTS_DIR: /app/docs                # Carpeta interna de documentos

# Procesamiento de PDFs
MAX_PDF_SIZE_MB: 100                     # Tamaño máximo por PDF
CHUNK_SIZE: 1200                         # Tamaño de chunks en caracteres
CHUNK_OVERLAP: 180                       # Solapamiento entre chunks
MIN_CONTENT_LENGTH: 10                   # Contenido mínimo por chunk

# Configuración multimodal
ENABLE_IMAGE_INGEST: true                # Habilitar extracción de imágenes
IMAGE_MAX_SIDE_PX: 1024                  # Tamaño máximo de imagen (píxeles)
IMAGE_MAX_BYTES: 5242880                 # Tamaño máximo de imagen (5MB)

# Configuración de reranking y búsqueda
JINA_RERANK_MODEL: jina-rerank-m0        # Modelo de reranking de Jina
TOP_K_SEARCH: 50                         # Resultados iniciales de búsqueda
TOP_N_RERANK: 10                         # Resultados finales tras reranking
PARALLEL_RETRIEVAL: true                 # Habilitar recuperación paralela
MAX_RETRIEVAL_WORKERS: 2                 # Máximo workers para procesamiento paralelo
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
# API Keys necesarias
export GOOGLE_API_KEY="tu_google_api_key_aqui"    # Para Gemini LLM
export JINA_API_KEY="tu_jina_api_key_aqui"        # Para embeddings y reranking
```

### Modelo de Embeddings Configurado

El sistema utiliza **Jina `jina-embeddings-v4`** para generar embeddings:

**Características:**
- 📊 **Alta calidad:** Embeddings de última generación optimizados para RAG multimodal
- 🚀 **Rápido:** Generación via API de Jina sin procesamiento local
- 🔄 **Consistente:** Embeddings estables entre sesiones
- 🌍 **Multimodal:** Soporte para texto e imágenes con el mismo modelo

**Configuración actual:**
```yaml
# En docker-compose.yml servicio ingest
EMBEDDING_MODEL: jina-embeddings-v4
EMBEDDING_PROVIDER: jina
JINA_API_KEY: "${JINA_API_KEY}"
```

**Reranker incluido:**
El sistema también utiliza **Jina Rerank M0** para mejorar la relevancia de los resultados:
```yaml
JINA_RERANK_MODEL: jina-rerank-m0
TOP_K_SEARCH: 50      # Búsqueda inicial
TOP_N_RERANK: 10      # Resultados finales tras reranking
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

#### 2. Errores de API Keys

**Error:** Fallos de autenticación o límites de API
```bash
# Verificar variables de entorno
echo $GOOGLE_API_KEY
echo $JINA_API_KEY

# Comprobar conectividad con APIs
curl -H "Authorization: Bearer $JINA_API_KEY" https://api.jina.ai/v1/embeddings

# Verificar logs de servicios
docker compose logs app | grep -i "api\|error"
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
docker inspect --format='{{.State.Health.Status}}' pyexec
```

### Rendimiento

**Optimizar memoria:**
```bash
# Monitorear uso de memoria
docker stats

# Limitar memoria de servicios si es necesario
docker update --memory=1g qdrant
docker update --memory=512m pyexec
```

**Optimizar velocidad:**
- Google Gemini 2.5 Flash Lite ya está optimizado para velocidad
- Jina embeddings v4 proporciona alta velocidad via API
- Habilitar procesamiento paralelo: `PARALLEL_RETRIEVAL=true`
- Ajustar CHUNK_SIZE según hardware (default: 1200)
- Considerar SSD para volúmenes Docker
- Optimizar reranking: ajustar `TOP_K_SEARCH` y `TOP_N_RERANK`

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
└── docs/               # Documentos para ingestar
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

**Última actualización:** Agosto 2025 (multimodal RAG con Jina embeddings y reranking)
**Versión:** 2.0.0
