
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
MAX_QUERY_LENGTH: 2000                   # Máximo caracteres por consulta
MIN_QUERY_LENGTH: 1                      # Mínimo caracteres por consulta

# Gestión de Memoria y Conversaciones
MAX_HISTORY_CHARS: 8000                  # Límite de caracteres antes de resumir historial
SLIDING_WINDOW_TURNS: 6                  # Turnos recientes preservados tras resumen

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

### Sistema de Gestión de Memoria Contextual

El chatbot implementa un **sistema automático de gestión de memoria** que mantiene conversaciones largas sin perder contexto relevante:

#### Funcionamiento Automático
```python
# Monitoreo continuo del tamaño del contexto
MAX_HISTORY_CHARS: 8000                 # Límite antes de activar resumen
SLIDING_WINDOW_TURNS: 6                 # Turnos recientes siempre preservados

# Trigger automático cuando se excede el límite:
# 1. Analiza toda la conversación
# 2. Extrae información crítica (nombres, fechas, decisiones, números)
# 3. Genera resumen inteligente con LLM
# 4. Conserva últimos 6 turnos + resumen optimizado
```

#### Información Preservada Automáticamente
- **📅 Fechas**: Formatos YYYY-MM-DD y DD/MM/YYYY
- **👤 Nombres propios**: Personas, lugares, organizaciones
- **⚖️ Decisiones**: Preferencias expresadas ("prefiero X", "decidí Y")
- **⚙️ Configuraciones**: Comandos y ajustes técnicos
- **🔢 Números importantes**: Con unidades (euros, porcentajes, medidas)
- **🐛 Problemas técnicos**: Errores y sus soluciones
- **🏗️ Contexto del dominio**: Terminología especializada

#### Ejemplo de Optimización
```
Conversación original (12,000 caracteres):
[Turno 1-15: conversación extensa sobre configuración]

Después del resumen automático (7,500 caracteres):
[RESUMEN: Usuario configuró API keys (Google, Jina), procesó 3 PDFs
sobre machine learning, prefiere explicaciones técnicas detalladas,
tuvo error con Qdrant que se resolvió reiniciando servicio]
[Turno 10-15: últimos 6 turnos completos preservados]
```

#### Configuración Personalizable
```yaml
# Ajustar límites según necesidades
MAX_HISTORY_CHARS: 12000               # Conversaciones más largas
SLIDING_WINDOW_TURNS: 10               # Más turnos recientes
```

**✨ Ventajas:**
- Conversaciones ilimitadamente largas sin pérdida de contexto
- Preservación inteligente de información crítica
- Optimización automática de prompts para mejor rendimiento
- Sin intervención manual requerida

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

## 📋 Flujo Completo de PDF a RAG Multimodal

### Proceso de Ingesta de PDFs (ingest.py)

#### 1. 📄 Extracción y Procesamiento de PDFs

**Extracción de texto:**
```python
# 1. Validación automática de PDFs
- Verificación de existencia, tamaño (<100MB), formato válido
- Carga con PyPDFLoader de LangChain
- Extracción de metadatos (título, autor, fecha de creación)

# 2. Procesamiento por páginas
- Asignación de metadatos mejorados: doc_id, title, page, source
- Validación de contenido mínimo (>10 caracteres)
- Filtrado de páginas vacías

# 3. Chunking optimizado de texto
- RecursiveCharacterTextSplitter
- CHUNK_SIZE: 1200 caracteres
- CHUNK_OVERLAP: 180 caracteres
- Estadísticas: promedio, mínimo, máximo chunk size
```

**Extracción multimodal de imágenes:**
```python
# 1. Extracción con PyMuPDF (fitz)
- extract_images_from_pdf() usa PyMuPDF para extraer imágenes embebidas
- Formatos soportados: PNG, JPEG, WebP
- Obtiene: dimensiones, bounding box, datos binarios, metadata

# 2. Preprocesamiento inteligente de imágenes
- _normalize_image_orientation(): Corrige orientación EXIF automáticamente
- _resize_image(): Redimensiona manteniendo aspect ratio (máximo 1024px lado mayor)
- _calculate_image_hash(): SHA-256 para deduplicación robusta

# 3. Filtrado avanzado de imágenes significativas
- Elimina imágenes < 50x50px (demasiado pequeñas)
- Filtra aspect ratios extremos (>10:1, normalmente logos/líneas)
- Descarta archivos < 500 bytes (probablemente corruptos)
- Excluye headers/footers (15% superior/inferior de página)
- Filtra baja complejidad visual (varianza de píxeles)
- Deduplicación automática por hash SHA-256
```

**Almacenamiento estructurado:**
```bash
# Estructura de directorios generada automáticamente:
/var/data/rag/images/
└── {doc_id}/
    └── p{page_number}/
        ├── {hash}.png          # Imagen original procesada
        └── thumbs/
            └── {hash}.jpg      # Thumbnail optimizado (256px max)

# Metadatos completos generados:
- image_path, thumbnail_path (rutas absolutas)
- image_uri, thumbnail_uri (rutas relativas para referencias)
- width, height, bbox (coordenadas en página)
- doc_id, page_number, image_index, hash SHA-256
```

#### 2. 🧮 Generación de Embeddings Multimodales

**Configuración unificada de embeddings:**
```python
# Para texto (retrieval.query - consultas de usuario)
task_config = {
    "task_type": "retrieval.query",
    "late_chunking": False,
    "normalized": True  # Para usar DOT distance en Qdrant
}

# Para documentos/imágenes (retrieval.passage - contenido indexado)
task_config = {
    "task_type": "retrieval.passage", 
    "late_chunking": True,  # Solo para chunks de texto grandes
    "normalized": True      # Consistente con queries
}

# Modelos configurados:
- Jina: jina-embeddings-v4 (1024 dimensiones, normalizado) - CONFIGURACIÓN ACTIVA
- Nota: El sistema está configurado únicamente para Jina embeddings
```

**Procesamiento por lotes optimizado:**
```python
# Embeddings de texto (paralelo por chunks)
texts = [doc.page_content for doc in batch]
embeddings = emb.embed_documents(texts)  # API call a Jina

# Embeddings de imágenes (paralelo por archivos)  
embeddings = emb.embed_images(batch_image_files)  # API call a Jina
# Procesa rutas de archivos PNG directamente
# Mismo espacio vectorial que el texto para búsqueda híbrida
```

#### 3. 🗄️ Almacenamiento en Base de Datos Vectorial (Qdrant)

**Configuración de colección multimodal:**
```python
# Configuración automática según proveedor de embeddings
MULTIMODAL_COLLECTION_CONFIG = {
    "collection_name": "rag_multimodal",
    "distance": Distance.DOT,      # Para Jina (normalizado)
    "vector_size": 1024,           # Jina embeddings-v4
}

# Índices de payload para búsqueda eficiente:
- modality: keyword index ("text" | "image")
- doc_id: keyword index (agrupación por documento)
- page_number: integer index (búsqueda por página)
- hash: keyword index (deduplicación rápida)
- title, author: text index (búsqueda por metadatos)
```

**Estructura de payload unificada:**
```python
@dataclass
class MultimodalPayload:
    # Campos requeridos comunes
    id: str                    # UUID único para cada vector
    modality: Modality         # "text" | "image"
    doc_id: str               # ID del documento padre
    page_number: int          # Página en el documento
    source_uri: str           # Archivo PDF original
    hash: str                 # SHA-256 para deduplicación
    embedding_model: str      # Modelo usado (jina-embeddings-v4)
    created_at: str           # Timestamp ISO de inserción
    
    # Campos específicos de texto
    page_content: str         # Contenido del chunk
    content_preview: str      # Primeros 200 caracteres
    
    # Campos específicos de imagen  
    thumbnail_uri: str        # Ruta relativa del PNG
    width: int, height: int   # Dimensiones en píxeles
    image_index: int          # Índice de imagen en la página
    bbox: Dict                # Coordenadas: {"x0": float, "y0": float, "x1": float, "y1": float}
    
    # Campos opcionales comunes
    title: str                # Título limpio del documento
    author: str               # Autor extraído del PDF
    creation_date: str        # Fecha de creación del PDF
```

**Inserción vectorial por lotes:**
```python
# Para texto: factory method optimizado
multimodal_payload = MultimodalPayload.from_text_chunk(
    page_content=text,
    doc_id=metadata['doc_id'],
    page_number=metadata['page'],
    embedding_model="jina-embeddings-v4"
)

# Para imagen: factory method con metadatos completos
multimodal_payload = MultimodalPayload.from_image_data(
    image_data=b"placeholder",  # No se almacenan datos binarios
    doc_id=doc_id,
    page_number=page_number, 
    thumbnail_uri=relative_path,
    width=800, height=600,
    embedding_model="jina-embeddings-v4"
)

# Inserción eficiente en lotes de 50 vectores
client.upsert(collection_name="rag_multimodal", points=[
    models.PointStruct(
        id=payload.id,
        vector=embedding,          # Vector 1024-dimensional
        payload=payload.to_dict()  # Metadatos completos
    )
])
```

### Proceso de Retrieval Multimodal (app.py)

#### 4. 🔍 Information Retrieval Híbrido

**Búsqueda vectorial unificada:**
```python
class CustomQdrantRetriever:
    def get_relevant_documents(self, query: str, k: int = 30):
        # 1. Generar embedding de la query del usuario
        query_embedding = self.embeddings.embed_query(query)  # Jina API
        
        # 2. Búsqueda vectorial híbrida en Qdrant
        search_result = self.client.query_points(
            collection_name="rag_multimodal",
            query=query_embedding,
            limit=k,                    # Búsqueda amplia inicial
            with_payload=True           # Incluir todos los metadatos
        )
        
        # 3. Procesar resultados por modalidad
        documents = []
        for result in search_result.points:
            payload = result.payload
            modality = payload.get('modality', 'text')
            
            if modality == 'image':
                # Crear descripción contextual para imágenes
                thumbnail_uri = payload.get('thumbnail_uri', '')
                image_index = payload.get('image_index', 0)
                width = payload.get('width', 0)
                height = payload.get('height', 0)
                
                page_content = f"Imagen {image_index + 1} en página {payload.get('page_number', 'N/A')} (dimensiones: {width}x{height}px, thumbnail: {thumbnail_uri})"
            
            # Preservar metadatos completos + similarity score
            metadata = payload.copy()
            metadata['similarity_score'] = result.score
            
        return documents
```

**Reranking inteligente con Jina:**
```python
# Configuración de reranking en dos etapas
RETRIEVAL_TOP_K = 30    # Búsqueda vectorial amplia
RERANK_TOP_K = 15       # Selección final precisa

if reranker and len(docs) > 1:
    # Aplicar reranking semántico avanzado
    reranked_docs, rerank_latency_ms = reranker.rerank_doc_objects(
        query=query,
        documents=docs,
        top_n=RERANK_TOP_K
    )
    docs = reranked_docs
    
    # Log de mejoras de relevancia
    for i, doc in enumerate(docs[:3], 1):
        rerank_score = doc.metadata.get('rerank_score', 'N/A')
        title = doc.metadata.get('title', 'Unknown')[:50]
        logger.debug(f"Reranked doc {i}: {title} (score: {rerank_score})")
```

#### 5. 🧠 Generación RAG Multimodal con Gemini

**Construcción de contexto híbrido:**
```python
# Contexto diferenciado por modalidad
context_parts = []
image_documents = []  # Tracking para procesamiento multimodal

for i, doc in enumerate(docs):
    doc_title = doc.metadata.get('title', 'Documento desconocido')
    page_num = doc.metadata.get('page_number', 'N/A')
    modality = doc.metadata.get('modality', 'text')
    
    if modality == 'image':
        # Contexto enriquecido para imágenes
        thumbnail_uri = doc.metadata.get('thumbnail_uri', '')
        image_index = doc.metadata.get('image_index', 0)
        width = doc.metadata.get('width', 0)
        height = doc.metadata.get('height', 0)
        
        context_part = f"""DOCUMENTO {i+1}: {doc_title}
PÁGINA: {page_num}
TIPO: IMAGEN {image_index + 1}
DIMENSIONES: {width}x{height}px
UBICACIÓN: {thumbnail_uri}
FUENTE: {doc.metadata.get('source_uri', '')}
CONTENIDO: {doc.page_content}
---"""
        
        # Guardar para procesamiento visual
        image_documents.append({
            'doc_index': i + 1,
            'thumbnail_uri': thumbnail_uri,  # Ruta del PNG original
            'title': doc_title,
            'page': page_num,
            'image_index': image_index
        })
    else:
        # Contexto estándar para texto
        context_part = f"""DOCUMENTO {i+1}: {doc_title}
PÁGINA: {page_num}
CONTENIDO:
{doc.page_content}
---"""
    
    context_parts.append(context_part)

context = "\n\n".join(context_parts)
```

**Procesamiento visual con Gemini 2.5 Flash Lite:**
```python
if image_documents:  # Hay imágenes relevantes
    logger.info(f"Preparando RAG multimodal con {len(image_documents)} imágenes")
    
    message_content = []
    
    # 1. Prompt textual con contexto híbrido
    text_prompt = f"""{sistema_prompt}

Contexto:
{context}

Historial:
{historial_formateado}

Pregunta: {query}

Por favor analiza tanto el texto como las imágenes proporcionadas. Si las imágenes contienen gráficos, tablas o diagramas relevantes, descríbelos y úsalos en tu respuesta.

Respuesta:"""
    
    message_content.append({
        "type": "text",
        "text": text_prompt
    })
    
    # 2. Cargar y agregar imágenes (limitado a 5 más relevantes)
    images_added = 0
    max_images = 5
    
    for img_doc in image_documents[:max_images]:
        # Cargar imagen desde almacenamiento
        image_path = img_doc['thumbnail_uri']  # PNG original (no thumbnail)
        image_base64 = load_image_as_base64(image_path)
        
        if image_base64:  # data:image/png;base64,iVBORw0KGgoAAAANSUhEU...
            message_content.append({
                "type": "image_url", 
                "image_url": image_base64
            })
            images_added += 1
            logger.info(f"Added image {images_added}: Doc {img_doc['doc_index']}, Page {img_doc['page']}")
    
    # 3. Invocar Gemini con contenido multimodal
    human_message = HumanMessage(content=message_content)
    resp = current_llm.invoke([human_message]).content.strip()
```

**Sistema de citas inline:**
```python
# 1. Extracción automática de documentos citados
cited_doc_numbers = extract_cited_documents(response)  
# Busca patrones: "DOCUMENTO 1", "DOCUMENTO 7"

# 2. Mapeo secuencial para legibilidad
citation_mapping = create_citation_mapping(cited_doc_numbers)  
# {1: 1, 7: 2} - doc original -> número secuencial

# 3. Reescritura con formato inline  
processed_response = rewrite_document_references(response, citation_mapping)
# "DOCUMENTO 1" → "[1]", "DOCUMENTO 7" → "[2]"

# 4. Generación de referencias completas
citations_text = generate_citations_for_cited_docs(docs, cited_doc_numbers, citation_mapping)

# Formato final multimodal:
"""
Respuesta con citas [1] y [2] basada en el análisis de texto e imágenes.

📚 Fuentes consultadas:
[1] Manual de Usuario (pág. 5)
[2] Imagen 2 en Configuración Avanzada (pág. 12) - 800x600px - thumbnail: config/p12/a1b2c3d4.png
"""
```

### Flujo Completo Integrado

```
📄 PDF Input → 🔧 Extract → 🖼️ Process → 🧮 Embed → 🗄️ Store → 🔍 Retrieve → 🧠 Generate

1. PDF Files (docs/)
   ↓
2. ingest.py
   ├─ PyPDFLoader (extracción de texto por páginas)
   ├─ PyMuPDF (extracción de imágenes embebidas)  
   ├─ Validación + Filtrado inteligente
   ├─ Chunking optimizado (1200/180 caracteres)
   └─ Almacenamiento estructurado (/var/data/rag/images/)
   ↓
3. embedding_factory.py  
   ├─ Jina API embeddings-v4 (texto + imágenes)
   ├─ Configuración: task_type retrieval.passage/query
   ├─ Normalización: True (DOT distance compatible)
   └─ Procesamiento por lotes optimizado
   ↓
4. Qdrant Vector Database
   ├─ Colección multimodal unificada (rag_multimodal)
   ├─ Payload estructurado: MultimodalPayload
   ├─ Índices optimizados: modality, doc_id, page_number  
   └─ Deduplicación SHA-256 automática
   ↓
5. User Query → app.py
   ↓
6. CustomQdrantRetriever
   ├─ embed_query(user_input) con Jina
   ├─ query_points(qdrant) búsqueda híbrida
   ├─ Jina Reranker (30→15 documentos)
   └─ Contexto multimodal estructurado
   ↓
7. Google Gemini 2.5 Flash Lite
   ├─ Prompt textual + Imágenes base64
   ├─ Sistema prompt especializado multimodal
   ├─ Procesamiento visual nativo
   └─ Generación de citas DOCUMENTO N
   ↓
8. Post-processing
   ├─ Extracción de documentos citados
   ├─ Mapeo secuencial de citas
   ├─ Reescritura inline [N]
   └─ Referencias completas multimodales
   ↓
9. Final Response
   ├─ Contenido híbrido (texto + visual)
   ├─ Citas inline [1], [2], [3]
   └─ Referencias: 📚 Fuentes consultadas con metadatos completos
```

Esta arquitectura permite un RAG verdaderamente multimodal donde:
- ✅ **PDFs se procesan completamente** (texto + todas las imágenes)
- ✅ **Búsquedas híbridas** encuentran contenido relevante independiente de modalidad
- ✅ **Respuestas enriquecidas** que combinan información textual y visual
- ✅ **Citas precisas** que referencian tanto texto como imágenes específicas
- ✅ **Escalabilidad** através de APIs (Jina + Google) sin procesamiento local pesado

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
