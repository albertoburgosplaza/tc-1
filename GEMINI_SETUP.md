# 🤖 Guía de Configuración Gemini 2.5 Flash Lite

Esta guía explica cómo configurar y usar Gemini 2.5 Flash Lite como alternativa a Llama 3.2 1B en tu chatbot RAG.

## 🚀 Configuración Rápida

### 1. Obtener API Key de Google AI

1. Visita [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Inicia sesión con tu cuenta de Google
3. Crea un nuevo proyecto (si no tienes uno)
4. Genera una nueva API Key
5. Copia la API Key (formato: `AIza...`)

### 2. Configurar Variables de Entorno

**Opción A: Archivo .env (Recomendado)**
```bash
# Crear archivo .env en la raíz del proyecto
echo "GOOGLE_API_KEY=tu_api_key_aqui" >> .env

# Ejemplo
echo "GOOGLE_API_KEY=AIzaSyC123...xyz789" >> .env
```

**Opción B: Variable de sistema**
```bash
# Linux/Mac
export GOOGLE_API_KEY="tu_api_key_aqui"

# Windows
set GOOGLE_API_KEY=tu_api_key_aqui
```

### 3. Reiniciar el Sistema

```bash
# Reiniciar solo la aplicación
docker compose restart app

# O reiniciar todo el stack
docker compose down
docker compose up -d
```

## 🎯 Usar Gemini en la Interfaz Web

1. **Accede a la interfaz:** http://localhost:7860
2. **Localiza el selector de proveedor** en la parte superior
3. **Selecciona "google"** del radio button "Proveedor LLM"
4. **Verifica el cambio** en el campo "Estado del modelo"
   - ✅ Éxito: "Modelo actual: gemini-2.5-flash-lite (google)"
   - ❌ Error: "Error al cambiar a google. Usando: llama3.2:1b (ollama)"

## ⚡ Ventajas de Gemini 2.5 Flash Lite

### Rendimiento
- **Ultra velocidad**: Respuestas en <1 segundo
- **Sin descarga**: No requiere descargar GB de modelos
- **Baja latencia**: Infraestructura optimizada de Google

### Capacidades
- **Multimodal**: Soporta texto e imágenes (futuras funcionalidades)
- **Contexto largo**: Hasta 1M tokens de contexto
- **Actualizado**: Modelo más reciente de Google AI

### Recursos
- **Sin consumo local**: Libera RAM y CPU del servidor
- **Escalabilidad**: No limitado por hardware local
- **Disponibilidad**: 99.9% uptime garantizado por Google

## 🔄 Cambio Dinámico Entre Modelos

### Desde la Interfaz Web
- **Cambiar a Gemini**: Selecciona "google" → Cambio automático
- **Volver a Ollama**: Selecciona "ollama" → Vuelve a llama3.2:1b

### Configuración por Defecto
```yaml
# En docker-compose.yml
environment:
  LLM_PROVIDER: ollama          # Cambiar a "google" para Gemini por defecto
  LLM_MODEL: llama3.2:1b       # Solo aplica cuando provider=ollama
  GOOGLE_API_KEY: "${GOOGLE_API_KEY:-}"  # Toma del .env
```

## 🛠️ Troubleshooting

### Error: "Google API key not found"
```bash
# Verificar que la variable existe
echo $GOOGLE_API_KEY

# Verificar el archivo .env
cat .env | grep GOOGLE_API_KEY

# Reiniciar después de configurar
docker compose restart app
```

### Error: "Failed to switch to google"
```bash
# Verificar logs
docker compose logs app | grep -i google

# Posibles causas:
# 1. API Key inválida o expirada
# 2. Límites de cuota excedidos
# 3. Problemas de conectividad

# Solución: Verificar API Key en Google AI Studio
```

### Error: "No response from model"
```bash
# Verificar conectividad
curl -H "x-goog-api-key: tu_api_key" \
     -H "Content-Type: application/json" \
     -X POST \
     -d '{"contents":[{"parts":[{"text":"Hello"}]}]}' \
     "https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash-lite:generateContent"

# Si falla, verificar:
# 1. API Key correcta
# 2. Modelo disponible en tu región
# 3. Cuotas de la API
```

## 💰 Consideraciones de Costo

### Google AI Studio - Nivel Gratuito
- **15 RPM (requests por minuto)**
- **1M tokens por día** 
- **Gratis hasta límites**

### Uso Estimado
```
Consulta típica RAG: ~500 tokens
Respuesta típica: ~200 tokens
Total por consulta: ~700 tokens

Con límite gratuito:
1,000,000 ÷ 700 = ~1,428 consultas/día
```

### Optimización de Costos
- **Usa contexto mínimo**: Solo documentos relevantes
- **Historial limitado**: Configuración automática (6 turnos)
- **Respuestas concisas**: Temperature=0.05 configurado

## 🔒 Seguridad

### Protección de API Keys
```bash
# Nunca versionar archivos con keys
echo ".env" >> .gitignore

# Usar variables de entorno en producción
# No hardcodear keys en docker-compose.yml
```

### Mejores Prácticas
- **Rotar API Keys** regularmente
- **Monitorear uso** en Google AI Studio
- **Configurar alertas** de cuota
- **Usar límites** por aplicación

## 📊 Comparativa: Gemini vs Llama 3.2 1B

| Aspecto | Gemini 2.5 Flash Lite | Llama 3.2 1B |
|---------|------------------------|---------------|
| **Velocidad** | <1s (nube) | ~2s (local) |
| **Instalación** | Solo API key | ~1.3GB descarga |
| **RAM Local** | 0MB | ~2GB |
| **Calidad** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Costo** | Gratis/Pago API | Gratis total |
| **Privacidad** | Nube Google | 100% local |
| **Disponibilidad** | Internet requerido | Funciona offline |

## ✅ Verificación de Configuración

```bash
# Script de verificación
curl -s http://localhost:8080/status | jq '.services'

# Verificar modelo activo desde logs
docker compose logs app | tail -20 | grep -i "model\|provider"

# Test básico con ambos modelos
# 1. Selecciona ollama → Haz una pregunta
# 2. Selecciona google → Haz la misma pregunta
# 3. Compara velocidad y calidad
```

---

**¿Problemas?** Consulta los logs: `docker compose logs app | grep -E "(ERROR|WARN|google|gemini)"`