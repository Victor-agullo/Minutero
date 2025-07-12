# Transcriptor de Audio con Whisper

Una aplicación GUI para transcribir audio en tiempo real usando el modelo Whisper de OpenAI.

## Características

- 🎤 **Captura de micrófono**: Transcribe audio del micrófono en tiempo real
- 🖥️ **Captura de escritorio**: Captura audio del sistema (música, videos, etc.)
- 📁 **Archivos de audio**: Procesa archivos de audio pregrabados
- 🏷️ **Nombres personalizables**: Asigna nombres custom a cada fuente de audio
- 📝 **Exportación Markdown**: Guarda transcripciones con timestamps
- 🎯 **Detección de actividad vocal**: Solo transcribe cuando detecta voz
- 🌍 **Múltiples idiomas**: Soporta español, inglés y muchos otros idiomas

## Requisitos

- Rust 1.70+
- CUDA (opcional, para aceleración GPU)
- Metal (opcional, para aceleración en macOS)

## Instalación

### 1. Clonar el repositorio

```bash
git clone <tu-repositorio>
cd whisper-transcriber
```

### 2. Descargar modelo Whisper

Descarga uno de estos modelos y colócalo en el directorio del proyecto:

**Modelos GGML (recomendado para CPU):**
- `ggml-base.bin` (140MB) - Buena calidad, velocidad moderada
- `ggml-small.bin` (460MB) - Mejor calidad, más lento
- `ggml-medium.bin` (1.4GB) - Excelente calidad, lento

**Descargar desde:**
```bash
# Modelo base (recomendado)
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin

# O usar curl
curl -L -o ggml-base.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin
```

**Modelos SafeTensors (para GPU):**
- Descarga desde Hugging Face: `openai/whisper-base`

### 3. Compilar la aplicación

```bash
# Compilación básica (CPU)
cargo build --release

# Con soporte CUDA (GPU NVIDIA)
cargo build --release --features cuda

# Con soporte Metal (macOS)
cargo build --release --features metal
```

### 4. Ejecutar

```bash
./target/release/whisper-transcriber
```

## Uso

### 1. Configurar fuentes de audio

- **Micrófono**: Marca la casilla y selecciona tu micrófono
- **Escritorio**: Marca la casilla para capturar audio del sistema
- **Archivo**: Selecciona un archivo de audio para transcribir

### 2. Personalizar nombres

Edita los nombres que aparecerán en la transcripción para cada fuente.

### 3. Iniciar transcripción

- Presiona "🎯 Empezar a escuchar"
- El modelo se cargará automáticamente
- La transcripción comenzará en tiempo real

### 4. Ver resultados

- Las transcripciones se guardan automáticamente en `transcripcion.md`
- Cada entrada incluye timestamp y fuente
- Puedes cambiar el archivo de salida en la interfaz

## Formatos de archivo soportados

- **Audio**: WAV, MP3, M4A, FLAC, OGG
- **Salida**: Markdown (.md)

## Configuración avanzada

### Filtros de audio

El código incluye filtros opcionales:

- **Filtro paso alto**: Elimina ruido de baja frecuencia
- **Detección de actividad vocal**: Solo transcribe cuando detecta voz
- **Resampleo**: Convierte automáticamente a 16kHz (requerido por Whisper)

### Idiomas soportados

- Español (es)
- Inglés (en)
- Francés (fr)
- Alemán (de)
- Italiano (it)
- Portugués (pt)
- Ruso (ru)
- Japonés (ja)
- Coreano (ko)
- Chino (zh)

## Solución de problemas

### El modelo no se carga

- Verifica que el archivo `ggml-base.bin` esté en el directorio del proyecto
- Asegúrate de que el archivo no esté corrupto
- Prueba con un modelo más pequeño si tienes poca RAM

### Sin audio del micrófono

- Verifica permisos de micrófono en tu sistema
- Prueba con diferentes micrófonos de la lista
- En Linux, asegúrate de que ALSA/PulseAudio estén configurados

### Captura de escritorio no funciona

- En Windows: Requiere permisos de administrador
- En macOS: Habilita permisos de grabación de pantalla
- En Linux: Configura PulseAudio monitor

### Rendimiento lento

- Usa modelos más pequeños (`ggml-tiny.bin`)
- Habilita aceleración GPU si tienes CUDA/Metal
- Reduce la frecuencia de procesamiento

## Desarrollo

### Estructura del proyecto

```
src/
├── main.rs              # Aplicación principal y GUI
├── whisper_engine.rs    # Integración con Whisper
├── audio_capture.rs     # Captura de audio
└── lib.rs              # Módulos y exports
```

### Dependencias principales

- `eframe/egui`: Interfaz gráfica
- `candle-whisper`: Modelo Whisper
- `cpal`: Captura de audio
- `symphonia`: Decodificación de archivos de audio

### Contribuir

1. Fork el repositorio
2. Crea una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Crea un Pull Request

## Licencia

MIT License - ve el archivo LICENSE para detalles.

## Créditos

- OpenAI por el modelo Whisper
- Candle por la implementación en Rust
- Comunidad de egui por la GUI