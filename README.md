# Minutador de Transcripción Multicanal

![Rust](https://img.shields.io/badge/Made_with-Rust-orange?style=flat-square)
![Whisper](https://img.shields.io/badge/Model-OpenAI_Whisper-blueviolet?style=flat-square)
![Diarización](https://img.shields.io/badge/Diarización-Wespeaker_ResNet293-informational?style=flat-square)
![GPU](https://img.shields.io/badge/GPU-CUDA_·_ROCm_·_Metal_·_DirectML-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Experimental-yellow?style=flat-square)

Aplicación de escritorio escrita en Rust para transcribir audio de forma completamente **local** usando el modelo **Whisper** de OpenAI. Captura múltiples fuentes de audio simultáneamente (micrófono y audio del sistema), identifica quién habla en cada momento mediante **diarización de voz**, y transcribe archivos de vídeo/audio con timestamps.

Todo el procesamiento ocurre en tu equipo: no se envía ningún dato a la nube.

---

## ⚠️ Estado del soporte multiplataforma

| Plataforma | Estado | Notas |
|---|---|---|
| 🐧 Linux | ✅ Probado | PulseAudio y PipeWire. GPU: CUDA, ROCm, Intel oneAPI |
| 🪟 Windows | 🧪 Experimental | Instalador `.exe` disponible. GPU: CUDA, DirectML |
| 🍎 macOS | 🧪 Experimental | GPU: Metal + CoreML (Apple Silicon). Requiere BlackHole para captura del sistema |

> Se agradecen PRs e issues para mejorar la estabilidad en Windows y macOS.

---

## ✨ Características

- **Transcripción local en tiempo real** — Sin nube, sin suscripción. Privacidad total.
- **Diarización de voz** — Identifica automáticamente quién habla en cada momento usando embeddings de voz ([Wespeaker ResNet293](https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet293-LM)). Los nombres de los hablantes son editables.
- **Multi-fuente** — Captura micrófonos y audio del sistema simultáneamente, con una etiqueta por fuente.
- **Transcripción de vídeo y audio** — Sube cualquier archivo `.mp4` o `.mp3` y obtén una transcripción con timestamps `[HH:MM:SS]` y, opcionalmente, con el hablante identificado.
- **Aceleración por GPU** — CUDA (NVIDIA), ROCm (AMD), Metal/CoreML (Apple Silicon), DirectML (AMD/Intel en Windows), con fallback automático a CPU.
- **Configuración de idioma** — Especifica el idioma original o activa la detección automática. Traducción al inglés integrada (función nativa de Whisper).
- **Descarga automática de modelos** — Los modelos Whisper se descargan de HuggingFace la primera vez que se usan.
- **Exportación a Markdown** — Las minutas se guardan automáticamente con fecha, hora y nombre de los hablantes.

---

## 🚀 Instalación

### 🪟 Windows — Instalador gráfico (recomendado)

1. Ve a la [página de Releases](../../releases) y descarga **`Install-Transcriptor.exe`**
2. Ejecútalo con doble clic
3. El asistente detectará tu GPU, descargará la variante correcta y creará un acceso directo

> Si Windows bloquea la ejecución del `.exe`, haz clic en *Más información → Ejecutar de todos modos* (el archivo no tiene firma de código).

**Alternativa en consola:** descarga `Install.bat` del mismo release y ejecútalo con doble clic. Lanza el instalador en PowerShell sin problemas de política de ejecución.

---

### 🐧 Linux / 🍎 macOS — Script de instalación

```bash
curl -fsSL https://github.com/Victor-agullo/Minutero/releases/latest/download/install.sh | bash
```

El script detecta tu GPU y arquitectura, descarga la variante correcta y configura el PATH.

---

### Binarios directos

Si prefieres instalar manualmente, descarga el `.zip` (Windows) o `.tar.gz` (Linux/macOS) que corresponda a tu GPU desde la [página de Releases](../../releases):

| Archivo | Plataforma | GPU |
|---|---|---|
| `transcriptor-windows-x86_64-cuda.zip` | Windows | NVIDIA (requiere CUDA Runtime 12.x) |
| `transcriptor-windows-x86_64-directml.zip` | Windows | AMD / Intel (Windows 10/11) |
| `transcriptor-windows-x86_64-cpu.zip` | Windows | Cualquiera |
| `transcriptor-linux-x86_64-cuda.tar.gz` | Linux | NVIDIA |
| `transcriptor-linux-x86_64-rocm.tar.gz` | Linux | AMD Radeon |
| `transcriptor-linux-x86_64-intel.tar.gz` | Linux | Intel Arc/Xe |
| `transcriptor-linux-x86_64-cpu.tar.gz` | Linux | Cualquiera |
| `transcriptor-macos-apple-silicon-metal.tar.gz` | macOS | Apple Silicon (M1/M2/M3/M4) |
| `transcriptor-macos-intel-cpu.tar.gz` | macOS | Intel |

---

### Compilar desde el código fuente

```bash
git clone https://github.com/Victor-agullo/Minutero
cd Minutero

# CPU (todas las plataformas, sin usar la GPU)
cargo build --release

# GPU NVIDIA — Linux o Windows
cargo build --release --features cuda

# GPU AMD — Linux (requiere ROCm instalado)
cargo build --release --features rocm

# GPU Intel — Linux (requiere Intel oneAPI Base Toolkit)
cargo build --release --features intel

# Apple Silicon — macOS (Metal + Neural Engine via CoreML)
cargo build --release --features metal
```

**Dependencias de compilación:**

| Plataforma | Paquetes necesarios |
|---|---|
| Linux | `build-essential libasound2-dev libclang-dev pkg-config cmake ffmpeg` |
| Windows | [LLVM](https://releases.llvm.org/), [Rust](https://rustup.rs/), ffmpeg en PATH |
| macOS | `brew install llvm ffmpeg` |

---

## 🛠️ Dependencias en tiempo de ejecución

### 🐧 Linux
```bash
sudo apt install ffmpeg pulseaudio-utils   # Debian/Ubuntu
sudo dnf install ffmpeg pulseaudio-utils   # Fedora
```
`pactl` gestiona los dispositivos de audio internamente. `ffmpeg` solo es necesario para la transcripción de archivos.

### 🪟 Windows
- **ffmpeg**: el instalador `.exe` lo instala con `winget` automáticamente, o manualmente desde [ffmpeg.org](https://ffmpeg.org/download.html)
- **Captura del sistema**: habilita **Mezcla estéreo (Stereo Mix)** en el Panel de Sonido, o instala [VB-Audio Cable](https://vb-audio.com/Cable/) si tu tarjeta no lo incluye

### 🍎 macOS
```bash
brew install ffmpeg
```
Para captura del audio del sistema, instala [BlackHole](https://github.com/ExistentialAudio/BlackHole) y configura un *Dispositivo Agregado* en *Audio MIDI Setup*.

---

## 🗣️ Modelo de diarización (identificación de hablantes)

La diarización es opcional. Para activarla necesitas el modelo ONNX de Wespeaker:

1. Ve a [huggingface.co/Wespeaker/wespeaker-voxceleb-resnet293-LM](https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet293-LM/tree/main)
2. Descarga **`voxceleb_resnet293_LM.onnx`** (~300 MB)
3. Renómbralo a **`speaker_embedding.onnx`** y colócalo en la carpeta `models/` del directorio de instalación

```
# Estructura esperada
~/.local/bin/                    # Linux
%LOCALAPPDATA%\Transcriptor\     # Windows
  transcriptor(.exe)
  models/
    speaker_embedding.onnx       ← aquí
```

> El modelo ResNet293 está entrenado en VoxCeleb (inglés y europeo). Para vídeos largos (> 8 minutos) la diarización puede tardar 1-2 minutos antes de que empiece la transcripción, ya que procesa el audio completo primero.

**Alternativa ligera** (~45 MB, algo menos precisa): descarga `voxceleb_resnet34_LM.onnx` del repo [wespeaker-voxceleb-resnet34-LM](https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM/tree/main).

---

## 📖 Guía de uso

### Transcripción en tiempo real

1. Ve a **⚙️ Configuración**
2. Añade fuentes con **➕ Entrada** (micrófono) o **➕ Salida** (audio del sistema/loopback)
3. Configura idioma y, si tienes el modelo, activa la **diarización**
4. Asigna nombres a los interlocutores (puedes cambiarlos después)
5. Vuelve a **🎙 Transcripción**, elige modelo y pulsa **▶ Iniciar Captura**
6. La minuta se guarda en `minutas/` al detener la captura

### Transcripción de vídeo y audio

1. Ve a **🎬 Vídeo**
2. Selecciona un archivo y, opcionalmente, activa la diarización y elige cuántos hablantes esperas
3. Pulsa **▶ Transcribir**
4. El resultado aparece con timestamps `[HH:MM:SS]` y se guarda automáticamente

> La primera ejecución descarga el modelo Whisper seleccionado (~1.5 GB para `large-v3`). Se almacena en caché para usos posteriores.

---

## 📂 Estructura del proyecto

| Archivo | Descripción |
|---|---|
| `main.rs` | Punto de entrada y configuración de la ventana |
| `ui.rs` | Interfaz gráfica (`egui`), estado de la aplicación y navegación |
| `audio.rs` | Captura en tiempo real, inferencia Whisper, gestión de hilos |
| `video.rs` | Extracción de audio con ffmpeg y transcripción por chunks con timestamps |
| `diarize.rs` | Motor de diarización: extracción de embeddings de voz (ONNX), clustering aglomerativo y asignación de hablantes |
| `system_audio.rs` | Detección de dispositivos loopback/monitor por plataforma |
| `data.rs` | Estructuras de datos compartidas (perfiles, mensajes, enums) |
| `installer.iss` | Script de Inno Setup para compilar el instalador `.exe` de Windows |
| `install.sh` | Instalador en Bash para Linux y macOS |
| `Install.bat` | Lanzador de consola para Windows (sin restricciones de política PS) |

---

## 🤝 Contribuciones

Son especialmente bienvenidas las contribuciones en estas áreas:

- Mejoras de precisión en la diarización (umbral, preprocesado de audio)
- Compatibilidad con Windows y macOS (captura, instalación, rutas)
- Soporte para modelos Whisper adicionales o cuantizados
- Detección de actividad de voz (VAD) más robusta

```bash
git fork
git checkout -b feature/mi-mejora
git commit -m 'Descripción clara del cambio'
git push origin feature/mi-mejora
# → Abre un Pull Request
```

---

## 📄 Licencia

MIT — consulta el archivo `LICENSE` para más detalles.

> Este software usa `whisper.cpp` a través de los bindings `whisper-rs` y el modelo de embeddings de voz de [Wespeaker](https://github.com/wenet-e2e/wespeaker). Los modelos se descargan de HuggingFace y están sujetos a sus propias licencias.