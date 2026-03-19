# Minutador de Transcripción Multicanal

![Rust](https://img.shields.io/badge/Made_with-Rust-orange?style=flat-square)
![Whisper](https://img.shields.io/badge/Model-OpenAI_Whisper-blueviolet?style=flat-square)
![Status](https://img.shields.io/badge/Status-Experimental-yellow?style=flat-square)

Aplicación de escritorio escrita en Rust para transcribir audio en tiempo real usando el modelo **Whisper** de OpenAI de forma completamente local. Permite capturar múltiples fuentes de audio simultáneamente (micrófono y audio del sistema), identificar interlocutores, y transcribir archivos de vídeo/audio con timestamps.

---

## ⚠️ Estado del soporte multiplataforma

| Plataforma | Estado | Notas |
|---|---|---|
| 🐧 Linux | ✅ Completo | Probado con PulseAudio y PipeWire |
| 🪟 Windows | 🧪 Experimental | Usa WASAPI vía `cpal`. No probado exhaustivamente |
| 🍎 macOS | 🧪 Experimental | Requiere BlackHole para captura del sistema |

> Se agradecen PRs e Issues para mejorar la estabilidad en Windows y macOS.

---

## ✨ Características

- **Transcripción local en tiempo real:** Sin enviar audio a la nube. Privacidad total.
- **Multi-interlocutor:** Captura micrófonos y audio del sistema simultáneamente, asignando un nombre a cada fuente.
- **Transcripción de vídeo/audio:** Sube un archivo y obtén una transcripción completa con timestamps (`[MM:SS]`).
- **Configuración de idioma:** Especifica el idioma original y, opcionalmente, traduce al inglés (única traducción nativa de Whisper).
- **Detección de silencio:** Filtra silencios para evitar alucinaciones del modelo.
- **Gestión automática de modelos:** Descarga `medium` o `large-v3` desde HuggingFace la primera vez.
- **Exportación a Markdown:** Guarda minutas automáticamente con fecha y hora.
- **GUI ligera:** Construida con `egui`/`eframe`.

---

## 🛠️ Prerrequisitos

> **Rust no es necesario** si usas un binario precompilado. Solo se necesita para compilar desde el código fuente.

### Dependencias de ejecución (todos los usuarios)

#### 🐧 Linux
```bash
sudo apt install pulseaudio-utils ffmpeg
```
> `pactl` y `parecord` gestionan los dispositivos de audio internamente. `ffmpeg` es necesario solo para la transcripción de vídeo.

#### 🪟 Windows
- [ffmpeg](https://ffmpeg.org/download.html) añadido al PATH — solo para transcripción de vídeo. Si no lo tienes, la pestaña de vídeo mostrará un error pero el resto funciona.
- Para captura del sistema: habilitar **Mezcla estéreo (Stereo Mix)** en el Panel de Sonido, o instalar [VB-Audio Cable](https://vb-audio.com/Cable/) si tu tarjeta no lo soporta.

#### 🍎 macOS
```bash
brew install ffmpeg
```
> Para captura del sistema, instala [BlackHole](https://github.com/ExistentialAudio/BlackHole) y configura un Dispositivo Agregado en *Audio MIDI Setup*.

### Dependencias de compilación (solo si compilas desde el código fuente)

- [Rust y Cargo](https://rustup.rs/)
- Linux: `sudo apt install build-essential libasound2-dev pkg-config`
- Windows: [LLVM](https://releases.llvm.org/) (necesario para `whisper-rs`)

---

## 🚀 Instalación

### Binarios precompilados
Descarga el binario para tu plataforma desde la [página de Releases](../../releases).

### Compilar desde el código fuente
```bash
git clone https://github.com/Victor-agullo/Minutero
cd Minutero

# CPU (todas las plataformas)
cargo build --release

# Con aceleración CUDA (Linux/Windows con GPU NVIDIA)
cargo build --release --features cuda
```

---

## 📖 Guía de uso

### Transcripción en tiempo real

1. Ve a **⚙️ Configuración**
2. Añade fuentes de audio con **➕ Entrada** (micrófono) o **➕ Salida** (audio del sistema/loopback)
3. Configura el **Idioma original** y el **Idioma destino** (opcional)
4. Asigna nombres a los interlocutores
5. Vuelve a **🎙 Transcripción**, elige el modelo y pulsa **▶ Iniciar Captura**
6. La minuta se guarda automáticamente en `minutas/` al detener la captura

### Transcripción de vídeo/audio

1. Ve a **🎬 Vídeo**
2. Selecciona un archivo (`.mp4`, `.mkv`, `.mp3`, `.wav`, `.flac`, etc.)
3. Pulsa **▶ Transcribir**
4. El resultado aparece con timestamps `[MM:SS]` y se guarda automáticamente

> La primera ejecución descarga el modelo seleccionado (~1.5 GB para `large-v3`).

---

## 📂 Estructura del proyecto

| Archivo | Descripción |
|---|---|
| `main.rs` | Punto de entrada y configuración de la ventana |
| `ui.rs` | Interfaz gráfica (`egui`), estado y lógica de navegación |
| `audio.rs` | Captura en tiempo real, procesamiento con Whisper, gestión de hilos |
| `video.rs` | Extracción de audio con ffmpeg y transcripción por chunks con timestamps |
| `system_audio.rs` | Detección de dispositivos loopback/monitor por plataforma |
| `data.rs` | Estructuras de datos compartidas (perfiles, mensajes, enums) |

---

## 🤝 Contribuciones

Se agradecen especialmente contribuciones para:
- Mejorar la compatibilidad con Windows y macOS
- Mejoras en la detección de silencios (VAD)
- Soporte para más idiomas o modelos

1. Fork → `git checkout -b feature/mi-mejora`
2. `git commit -m 'Descripción del cambio'`
3. `git push origin feature/mi-mejora`
4. Abre un Pull Request

---

## 📄 Licencia

MIT — consulta el archivo `LICENSE` para más detalles.

> Este software usa `whisper.cpp` a través de los bindings `whisper-rs`. Los modelos se descargan de HuggingFace y están sujetos a sus propias licencias.