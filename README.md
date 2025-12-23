# Minutador de Transcripción Multicanal (Rust + Whisper)

![Rust](https://img.shields.io/badge/Made_with-Rust-orange?style=flat-square)
![Whisper](https://img.shields.io/badge/Model-OpenAI_Whisper-blueviolet?style=flat-square)
![Status](https://img.shields.io/badge/Status-Experimental-yellow?style=flat-square)

Una aplicación de escritorio escrita en Rust para transcribir audio en tiempo real utilizando el modelo **Whisper** de OpenAI de forma local. Diseñada para generar minutas de reuniones, permite capturar simultáneamente micrófonos y audio del sistema (loopback), identificando a diferentes interlocutores.

## ⚠️ Estado del Soporte Multiplataforma

**Por favor, lee esto antes de usar:**

Este proyecto ha sido desarrollado y probado principalmente en **Linux** (bajo entornos PulseAudio y PipeWire).

* **🐧 Linux:** Soporte completo. Requiere herramientas estándar de audio (`pactl`, `parecord`).
* **🪟 Windows:** El código incluye lógica para detectar "Mezcla Estéreo" (Stereo Mix) vía PowerShell y usar WASAPI vía `cpal`, pero **no ha sido probado exhaustivamente**.
* **🍎 macOS:** Se incluye lógica básica, pero **no ha sido probado**. La captura de audio del sistema en macOS requiere software de terceros (como BlackHole) debido a limitaciones del sistema operativo.

> Se agradecen PRs (Pull Requests) y reportes de errores ("Issues") para mejorar la estabilidad en Windows y macOS.

## ✨ Características

* **Transcripción Local:** Ejecuta modelos Whisper (`ggml`) localmente. Privacidad total, sin enviar audio a la nube.
* **Multicanal / Multi-Interlocutor:**
    * Captura tu micrófono (Entrada).
    * Captura lo que escuchas en la reunión (Salida/Loopback).
    * Asigna nombres a cada fuente para generar un guion tipo chat.
* **Detección de Silencio (VAD):** Filtra los silencios para evitar alucinaciones del modelo y procesar solo cuando se habla.
* **Interfaz Gráfica (GUI):** Construida con `egui` para una experiencia ligera y rápida.
* **Gestión Automática de Modelos:** Descarga automáticamente los modelos necesarios (`base`, `medium`, `large-v3`) desde HuggingFace.
* **Exportación:** Guarda las transcripciones automáticamente en formato Markdown con fecha y hora.

## 🛠️ Prerrequisitos

Necesitas tener instalado [Rust y Cargo](https://rustup.rs/).

### Dependencias del Sistema (Linux)

En sistemas basados en Debian/Ubuntu, necesitarás las librerías de desarrollo de ALSA y utilidades de audio:

```bash
sudo apt update
sudo apt install build-essential libasound2-dev pkg-config pulseaudio-utils

```

*Nota: La aplicación utiliza `pactl` y `parecord` internamente para gestionar dispositivos en Linux de manera robusta.*

## 🚀 Instalación y Ejecución

1. **Clonar el repositorio:**
```bash
git clone [https://github.com/Victor-agullo/Minutero](https://github.com/Victor-agullo/Minutero)
cd minutador-whisper-rust

```


2. **Compilar y Ejecutar:**
```bash
cargo run --release

```


*Se recomienda usar `--release` para que la inferencia del modelo Whisper sea rápida y en tiempo real.*

## 📖 Guía de Uso

1. **Inicio:** Al abrir la app, verás la pestaña de "Transcripción".
2. **Configuración:** Ve a la pestaña **⚙️ Configuración**.
* **Añadir Fuente:** Pulsa "➕ Entrada" para micrófonos o "➕ Salida" para el audio del sistema.
* **Loopback (Audio del sistema):** Si estás en Linux, detectará los monitores automáticamente. En Windows, asegúrate de tener habilitada la "Mezcla Estéreo".
* **Activar:** Marca la casilla (checkbox) de los perfiles que quieras grabar.


3. **Modelo:** En la pantalla principal, selecciona el modelo (ej. `medium` o `large-v3`). La primera vez que inicies la captura, el programa descargará el modelo (puede tardar unos minutos dependiendo de tu conexión).
4. **Transcribir:** Pulsa **▶ Iniciar Captura**.
5. **Resultados:** El texto aparecerá en tiempo real. Al finalizar, la minuta se guardará en la carpeta `minutas/`.

## 📂 Estructura del Proyecto

* `main.rs`: Punto de entrada y configuración de la ventana.
* `ui.rs`: Lógica de la interfaz gráfica (`egui`), gestión de estado y renderizado.
* `audio.rs`: Núcleo de la captura de audio y procesamiento con Whisper. Gestiona hilos y conversión de audio.
* `system_audio.rs`: Utilidades para detectar capacidades de loopback/monitor según el sistema operativo.
* `data.rs`: Estructuras de datos compartidas (perfiles, mensajes, enums).

## 🤝 Contribuciones

Las contribuciones son bienvenidas, especialmente para mejorar la capa de abstracción de audio (`cpal`) en Windows y macOS para reducir la dependencia de comandos externos de Linux.

1. Haz un Fork del proyecto.
2. Crea tu rama de funcionalidad (`git checkout -b feature/AmazingFeature`).
3. Haz Commit de tus cambios (`git commit -m 'Add some AmazingFeature'`).
4. Push a la rama (`git push origin feature/AmazingFeature`).
5. Abre un Pull Request.

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Consulta el archivo `LICENSE` para más detalles.

**Nota:** Este software utiliza `whisper.cpp` a través de los bindings `whisper-rs`. Los modelos se descargan de HuggingFace y están sujetos a sus propias licencias de uso.