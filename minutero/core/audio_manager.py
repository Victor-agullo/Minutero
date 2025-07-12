# core/audio_manager.py

import asyncio
import numpy as np
import sounddevice as sd
import wave
import os
from typing import AsyncGenerator, Literal, Dict, Any, Optional
from utils.logger import logger
from config.settings import settings

class AudioManager:
    """
    Gestiona la captura de audio de diferentes fuentes (micrófono, archivo, simulación de pantalla).
    Devuelve streams de audio como generadores asíncronos de bytes PCM.
    """

    def __init__(self):
        self.active_streams: Dict[str, asyncio.Task] = {}
        logger.info("AudioManager inicializado.")

    async def _microphone_stream_generator(self, source_tag: str) -> AsyncGenerator[bytes, None]:
        """Generador asíncrono para capturar audio del micrófono."""
        try:
            # Obtener el índice del dispositivo por defecto.
            # Puedes listar dispositivos con sd.query_devices()
            device_info = sd.query_devices(kind='input')
            device_id = device_info['index']
            logger.info(f"Iniciando captura de micrófono. Dispositivo: {device_info['name']} (ID: {device_id})")

            # sd.InputStream es un contexto asíncrono
            loop = asyncio.get_running_loop()
            event = asyncio.Event() # Para señalizar nuevos datos
            q = asyncio.Queue()     # Para pasar los chunks de audio

            def callback(indata, frames, time, status):
                if status:
                    logger.warning(f"Status del stream de micrófono: {status}")
                # El callback se ejecuta en un hilo separado de sounddevice.
                # Para evitar bloquearlo, ponemos los datos en una cola y señalizamos el evento.
                loop.call_soon_threadsafe(q.put_nowait, bytes(indata))
                loop.call_soon_threadsafe(event.set)

            with sd.InputStream(
                samplerate=settings.AUDIO_SAMPLE_RATE,
                channels=1, # Mono
                dtype='int16', # PCM de 16 bits
                callback=callback,
                blocksize=settings.AUDIO_CHUNK_SIZE,
                device=device_id
            ) as stream:
                while True:
                    # Esperar hasta que haya nuevos datos
                    await event.wait()
                    event.clear() # Limpiar el evento para la próxima espera

                    while not q.empty():
                        chunk = q.get_nowait()
                        yield chunk # Rendir los bytes del chunk de audio
                    await asyncio.sleep(0.001) # Pequeña pausa para no saturar el loop de eventos

        except sd.PortAudioError as e:
            logger.error(f"Error de PortAudio al iniciar el micrófono: {e}. Asegúrate de que el micrófono esté conectado y los drivers instalados.", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Error inesperado en el stream del micrófono: {e}", exc_info=True)
            raise
        finally:
            logger.info(f"Stream de micrófono para {source_tag} finalizado.")

    async def _file_stream_generator(self, file_path: str, source_tag: str) -> AsyncGenerator[bytes, None]:
        """Generador asíncrono para leer audio de un archivo WAV."""
        if not os.path.exists(file_path):
            logger.error(f"Archivo de audio no encontrado: {file_path}")
            raise FileNotFoundError(f"Archivo de audio no encontrado: {file_path}")

        try:
            with wave.open(file_path, 'rb') as wf:
                if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != settings.AUDIO_SAMPLE_RATE:
                    logger.error(
                        f"El archivo '{file_path}' no cumple con el formato requerido: "
                        f"Mono, 16-bit PCM, {settings.AUDIO_SAMPLE_RATE} Hz. "
                        f"Actual: Canales={wf.getnchannels()}, Bits={wf.getsampwidth()*8}, SR={wf.getframerate()}"
                    )
                    raise ValueError("Formato de archivo WAV no soportado.")

                logger.info(f"Iniciando lectura de archivo: {file_path}")
                while True:
                    # Leer un chunk de frames
                    frames = wf.readframes(settings.AUDIO_CHUNK_SIZE)
                    if not frames:
                        break # Fin del archivo
                    yield frames
                    await asyncio.sleep(settings.AUDIO_CHUNK_SIZE / settings.AUDIO_SAMPLE_RATE) # Simular tiempo real

        except Exception as e:
            logger.error(f"Error al leer el archivo de audio '{file_path}': {e}", exc_info=True)
            raise
        finally:
            logger.info(f"Stream de archivo para {source_tag} finalizado.")

    async def _screen_capture_stream_generator(self, source_tag: str) -> AsyncGenerator[bytes, None]:
        """
        Generador asíncrono para simular la captura de audio de la pantalla.
        NOTA: La captura de audio de la pantalla es compleja y específica del SO.
        Aquí se simula con un stream vacío o se podría integrar con librerías como
        `mss` para captura de pantalla (solo imagen) y luego procesar el audio del sistema
        con herramientas externas o librerías específicas como `soundcard` (experimental).
        Para este ejemplo, simplemente generará chunks vacíos o silenciados.
        """
        logger.warning(
            "La captura de audio de la pantalla es una funcionalidad avanzada y altamente "
            "dependiente del sistema operativo. Esta implementación es una simulación. "
            "Se requiere integración con librerías específicas (ej. `py-get-window-audio` "
            "o similares) para una implementación real."
        )
        # Simular un stream de silencio
        silence_chunk = b'\x00' * settings.AUDIO_CHUNK_SIZE * 2 # 2 bytes por muestra (int16)
        for _ in range(500): # Simular ~10 segundos de silencio
            yield silence_chunk
            await asyncio.sleep(settings.AUDIO_CHUNK_SIZE / settings.AUDIO_SAMPLE_RATE)
        logger.info(f"Stream de simulación de captura de pantalla para {source_tag} finalizado.")

    async def start_stream(self, source_type: Literal["microphone", "file", "screen"], source_tag: str, **kwargs) -> AsyncGenerator[bytes, None]:
        """
        Inicia un stream de audio para una fuente específica.
        Args:
            source_type: Tipo de fuente de audio ("microphone", "file", "screen").
            source_tag: Etiqueta única para esta fuente (ej. "mic_input_1", "my_meeting.wav").
            **kwargs: Parámetros específicos para la fuente (ej. file_path para "file").
        Returns:
            Un generador asíncrono que produce chunks de bytes de audio.
        """
        if source_tag in self.active_streams:
            logger.warning(f"El stream con la etiqueta '{source_tag}' ya está activo. No se iniciará uno nuevo.")
            return

        generator = None
        if source_type == "microphone":
            generator = self._microphone_stream_generator(source_tag)
        elif source_type == "file":
            file_path = kwargs.get("file_path")
            if not file_path:
                raise ValueError("Se requiere 'file_path' para el tipo de fuente 'file'.")
            generator = self._file_stream_generator(file_path, source_tag)
        elif source_type == "screen":
            generator = self._screen_capture_stream_generator(source_tag)
        else:
            raise ValueError(f"Tipo de fuente de audio no soportado: {source_type}")

        return generator

    def stop_stream(self, source_tag: str):
        """Detiene un stream de audio activo."""
        if source_tag in self.active_streams:
            task = self.active_streams.pop(source_tag)
            task.cancel()
            logger.info(f"Stream de audio '{source_tag}' cancelado.")
        else:
            logger.warning(f"No se encontró un stream activo con la etiqueta '{source_tag}' para detener.")

    def get_active_streams(self) -> list[str]:
        """Devuelve una lista de las etiquetas de los streams activos."""
        return list(self.active_streams.keys())

# Instancia global del AudioManager
audio_manager = AudioManager()