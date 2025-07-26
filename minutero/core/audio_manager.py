import asyncio
import wave
import os
import ctypes
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator, Literal, Dict

import sounddevice as sd
import soundcard as sc

from utils.logger import logger
from config.settings import settings


class AudioManager:
    """
    Gestiona la captura de audio de diferentes fuentes (micrófono, archivo, loopback de pantalla).
    Devuelve streams de audio como generadores asíncronos de bytes PCM.
    """

    def __init__(self):
        self.active_streams: Dict[str, asyncio.Task] = {}
        logger.info("AudioManager inicializado.")

    async def _microphone_stream_generator(self, source_tag: str) -> AsyncGenerator[bytes, None]:
        """Generador asíncrono para capturar audio del micrófono."""
        try:
            device_info = sd.query_devices(kind='input')
            device_id = device_info['index']
            logger.info(f"Iniciando captura de micrófono. Dispositivo: {device_info['name']} (ID: {device_id})")

            loop = asyncio.get_running_loop()
            event = asyncio.Event()
            q: asyncio.Queue[bytes] = asyncio.Queue()

            def callback(indata, frames, time, status):
                if status:
                    logger.warning(f"Status del stream de micrófono: {status}")
                loop.call_soon_threadsafe(q.put_nowait, bytes(indata))
                loop.call_soon_threadsafe(event.set)

            with sd.InputStream(
                samplerate=settings.AUDIO_SAMPLE_RATE,
                channels=1,
                dtype='int16',
                callback=callback,
                blocksize=settings.AUDIO_CHUNK_SIZE,
                device=device_id
            ) as stream:
                while True:
                    await event.wait()
                    event.clear()
                    while not q.empty():
                        chunk = q.get_nowait()
                        yield chunk
                    await asyncio.sleep(0.001)

        except sd.PortAudioError as e:
            logger.error(
                f"Error de PortAudio al iniciar el micrófono: {e}. "
                "Asegúrate de que el micrófono esté conectado y los drivers instalados.",
                exc_info=True
            )
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
                    frames = wf.readframes(settings.AUDIO_CHUNK_SIZE)
                    if not frames:
                        break
                    yield frames
                    await asyncio.sleep(settings.AUDIO_CHUNK_SIZE / settings.AUDIO_SAMPLE_RATE)

        except Exception as e:
            logger.error(f"Error al leer el archivo de audio '{file_path}': {e}", exc_info=True)
            raise
        finally:
            logger.info(f"Stream de archivo para {source_tag} finalizado.")

    async def _screen_capture_stream_generator(self, source_tag: str) -> AsyncGenerator[bytes, None]:
        """
        Captura audio de la salida del sistema (loopback) usando SoundCard.
        Requiere que exista un micrófono virtual de loopback asociado al altavoz por defecto.
        Actualmente probado en Windows (WASAPI-loopback) con inicialización COM.
        """
        loop = asyncio.get_running_loop()
        q: asyncio.Queue[bytes] = asyncio.Queue()

        def _record_loopback(queue: asyncio.Queue, event_loop: asyncio.AbstractEventLoop):
            # Inicializar COM en este hilo (MTA)
            ctypes.windll.ole32.CoInitializeEx(None, 0)
            try:
                # Obtiene el micrófono loopback asociado al altavoz por defecto
                default_speaker = sc.default_speaker()
                loopback_mic = sc.get_microphone(
                    id=default_speaker.name,
                    include_loopback=True
                )
                logger.info(f"Iniciando loopback de sistema: {loopback_mic.name}")

                with loopback_mic.recorder(
                    samplerate=settings.AUDIO_SAMPLE_RATE,
                    channels=loopback_mic.channels
                ) as recorder:
                    while True:
                        frames = recorder.record(numframes=settings.AUDIO_CHUNK_SIZE)
                        event_loop.call_soon_threadsafe(queue.put_nowait, frames.tobytes())
            except Exception as e:
                logger.error(f"Error en loopback thread: {e}", exc_info=True)
            finally:
                ctypes.windll.ole32.CoUninitialize()

        # Arrancar el hilo de grabación
        executor = ThreadPoolExecutor(max_workers=1)
        record_task = loop.run_in_executor(executor, _record_loopback, q, loop)
        self.active_streams[source_tag] = record_task

        try:
            while True:
                chunk = await q.get()
                yield chunk
        except asyncio.CancelledError:
            logger.info(f"Stream de loopback '{source_tag}' cancelado por petición.")
        finally:
            executor.shutdown(wait=False)
            logger.info(f"Stream de loopback para {source_tag} finalizado.")

    async def start_stream(
        self,
        source_type: Literal["microphone", "file", "screen"],
        source_tag: str,
        **kwargs
    ) -> AsyncGenerator[bytes, None]:
        """
        Inicia un stream de audio para una fuente específica.
        Args:
            source_type: "microphone", "file" o "screen"
            source_tag: Etiqueta única para este stream
            **kwargs: Parámetros extra (p.ej. file_path para "file")
        """
        if source_tag in self.active_streams:
            logger.warning(f"El stream con la etiqueta '{source_tag}' ya está activo.")
            return

        if source_type == "microphone":
            generator = self._microphone_stream_generator(source_tag)
        elif source_type == "file":
            file_path = kwargs.get("file_path")
            if not file_path:
                raise ValueError("Se requiere 'file_path' para fuente 'file'.")
            generator = self._file_stream_generator(file_path, source_tag)
        elif source_type == "screen":
            generator = self._screen_capture_stream_generator(source_tag)
        else:
            raise ValueError(f"Tipo de fuente no soportado: {source_type}")

        return generator

    def stop_stream(self, source_tag: str):
        """Detiene un stream de audio activo."""
        if source_tag in self.active_streams:
            task = self.active_streams.pop(source_tag)
            task.cancel()
            logger.info(f"Stream de audio '{source_tag}' cancelado.")
        else:
            logger.warning(f"No existe stream activo con etiqueta '{source_tag}'.")

    def get_active_streams(self) -> list[str]:
        """Devuelve una lista de las etiquetas de los streams activos."""
        return list(self.active_streams.keys())


# Instancia global
audio_manager = AudioManager()
