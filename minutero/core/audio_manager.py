import asyncio
import wave
import os
import sys
import ctypes
import threading
import numpy as np
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
        self.active_streams: Dict[str, tuple[asyncio.Task, threading.Event | None]] = {}
        logger.info("AudioManager inicializado.")

    async def _microphone_stream_generator(self, source_tag: str) -> AsyncGenerator[bytes, None]:
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
                # Convertir a int16 si no lo está ya
                if indata.dtype != np.int16:
                    audio_data = (indata * 32767).astype(np.int16)
                else:
                    audio_data = indata
                loop.call_soon_threadsafe(q.put_nowait, audio_data.tobytes())
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
        Versión mejorada con mejor detección de dispositivos y manejo de datos.
        """
        loop = asyncio.get_running_loop()
        q: asyncio.Queue[bytes] = asyncio.Queue()
        stop_event = threading.Event()

        def _record_loopback(queue: asyncio.Queue, event_loop: asyncio.AbstractEventLoop, stop_event: threading.Event):
            if sys.platform == "win32":
                ctypes.windll.ole32.CoInitializeEx(None, 0)
            
            loopback_mic = None
            try:
                # Método 1: Intentar obtener el loopback del altavoz por defecto
                try:
                    default_speaker = sc.default_speaker()
                    logger.info(f"Altavoz por defecto: '{default_speaker.name}' (ID: {default_speaker.id})")
                    
                    # Buscar el loopback correspondiente al altavoz por defecto
                    all_mics = sc.all_microphones(include_loopback=True)
                    for mic in all_mics:
                        if mic.isloopback and mic.id == default_speaker.id:
                            loopback_mic = mic
                            logger.info(f"Loopback del altavoz por defecto encontrado: '{loopback_mic.name}'")
                            break
                            
                except Exception as e:
                    logger.warning(f"Error al buscar loopback del altavoz por defecto: {e}")

                # Método 2: Si no se encuentra, buscar cualquier loopback activo
                if not loopback_mic:
                    logger.info("Buscando cualquier dispositivo de loopback disponible...")
                    try:
                        all_mics = sc.all_microphones(include_loopback=True)
                        logger.info(f"Dispositivos de audio encontrados: {len(all_mics)}")
                        
                        for i, mic in enumerate(all_mics):
                            logger.info(f"  {i}: '{mic.name}' - Loopback: {mic.isloopback}")
                            if mic.isloopback:
                                loopback_mic = mic
                                logger.info(f"Usando dispositivo de loopback: '{loopback_mic.name}'")
                                break
                    except Exception as e:
                        logger.error(f"Error al listar dispositivos de audio: {e}")

                if not loopback_mic:
                    msg = "No se encontró ningún dispositivo de loopback. Verifica que el audio del sistema esté activo."
                    logger.error(msg)
                    event_loop.call_soon_threadsafe(queue.put_nowait, RuntimeError(msg))
                    return

                logger.info(f"Iniciando grabación con: '{loopback_mic.name}' (SR: {settings.AUDIO_SAMPLE_RATE}, Chunk: {settings.AUDIO_CHUNK_SIZE})")
                
                # Buffer para acumular datos y garantizar chunks consistentes
                audio_buffer = np.array([], dtype=np.float32)
                target_chunk_size = settings.AUDIO_CHUNK_SIZE
                
                with loopback_mic.recorder(
                    samplerate=settings.AUDIO_SAMPLE_RATE,
                    channels=1
                ) as recorder:
                    while not stop_event.is_set():
                        try:
                            # Grabar un chunk pequeño
                            data = recorder.record(numframes=target_chunk_size // 4)
                            
                            if data is not None and len(data) > 0:
                                # Normalizar forma de datos
                                if data.ndim == 2:
                                    data = data.flatten()
                                
                                # Convertir a float32 y normalizar
                                if data.dtype != np.float32:
                                    data = data.astype(np.float32)
                                
                                # Agregar al buffer
                                audio_buffer = np.concatenate([audio_buffer, data])
                                
                                # Enviar chunks completos del buffer
                                while len(audio_buffer) >= target_chunk_size:
                                    chunk = audio_buffer[:target_chunk_size]
                                    audio_buffer = audio_buffer[target_chunk_size:]
                                    
                                    # Convertir a int16 PCM
                                    chunk = np.clip(chunk, -1.0, 1.0)
                                    audio_data = (chunk * 32767).astype(np.int16)
                                    
                                    # Verificar audio válido
                                    if np.max(np.abs(audio_data)) > 50:
                                        logger.debug(f"Enviando chunk de audio: {len(audio_data)} muestras")
                                    
                                    if not stop_event.is_set():
                                        event_loop.call_soon_threadsafe(queue.put_nowait, audio_data.tobytes())
                            
                        except Exception as e:
                            logger.error(f"Error al grabar chunk de audio: {e}")
                            if not stop_event.is_set():
                                event_loop.call_soon_threadsafe(queue.put_nowait, e)
                            break

            except Exception as e:
                logger.error(f"Error en el hilo de grabación de loopback: {e}", exc_info=True)
                event_loop.call_soon_threadsafe(queue.put_nowait, e)
            finally:
                if sys.platform == "win32":
                    ctypes.windll.ole32.CoUninitialize()
                logger.info(f"Hilo de grabación de loopback para '{source_tag}' finalizado.")

        executor = ThreadPoolExecutor(max_workers=1)
        record_task = loop.run_in_executor(executor, _record_loopback, q, loop, stop_event)
        self.active_streams[source_tag] = (record_task, stop_event)

        try:
            chunk_count = 0
            while True:
                chunk = await q.get()
                if isinstance(chunk, Exception):
                    raise chunk
                
                chunk_count += 1
                if chunk_count % 100 == 0:  # Log cada 100 chunks para debugging
                    logger.debug(f"Procesados {chunk_count} chunks de audio del sistema")
                
                yield chunk
                
        except asyncio.CancelledError:
            logger.info(f"Stream de loopback '{source_tag}' cancelado por petición.")
        finally:
            stop_event.set()
            if record_task and not record_task.done():
                await asyncio.sleep(0.1)
            executor.shutdown(wait=True)
            if source_tag in self.active_streams:
                del self.active_streams[source_tag]
            logger.info(f"Stream de loopback para {source_tag} completamente finalizado.")

    async def start_stream(
        self,
        source_type: Literal["microphone", "file", "screen"],
        source_tag: str,
        **kwargs
    ) -> AsyncGenerator[bytes, None]:
        if source_tag in self.active_streams:
            logger.warning(f"El stream con la etiqueta '{source_tag}' ya está activo.")
            return
        
        if source_type == "microphone":
            return self._microphone_stream_generator(source_tag)
        elif source_type == "file":
            file_path = kwargs.get("file_path", "")
            if not file_path:
                raise ValueError("Se requiere 'file_path' para fuente 'file'.")
            return self._file_stream_generator(file_path, source_tag)
        elif source_type == "screen":
            return self._screen_capture_stream_generator(source_tag)
        else:
            raise ValueError(f"Tipo de fuente no soportado: {source_type}")

    async def stop_stream(self, source_tag: str):
        if source_tag in self.active_streams:
            task, stop_event = self.active_streams.pop(source_tag)
            
            if stop_event:
                stop_event.set()
            
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            logger.info(f"Se ha solicitado la detención del stream de audio '{source_tag}'.")
        else:
            logger.warning(f"No se encontró un stream activo con etiqueta '{source_tag}' para detener.")

    def get_active_streams(self) -> list[str]:
        return list(self.active_streams.keys())

audio_manager = AudioManager()