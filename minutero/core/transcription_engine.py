# core/transcription_engine.py

import asyncio
from typing import Dict, AsyncGenerator, Callable, Any, Literal, Optional
from core.models import BaseModel, TranscriptionSegment
from core.audio_manager import AudioManager, audio_manager
from core.model_factory import ModelFactory, model_factory
from utils.logger import logger

class TranscriptionEngine:
    """
    Orquesta la captura de audio, la transcripción y la emisión de resultados.
    Gestiona múltiples fuentes de audio y utiliza un modelo de transcripción cargado.
    """
    def __init__(self, audio_manager: AudioManager, model_factory: ModelFactory):
        self.audio_manager = audio_manager
        self.model_factory = model_factory
        self.current_model: BaseModel | None = None
        self.transcription_tasks: Dict[str, asyncio.Task] = {}
        logger.info("TranscriptionEngine inicializado.")

    async def load_transcription_model(self, model_name_key: str, **kwargs) -> None:
        """Carga un modelo de transcripción y lo establece como el modelo actual."""
        # CAMBIO: Lógica mejorada para comprobar si el modelo exacto ya está cargado.
        # WhisperModel nombra sus instancias como 'whisper-base', 'whisper-large', etc.
        model_size = kwargs.get('model_size', 'base')
        requested_model_name = f"whisper-{model_size}"

        if self.current_model and self.current_model.model_name == requested_model_name and self.current_model.is_loaded:
            logger.info(f"El modelo '{requested_model_name}' ya está cargado y es el actual.")
            return

        if self.current_model:
            logger.info(f"Descargando modelo actual: {self.current_model.model_name}")
            await self.current_model.unload_model()
            self.current_model = None

        logger.info(f"Cargando modelo de transcripción: {model_name_key} con configuración: {kwargs}")
        try:
            self.current_model = self.model_factory.get_model_instance(model_name_key, **kwargs)
            await self.current_model.load_model()
            logger.info(f"Modelo '{self.current_model.model_name}' cargado y listo para usar.")
        except Exception as e:
            logger.error(f"Fallo al cargar el modelo '{model_name_key}': {e}", exc_info=True)
            self.current_model = None
            raise

    async def _process_audio_stream(
        self,
        audio_stream_generator: AsyncGenerator[bytes, None],
        source_tag: str,
        output_callback: Callable[[TranscriptionSegment], Any],
        **transcription_kwargs
    ) -> None:
        """
        Procesa un stream de audio, lo pasa al modelo de transcripción
        y envía los segmentos resultantes al callback.
        """
        if not self.current_model:
            logger.error(f"No hay un modelo de transcripción cargado para procesar '{source_tag}'.")
            # Enviar mensaje de error al cliente
            await output_callback(TranscriptionSegment(text=f"[ERROR] No hay modelo cargado.", start=0, end=0, source_tag=source_tag))
            return

        logger.info(f"Iniciando procesamiento de stream para '{source_tag}' con modelo '{self.current_model.model_name}'")
        try:
            async for segment in self.current_model.transcribe_stream(audio_stream_generator, source_tag=source_tag, **transcription_kwargs):
                await output_callback(segment)
        except asyncio.CancelledError:
            logger.info(f"Tarea de transcripción para '{source_tag}' cancelada.")
        except Exception as e:
            logger.error(f"Error procesando stream de audio para '{source_tag}': {e}", exc_info=True)
        finally:
            logger.info(f"Procesamiento de stream para '{source_tag}' finalizado.")
            # Asegurarse de que el stream de audio subyacente también se detiene
            await self.audio_manager.stop_stream(source_tag)

    async def start_transcription_stream(
        self,
        source_type: Literal["microphone", "file", "screen"],
        source_tag: str,
        output_callback: Callable[[TranscriptionSegment], Any],
        file_path: Optional[str] = None,
        language: str = "en",
        initial_prompt: Optional[str] = None
    ) -> None:
        """
        Inicia la transcripción en tiempo real de una fuente de audio.
        """
        if source_tag in self.transcription_tasks:
            logger.warning(f"La transcripción para '{source_tag}' ya está activa. Deteniendo la anterior.")
            await self.stop_transcription_stream(source_tag)

        if not self.current_model or not self.current_model.is_loaded:
            logger.error("No hay un modelo de transcripción cargado. Por favor, carga un modelo primero.")
            raise RuntimeError("Modelo de transcripción no cargado.")

        try:
            audio_generator = await self.audio_manager.start_stream(
                source_type=source_type,
                source_tag=source_tag,
                file_path=file_path
            )

            transcription_task = asyncio.create_task(
                self._process_audio_stream(
                    audio_generator,
                    source_tag,
                    output_callback,
                    language=language,
                    initial_prompt=initial_prompt
                ),
                name=f"transcription_task_{source_tag}"
            )
            self.transcription_tasks[source_tag] = transcription_task
            logger.info(f"Transcripción en tiempo real iniciada para '{source_tag}'.")

        except Exception as e:
            logger.error(f"Error al iniciar la transcripción para '{source_tag}': {e}", exc_info=True)
            if source_tag in self.transcription_tasks:
                self.transcription_tasks[source_tag].cancel()
                del self.transcription_tasks[source_tag]
            raise

    async def stop_transcription_stream(self, source_tag: str) -> None:
        """Detiene la transcripción de una fuente de audio específica."""
        if source_tag in self.transcription_tasks:
            task = self.transcription_tasks.pop(source_tag)
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            logger.info(f"Transcripción para '{source_tag}' detenida.")
        else:
            logger.warning(f"No se encontró una transcripción activa para '{source_tag}' para detener.")
        
        # Detener explícitamente el stream de audio asociado
        await self.audio_manager.stop_stream(source_tag)

    def get_active_transcriptions(self) -> Dict[str, str]:
        """Devuelve un diccionario de las transcripciones activas y sus estados."""
        return {tag: "active" for tag in self.transcription_tasks.keys()}

# Instancia global del TranscriptionEngine
transcription_engine = TranscriptionEngine(audio_manager, model_factory)