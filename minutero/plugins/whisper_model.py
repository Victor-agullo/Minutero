# plugins/whisper_model.py

import asyncio
import io
import numpy as np
from typing import AsyncGenerator
import whisper
from core.models import BaseModel, TranscriptionOutput, TranscriptionSegment, ModelCapabilities
from utils.logger import logger
from config.settings import settings

class WhisperModel(BaseModel):
    """
    Implementación del modelo de transcripción usando OpenAI Whisper (local).
    Soporta transcripción de archivos y, con manejo de chunks, simula streaming.
    """
    def __init__(self, model_size: str = "base"):
        super().__init__(model_name=f"whisper-{model_size}")
        self._model = None
        self.model_size = model_size
        logger.info(f"Inicializando WhisperModel con tamaño: {model_size}")

    async def load_model(self) -> None:
        # ... (sin cambios)
        if self._model is None:
            try:
                self._model = await asyncio.to_thread(whisper.load_model, self.model_size)
                self.is_loaded = True
                logger.info(f"Modelo Whisper '{self.model_size}' cargado exitosamente.")
            except Exception as e:
                logger.error(f"Error al cargar el modelo Whisper '{self.model_size}': {e}")
                raise
        else:
            logger.info(f"Modelo Whisper '{self.model_size}' ya está cargado.")

    async def unload_model(self) -> None:
        # ... (sin cambios)
        if self._model is not None:
            del self._model
            self._model = None
            self.is_loaded = False
            logger.info(f"Modelo Whisper '{self.model_size}' descargado.")

    async def transcribe(self, audio_data: bytes, **kwargs) -> TranscriptionOutput:
        # ... (sin cambios en la lógica principal, pero se usará el nuevo parámetro desde donde se llame)
        if not self.is_loaded or self._model is None:
            raise RuntimeError("El modelo Whisper no está cargado. Llama a load_model() primero.")

        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        language = kwargs.get("language", "en")
        initial_prompt = kwargs.get("initial_prompt", None)
        
        # CAMBIO: Añadir no_speech_threshold para evitar alucinaciones con silencio
        transcribe_options = {
            "language": language,
            "initial_prompt": initial_prompt,
            "fp16": False,
            "no_speech_threshold": 0.6, # Umbral de confianza para detectar si hay voz
            "condition_on_previous_text": False # Ayuda a reducir la repetición
        }
        
        try:
            result = await asyncio.to_thread(
                self._model.transcribe,
                audio_np,
                **transcribe_options
            )

            segments_output = []
            for seg in result.get("segments", []):
                segments_output.append(
                    TranscriptionSegment(
                        text=seg["text"].strip(),
                        start=seg["start"],
                        end=seg["end"],
                        source_tag=kwargs.get("source_tag", "")
                    )
                )

            logger.info(f"Transcripción Whisper finalizada para {kwargs.get('source_tag', 'desconocida')}. Texto: '{result.get('text', '')[:50]}...'")
            return TranscriptionOutput(
                segments=segments_output,
                model_name=self.model_name,
                language=language,
                total_duration=audio_np.shape[0] / settings.AUDIO_SAMPLE_RATE
            )
        except Exception as e:
            logger.error(f"Error durante la transcripción Whisper: {e}", exc_info=True)
            raise

    async def transcribe_stream(self, audio_stream: AsyncGenerator[bytes, None], **kwargs) -> AsyncGenerator[TranscriptionSegment, None]:
        if not self.is_loaded or self._model is None:
            raise RuntimeError("El modelo Whisper no está cargado. Llama a load_model() primero.")

        audio_buffer = io.BytesIO()
        total_audio_length = 0.0
        BUFFER_DURATION_SECONDS = 5
        
        # CAMBIO: Mismas opciones de transcripción para el modo streaming
        transcribe_options = {
            "language": kwargs.get("language", "en"),
            "fp16": False,
            "no_speech_threshold": 0.6,
            "condition_on_previous_text": False
        }

        async for chunk in audio_stream:
            audio_buffer.write(chunk)
            current_buffer_size_bytes = audio_buffer.tell()
            
            if current_buffer_size_bytes / (settings.AUDIO_SAMPLE_RATE * 2) >= BUFFER_DURATION_SECONDS:
                audio_for_processing = audio_buffer.getvalue()
                audio_buffer = io.BytesIO()

                try:
                    audio_np = np.frombuffer(audio_for_processing, dtype=np.int16).astype(np.float32) / 32768.0
                    result = await asyncio.to_thread(
                        self._model.transcribe, audio_np, **transcribe_options
                    )
                    for seg in result.get("segments", []):
                        yield TranscriptionSegment(
                            text=seg["text"].strip(),
                            start=seg["start"] + total_audio_length,
                            end=seg["end"] + total_audio_length,
                            source_tag=kwargs.get("source_tag", "")
                        )
                    total_audio_length += len(audio_for_processing) / (settings.AUDIO_SAMPLE_RATE * 2)
                except Exception as e:
                    logger.error(f"Error en transcripción de stream Whisper: {e}", exc_info=True)

        remaining_audio = audio_buffer.getvalue()
        if remaining_audio:
            try:
                audio_np = np.frombuffer(remaining_audio, dtype=np.int16).astype(np.float32) / 32768.0
                result = await asyncio.to_thread(
                    self._model.transcribe, audio_np, **transcribe_options
                )
                for seg in result.get("segments", []):
                    yield TranscriptionSegment(
                        text=seg["text"].strip(),
                        start=seg["start"] + total_audio_length,
                        end=seg["end"] + total_audio_length,
                        source_tag=kwargs.get("source_tag", "")
                    )
            except Exception as e:
                logger.error(f"Error al transcribir audio restante: {e}", exc_info=True)

    def get_capabilities(self) -> ModelCapabilities:
        # ... (sin cambios)
        return ModelCapabilities(
            realtime_streaming=True,
            supported_languages=["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko", "ar", "hi", "tr", "pl", "nl", "sv", "da", "fi", "no", "he", "uk", "hu", "cs", "el", "bg", "ro", "sk", "sl", "lt", "lv", "et", "sq", "mk", "mt", "is", "ga", "gd", "cy", "eu", "ca", "gl"],
            description=f"Modelo OpenAI Whisper ({self.model_size} size) para transcripción offline/semi-streaming."
        )

    @staticmethod
    async def download_model_only(model_size: str):
        # ... (sin cambios)
        try:
            logger.info(f"Iniciando descarga del modelo Whisper '{model_size}'...")
            model = await asyncio.to_thread(whisper.load_model, model_size)
            del model
            logger.info(f"Modelo Whisper '{model_size}' descargado exitosamente (o ya estaba presente en la caché).")
        except Exception as e:
            logger.error(f"Error al descargar el modelo Whisper '{model_size}': {e}")
            raise