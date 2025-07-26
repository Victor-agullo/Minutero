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
        """Carga el modelo Whisper."""
        if self._model is None:
            try:
                # Ejecutar la carga del modelo en un thread pool para no bloquear el bucle de eventos
                self._model = await asyncio.to_thread(whisper.load_model, self.model_size)
                self.is_loaded = True
                logger.info(f"Modelo Whisper '{self.model_size}' cargado exitosamente.")
            except Exception as e:
                logger.error(f"Error al cargar el modelo Whisper '{self.model_size}': {e}")
                raise
        else:
            logger.info(f"Modelo Whisper '{self.model_size}' ya está cargado.")

    async def unload_model(self) -> None:
        """Descarga el modelo Whisper. Para modelos locales, esto es principalmente marcar como no cargado."""
        if self._model is not None:
            # En Python, el garbage collector se encarga de esto. Aquí, simplemente liberamos la referencia.
            del self._model
            self._model = None
            self.is_loaded = False
            logger.info(f"Modelo Whisper '{self.model_size}' descargado.")

    async def transcribe(self, audio_data: bytes, **kwargs) -> TranscriptionOutput:
        """
        Transcribe un bloque de audio completo usando Whisper.
        audio_data debe ser PCM de 16kHz, 16-bit, mono.
        """
        if not self.is_loaded or self._model is None:
            raise RuntimeError("El modelo Whisper no está cargado. Llama a load_model() primero.")

        # Convertir bytes a array de numpy (float32)
        # Asumiendo audio_data es PCM 16-bit little-endian
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        language = kwargs.get("language", "en")
        initial_prompt = kwargs.get("initial_prompt", None)

        try:
            result = await asyncio.to_thread(
                self._model.transcribe,
                audio_np,
                language=language,
                initial_prompt=initial_prompt,
                fp16=False # Usar fp16=False para evitar problemas en CPUs sin soporte AVX512_FP16
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
                total_duration=audio_np.shape[0] / settings.AUDIO_SAMPLE_RATE # Duración en segundos
            )
        except Exception as e:
            logger.error(f"Error durante la transcripción Whisper: {e}", exc_info=True)
            raise

    async def transcribe_stream(self, audio_stream: AsyncGenerator[bytes, None], **kwargs) -> AsyncGenerator[TranscriptionSegment, None]:
        """
        Simula la transcripción de un stream en tiempo real con Whisper.
        Acumula audio y lo transcribe en bloques para simular streaming.
        """
        if not self.is_loaded or self._model is None:
            raise RuntimeError("El modelo Whisper no está cargado. Llama a load_model() primero.")

        audio_buffer = io.BytesIO()
        total_audio_length = 0.0
        last_transcription_end = 0.0
        BUFFER_DURATION_SECONDS = 5

        async for chunk in audio_stream:
            audio_buffer.write(chunk)
            current_buffer_size_bytes = audio_buffer.tell()
            
            # Procesar si el buffer es lo suficientemente grande
            if current_buffer_size_bytes / (settings.AUDIO_SAMPLE_RATE * 2) >= BUFFER_DURATION_SECONDS:
                audio_for_processing = audio_buffer.getvalue()
                audio_buffer = io.BytesIO() # Reset buffer

                try:
                    audio_np = np.frombuffer(audio_for_processing, dtype=np.int16).astype(np.float32) / 32768.0
                    result = await asyncio.to_thread(
                        self._model.transcribe, audio_np, language=kwargs.get("language", "en"), fp16=False
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

        # Procesar el audio restante en el buffer al final
        remaining_audio = audio_buffer.getvalue()
        if remaining_audio:
            try:
                audio_np = np.frombuffer(remaining_audio, dtype=np.int16).astype(np.float32) / 32768.0
                result = await asyncio.to_thread(
                    self._model.transcribe, audio_np, language=kwargs.get("language", "en"), fp16=False
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
        """Devuelve las capacidades del modelo Whisper."""
        return ModelCapabilities(
            realtime_streaming=True, # Simulada, pero funcional para el usuario
            supported_languages=["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko", "ar", "hi", "tr", "pl", "nl", "sv", "da", "fi", "no", "he", "uk", "hu", "cs", "el", "bg", "ro", "sk", "sl", "lt", "lv", "et", "sq", "mk", "mt", "is", "ga", "gd", "cy", "eu", "ca", "gl"],
            description=f"Modelo OpenAI Whisper ({self.model_size} size) para transcripción offline/semi-streaming."
        )

    @staticmethod
    async def download_model_only(model_size: str):
        """Descarga un modelo Whisper sin cargarlo permanentemente en memoria."""
        try:
            logger.info(f"Iniciando descarga del modelo Whisper '{model_size}'...")
            # Cargar el modelo lo descarga a la caché por defecto (~/.cache/whisper)
            # si no está presente. Luego lo eliminamos de memoria.
            model = await asyncio.to_thread(whisper.load_model, model_size)
            del model # Liberar la memoria
            logger.info(f"Modelo Whisper '{model_size}' descargado exitosamente (o ya estaba presente en la caché).")
        except Exception as e:
            logger.error(f"Error al descargar el modelo Whisper '{model_size}': {e}")
            raise