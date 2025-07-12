# plugins/whisper_model.py

import asyncio
import io
import numpy as np
from typing import AsyncGenerator, List, Dict, Any
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

    async def load_model(self, **kwargs) -> None:
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
        Whisper no soporta streaming "verdadero" en el sentido de procesar un chunk
        y devolver un segmento incremental en milisegundos.
        Aquí, acumulamos chunks y los transcribimos periódicamente o al final.
        Para un streaming real, se necesitaría un modelo diseñado para ello
        o una implementación de "vad + transcribe chunk".
        """
        if not self.is_loaded or self._model is None:
            raise RuntimeError("El modelo Whisper no está cargado. Llama a load_model() primero.")

        # Buffering para acumular audio antes de transcribir
        audio_buffer = io.BytesIO()
        total_audio_length = 0.0 # Duración acumulada en segundos
        last_transcription_end = 0.0 # Marca de tiempo del final de la última transcripción

        # Parámetros para la transcripción en streaming simulado
        # Podemos transcribir cada N segundos de audio acumulado, o cuando haya una pausa.
        # Por simplicidad, transcribiremos cuando el buffer alcance cierto tamaño o al finalizar.
        BUFFER_DURATION_SECONDS = 5 # Transcribir cada 5 segundos de audio

        async for chunk in audio_stream:
            audio_buffer.write(chunk)
            total_audio_length = audio_buffer.tell() / (settings.AUDIO_SAMPLE_RATE * 2) # Bytes a segundos (16-bit PCM)

            # Si hemos acumulado suficiente audio o es el final del stream, transcribimos
            if total_audio_length - last_transcription_end >= BUFFER_DURATION_SECONDS:
                # Obtener el audio para transcribir desde la última marca
                current_audio_bytes = audio_buffer.getvalue()
                # Cortar desde el inicio o desde el último segmento transcrito
                audio_for_processing = current_audio_bytes[int(last_transcription_end * settings.AUDIO_SAMPLE_RATE * 2):]

                if audio_for_processing:
                    logger.debug(f"Transcribiendo chunk de {len(audio_for_processing)} bytes, duración: {len(audio_for_processing) / (settings.AUDIO_SAMPLE_RATE * 2):.2f}s")
                    try:
                        # Convertir a numpy array y normalizar
                        audio_np = np.frombuffer(audio_for_processing, dtype=np.int16).astype(np.float32) / 32768.0

                        result = await asyncio.to_thread(
                            self._model.transcribe,
                            audio_np,
                            language=kwargs.get("language", "en"),
                            initial_prompt=kwargs.get("initial_prompt", None),
                            fp16=False
                        )

                        # Ajustar los timestamps de los segmentos relativos al inicio de la grabación total
                        for seg in result.get("segments", []):
                            absolute_start = seg["start"] + last_transcription_end
                            absolute_end = seg["end"] + last_transcription_end
                            yield TranscriptionSegment(
                                text=seg["text"].strip(),
                                start=absolute_start,
                                end=absolute_end,
                                source_tag=kwargs.get("source_tag", "")
                            )
                        # Actualizar la marca de tiempo de la última transcripción
                        # Usamos la duración total del audio procesado para evitar saltos.
                        # Esto podría mejorarse con VAD para detectar silencios.
                        last_transcription_end = total_audio_length

                    except Exception as e:
                        logger.error(f"Error durante la transcripción de stream Whisper: {e}", exc_info=True)
                        # Continuar el stream a pesar del error de un chunk
                        continue

        # Al final del stream, transcribir cualquier audio restante
        remaining_audio_bytes = audio_buffer.getvalue()[int(last_transcription_end * settings.AUDIO_SAMPLE_RATE * 2):]
        if remaining_audio_bytes:
            logger.info(f"Transcribiendo audio restante al final del stream. Duración: {len(remaining_audio_bytes) / (settings.AUDIO_SAMPLE_RATE * 2):.2f}s")
            try:
                audio_np = np.frombuffer(remaining_audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                result = await asyncio.to_thread(
                    self._model.transcribe,
                    audio_np,
                    language=kwargs.get("language", "en"),
                    initial_prompt=kwargs.get("initial_prompt", None),
                    fp16=False
                )
                for seg in result.get("segments", []):
                    absolute_start = seg["start"] + last_transcription_end
                    absolute_end = seg["end"] + last_transcription_end
                    yield TranscriptionSegment(
                        text=seg["text"].strip(),
                        start=absolute_start,
                        end=absolute_end,
                        source_tag=kwargs.get("source_tag", "")
                    )
            except Exception as e:
                logger.error(f"Error al transcribir audio restante del stream: {e}", exc_info=True)


    def get_capabilities(self) -> ModelCapabilities:
        """Devuelve las capacidades del modelo Whisper."""
        return ModelCapabilities(
            realtime_streaming=False, # Whisper no es verdaderamente streaming en tiempo real
            supported_languages=["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko", "ar", "hi", "tr", "pl", "nl", "sv", "da", "fi", "no", "he", "uk", "hu", "cs", "el", "bg", "ro", "sk", "sl", "lt", "lv", "et", "sq", "mk", "mt", "is", "ga", "gd", "cy", "eu", "ca", "gl"],
            description=f"Modelo OpenAI Whisper ({self.model_size} size) para transcripción offline/semi-streaming."
        )