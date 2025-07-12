# core/models.py

from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, Any, List, Literal
from pydantic import BaseModel, Field

# --- Modelos de Datos Pydantic ---

class TranscriptionSegment(BaseModel):
    """Representa un segmento de texto transcrito con su marca de tiempo."""
    text: str
    start: float  # Tiempo de inicio en segundos
    end: float    # Tiempo de fin en segundos
    source_tag: str = "" # Etiqueta de la fuente de audio (ej: "mic", "screen", "file")

class TranscriptionOutput(BaseModel):
    """Representa el resultado completo de una transcripción."""
    segments: List[TranscriptionSegment]
    model_name: str
    language: str = "en"
    total_duration: float = 0.0

class ModelCapabilities(BaseModel):
    """Define las capacidades de un modelo de transcripción."""
    realtime_streaming: bool = False
    supported_languages: List[str] = ["en"]
    max_audio_length_seconds: int | None = None # None si no hay límite
    description: str = ""

# --- Interfaz Abstracta para Modelos de Transcripción ---

class BaseModel(ABC):
    """
    Interfaz abstracta para los modelos de transcripción.
    Define los métodos que cualquier modelo de transcripción debe implementar.
    """
    model_name: str
    is_loaded: bool = False

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    async def load_model(self, **kwargs) -> None:
        """Carga el modelo de transcripción. Puede ser asíncrono."""
        pass

    @abstractmethod
    async def transcribe(self, audio_data: bytes, **kwargs) -> TranscriptionOutput:
        """
        Transcribe un bloque de audio completo.
        audio_data: bytes raw de audio (PCM, 16kHz, 16-bit, mono).
        """
        pass

    @abstractmethod
    async def transcribe_stream(self, audio_stream: AsyncGenerator[bytes, None], **kwargs) -> AsyncGenerator[TranscriptionSegment, None]:
        """
        Transcribe un stream de audio en tiempo real.
        audio_stream: Generador asíncrono de chunks de bytes de audio.
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> ModelCapabilities:
        """Devuelve las capacidades del modelo."""
        pass

    @abstractmethod
    async def unload_model(self) -> None:
        """Descarga el modelo de la memoria si es necesario."""
        pass