# config/settings.py

import os

class Settings:
    # Directorio base del proyecto
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Directorio donde se buscarán los plugins de modelos
    PLUGINS_DIR: str = os.path.join(BASE_DIR, "plugins")
    # Modelo por defecto a cargar (inicialmente Whisper)
    DEFAULT_MODEL: str = "whisper"
    # Frecuencia de muestreo para el audio (común para audio y micrófono)
    AUDIO_SAMPLE_RATE: int = 16000
    # Tamaño del chunk de audio para procesamiento
    AUDIO_CHUNK_SIZE: int = 1024
    # Tiempo máximo de inactividad para las conexiones WebSocket (en segundos)
    WEBSOCKET_TIMEOUT: int = 60

settings = Settings()