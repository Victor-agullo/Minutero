# utils/logger.py

import logging
import sys
from logging.handlers import TimedRotatingFileHandler

def setup_logging():
    """
    Configura un logger estructurado para el proyecto.
    Permite salida a consola y a archivo rotativo.
    """
    logger = logging.getLogger("audio_transcriber")
    logger.setLevel(logging.INFO)

    # Formateador para logs JSON (puedes personalizarlo)
    # En un entorno de producción, podrías usar una librería como `python-json-logger`
    # Para simplicidad, usaremos un formato más legible aquí.
    formatter = logging.Formatter(
        '{"time": "%(asctime)s", "level": "%(levelname)s", "module": "%(name)s", "message": "%(message)s"}'
    )

    # Handler para la consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Handler para archivos rotativos (opcional, para producción)
    # log_file = "logs/audio_transcriber.log"
    # os.makedirs(os.path.dirname(log_file), exist_ok=True)
    # file_handler = TimedRotatingFileHandler(
    #     log_file, when="midnight", interval=1, backupCount=7
    # )
    # file_handler.setFormatter(formatter)
    # logger.addHandler(file_handler)

    # Evitar que los loggers propaguen mensajes a los handlers de la raíz
    logger.propagate = False

    return logger

logger = setup_logging()