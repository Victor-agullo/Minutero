# main.py

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.responses import HTMLResponse
from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel as PydanticBaseModel
from fastapi.middleware.cors import CORSMiddleware

from core.models import TranscriptionSegment, ModelCapabilities
from core.transcription_engine import transcription_engine
from core.model_factory import model_factory
from utils.logger import logger
from config.settings import settings

# Importar WhisperModel para acceder a su método estático
from plugins.whisper_model import WhisperModel

# --- Lifespan Event Handler ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestiona los eventos de arranque y apagado de la aplicación.
    """
    # Lógica de arranque (startup)
    logger.info("La aplicación está arrancando. Precargando modelo por defecto...")
    try:
        # Pre-cargar el modelo por defecto al iniciar el servidor
        await transcription_engine.load_transcription_model(settings.DEFAULT_MODEL)
    except Exception as e:
        logger.error(f"No se pudo precargar el modelo por defecto '{settings.DEFAULT_MODEL}': {e}", exc_info=True)
        # Aquí puedes decidir si el arranque debería fallar o continuar sin el modelo cargado.
    
    yield # La aplicación se ejecuta aquí, entre el arranque y el apagado.

    # Lógica de apagado (shutdown)
    logger.info("La aplicación se está cerrando. Deteniendo todas las transcripciones activas y descargando modelos...")
    # Detener todas las tareas de transcripción activas
    active_tags = list(transcription_engine.transcription_tasks.keys())
    for tag in active_tags:
        await transcription_engine.stop_transcription_stream(tag)

    # Descargar el modelo actual si está cargado
    if transcription_engine.current_model:
        await transcription_engine.current_model.unload_model()
    logger.info("Aplicación apagada.")


# Inicializar el logger al inicio de la aplicación
logger.info("Iniciando aplicación FastAPI.")

app = FastAPI(
    title="Audio Transcriber Backend",
    description="Backend modular para transcripción de audio en tiempo real.",
    version="1.0.0",
    lifespan=lifespan # Registrar el nuevo manejador de eventos
)

# <--- Añadir la configuración del middleware CORS aquí --->
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todos los orígenes. En producción, especifica tus dominios.
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos (GET, POST, PUT, DELETE, OPTIONS, etc.)
    allow_headers=["*"],  # Permite todas las cabeceras
)
# <--- Fin de la configuración del middleware CORS --->

# --- Dependencias (opcional, pero buena práctica para inyección) ---
def get_transcription_engine():
    return transcription_engine

def get_model_factory():
    return model_factory

# --- Modelos Pydantic para Request/Response ---

class LoadModelRequest(PydanticBaseModel):
    model_name: str = settings.DEFAULT_MODEL # Por defecto a Whisper
    model_config_kwargs: Dict[str, Any] = {} # Para pasar 'model_size' a WhisperModel

class DownloadModelRequest(PydanticBaseModel):
    model_size: str = "base" # Tamaño del modelo Whisper a descargar

class StartTranscriptionRequest(PydanticBaseModel):
    source_type: Literal["microphone", "file", "screen"]
    source_tag: str # Etiqueta única para identificar la fuente (ej: "mic_user1", "meeting_recording.wav")
    file_path: Optional[str] = None # Solo si source_type es "file"
    language: str = "en"
    initial_prompt: Optional[str] = None

class StopTranscriptionRequest(PydanticBaseModel):
    source_tag: str

# --- Endpoint de prueba para el frontend (opcional) ---
@app.get("/")
async def read_root():
    return HTMLResponse("""
    <h1>Bienvenido al Backend de Transcripción de Audio</h1>
    <p>Usa los endpoints de API para interactuar:</p>
    <ul>
        <li><code>/models/load</code> (POST) para cargar un modelo.</li>
        <li><code>/models/download</code> (POST) para descargar un modelo.</li>
        <li><code>/models/available</code> (GET) para ver modelos disponibles.</li>
        <li><code>/models/capabilities/{model_name}</code> (GET) para ver capacidades de un modelo.</li>
        <li><code>/transcribe/ws/{source_tag}</code> (WebSocket) para iniciar y recibir transcripciones en tiempo real.</li>
        <li><code>/transcribe/active</code> (GET) para ver transcripciones activas.</li>
    </ul>
    <p>Para la transcripción en tiempo real, tu frontend debe conectarse vía WebSocket a <code>ws://localhost:8000/transcribe/ws/{source_tag}</code>.</p>
    """)


# --- Endpoints HTTP ---

@app.post("/models/load", summary="Cargar un modelo de transcripción")
async def load_model(
    request: LoadModelRequest,
    engine = Depends(get_transcription_engine)
):
    """
    Carga el modelo de transcripción especificado.
    Solo un modelo puede estar cargado a la vez. Si ya hay uno cargado, se descargará primero.
    """
    try:
        # Pasa model_config_kwargs a load_transcription_model
        # Esto permite que el WhisperModel reciba "model_size" al instanciarse
        await engine.load_transcription_model(request.model_name, **request.model_config_kwargs)
        return {"message": f"Modelo '{request.model_name}' cargado exitosamente."}
    except Exception as e:
        logger.error(f"Error al cargar el modelo '{request.model_name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Fallo al cargar el modelo: {e}")

@app.post("/models/download", summary="Descargar un modelo de Whisper")
async def download_whisper_model(
    request: DownloadModelRequest
):
    """
    Descarga un modelo de Whisper específico (ej. "base", "medium", "large") a la caché local.
    Esto no carga el modelo en memoria.
    """
    try:
        await WhisperModel.download_model_only(request.model_size)
        return {"message": f"Modelo Whisper '{request.model_size}' descargado exitosamente (o ya estaba en caché)."}
    except Exception as e:
        logger.error(f"Error al descargar el modelo Whisper '{request.model_size}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Fallo al descargar el modelo: {e}")


@app.get("/models/available", response_model=List[str], summary="Obtener modelos de transcripción disponibles")
async def get_available_models(
    factory = Depends(get_model_factory)
):
    """
    Devuelve una lista de los nombres de los modelos de transcripción que están disponibles.
    Estos modelos se descubren dinámicamente en el directorio 'plugins'.
    """
    return factory.get_available_models()

@app.get("/models/capabilities/{model_name}", response_model=ModelCapabilities, summary="Obtener capacidades de un modelo")
async def get_model_capabilities(
    model_name: str,
    factory = Depends(get_model_factory)
):
    """
    Devuelve las capacidades de un modelo de transcripción específico (ej. soporte de streaming, idiomas).
    El modelo no necesita estar cargado para obtener sus capacidades.
    """
    try:
        # Instanciar temporalmente el modelo para obtener sus capacidades sin cargarlo completamente
        # Esto asume que get_capabilities no requiere que el modelo esté cargado internamente
        # Si un modelo necesitara cargarse para get_capabilities, esto debería ajustarse.
        temp_model = factory.get_model_instance(model_name)
        return temp_model.get_capabilities()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error al obtener capacidades para el modelo '{model_name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error interno: {e}")


@app.get("/transcribe/active", summary="Obtener transcripciones activas")
async def get_active_transcriptions(
    engine = Depends(get_transcription_engine)
):
    """
    Devuelve una lista de las fuentes de audio que actualmente están siendo transcritas
    en tiempo real a través de WebSockets.
    """
    return engine.get_active_transcriptions()

# --- Endpoint WebSocket para Transcripción en Tiempo Real ---

@app.websocket("/transcribe/ws/{source_tag}")
async def websocket_transcribe(
    websocket: WebSocket,
    source_tag: str, # Etiqueta única para esta conexión/fuente
    language: str = "en",
    initial_prompt: Optional[str] = None,
    engine = Depends(get_transcription_engine)
):
    """
    Establece una conexión WebSocket para transcripción en tiempo real.
    El cliente puede enviar un mensaje JSON al inicio para configurar la fuente de audio.
    El backend enviará segmentos de transcripción a medida que estén disponibles.

    Para iniciar:
    1. Conectar a `ws://localhost:8000/transcribe/ws/{your_source_tag}?language=es`
    2. Enviar un JSON al inicio para configurar la fuente:
       `{"action": "start", "source_type": "microphone"}`
       o `{"action": "start", "source_type": "file", "file_path": "/path/to/your/audio.wav"}`
       o `{"action": "start", "source_type": "screen"}`
    3. El backend empezará a enviar segmentos de transcripción.
    4. Para detener (desde el cliente): cerrar la conexión WebSocket o enviar `{"action": "stop"}`.
    """
    await websocket.accept()
    logger.info(f"Conexión WebSocket establecida para source_tag: {source_tag}")

    # Callback para enviar segmentos de transcripción al cliente via WebSocket
    async def send_segment_to_client(segment: TranscriptionSegment):
        try:
            await websocket.send_json(segment.model_dump()) # Usar model_dump() para Pydantic v2
        except WebSocketDisconnect:
            logger.warning(f"Cliente '{source_tag}' desconectado mientras se enviaba segmento.")
            # La tarea principal detectará la desconexión
        except Exception as e:
            logger.error(f"Error al enviar segmento a '{source_tag}' via WebSocket: {e}", exc_info=True)

    transcription_active = False
    try:
        # El cliente envía un mensaje inicial para configurar la transcripción
        initial_message = await asyncio.wait_for(websocket.receive_json(), timeout=settings.WEBSOCKET_TIMEOUT)
        action = initial_message.get("action")

        if action == "start":
            start_req = StartTranscriptionRequest(**initial_message)
            # Asegurarse de que el source_tag del path coincida con el del mensaje
            if start_req.source_tag != source_tag:
                raise ValueError("El 'source_tag' en el path no coincide con el del mensaje inicial.")

            logger.info(f"Recibida solicitud de inicio de transcripción para '{source_tag}' (Tipo: {start_req.source_type})")

            # Iniciar la transcripción en el motor
            await engine.start_transcription_stream(
                source_type=start_req.source_type,
                source_tag=start_req.source_tag,
                output_callback=send_segment_to_client,
                file_path=start_req.file_path,
                language=language, # Usar el idioma de la URL o por defecto
                initial_prompt=initial_prompt
            )
            transcription_active = True
            await websocket.send_json({"status": "started", "message": f"Transcripción iniciada para {source_tag}"})

        elif action == "stop":
            logger.info(f"Recibida solicitud de detención de transcripción para '{source_tag}'")
            await engine.stop_transcription_stream(source_tag)
            await websocket.send_json({"status": "stopped", "message": f"Transcripción detenida para {source_tag}"})
            await websocket.close() # Cerrar la conexión después de detener
            return
        else:
            raise ValueError(f"Acción inicial no válida: {action}. Esperado 'start' o 'stop'.")

        # Mantener la conexión abierta mientras la transcripción está activa
        # Esto permite al cliente enviar mensajes adicionales (ej. 'stop') o simplemente esperar.
        while transcription_active:
            try:
                # Esperar mensajes del cliente (ej. para detener) o simplemente mantener viva la conexión
                message = await asyncio.wait_for(websocket.receive_json(), timeout=settings.WEBSOCKET_TIMEOUT)
                if message.get("action") == "stop":
                    logger.info(f"Recibida solicitud de detención vía mensaje para '{source_tag}'")
                    await engine.stop_transcription_stream(source_tag)
                    transcription_active = False # Salir del bucle
                    await websocket.send_json({"status": "stopped", "message": f"Transcripción detenida para {source_tag}"})
                # Aquí podrías manejar otros comandos del cliente si los hubiera
                else:
                    logger.warning(f"Mensaje no reconocido del cliente '{source_tag}': {message}")
            except asyncio.TimeoutError:
                # Si no hay mensajes del cliente por un tiempo, se puede enviar un ping o cerrar
                logger.debug(f"Timeout en la espera de mensaje del cliente '{source_tag}'. Conexión activa.")
                # Puedes enviar un ping aquí si quieres mantener la conexión viva proactivamente
            except WebSocketDisconnect:
                logger.info(f"Cliente '{source_tag}' se ha desconectado.")
                break # Salir del bucle y limpiar
            except Exception as e:
                logger.error(f"Error en el bucle WebSocket para '{source_tag}': {e}", exc_info=True)
                break # Salir del bucle en caso de error

    except WebSocketDisconnect:
        logger.info(f"Cliente '{source_tag}' desconectado inesperadamente.")
    except asyncio.TimeoutError:
        logger.warning(f"Timeout inicial para la configuración del WebSocket de '{source_tag}'. Cerrando conexión.")
    except Exception as e:
        logger.error(f"Error fatal durante la conexión WebSocket para '{source_tag}': {e}", exc_info=True)
        try:
            await websocket.send_json({"status": "error", "message": str(e)})
        except Exception:
            pass # No podemos enviar si ya hay un error de conexión
    finally:
        if source_tag in engine.transcription_tasks:
            await engine.stop_transcription_stream(source_tag)
        try:
            if not websocket.client_state.DISCONNECTED:
                await websocket.close()
        except Exception:
            pass # Ignorar errores al cerrar la conexión
        logger.info(f"Conexión WebSocket para '{source_tag}' cerrada.")

# --- Función para iniciar el servidor (opcional) ---
if __name__ == "__main__":
    import uvicorn
    # Se añade la opción loop="asyncio" y ws="websockets" para compatibilidad y rendimiento
    # En versiones recientes de Uvicorn, esto puede ser inferido, pero es explícito.
    uvicorn.run(app, host="0.0.0.0", port=8000, loop="asyncio", ws="websockets")