use std::sync::mpsc::Sender;
pub const WHISPER_SAMPLE_RATE: u32 = 16000;
pub const CHUNK_DURATION_SECS: u32 = 5; 
pub const SILENCE_THRESHOLD: f32 = 0.1; 

// Tipos de fuente de audio
#[derive(Clone, Debug, PartialEq)]
pub enum SourceType {
    Input,
    Output,
}

// Estructura para listar dispositivos brutos
#[derive(Clone, Debug, PartialEq)]
pub struct DeviceInfo {
    pub id: usize,
    pub name: String,
    pub technical_name: Option<String>, // Nombre técnico para búsqueda en cpal/pactl
}

// Perfil completo del Interlocutor
#[derive(Clone, Debug, PartialEq)]
pub struct InterlocutorProfile {
    pub id: usize,
    pub device_id: usize,
    pub source_type: SourceType,
    pub name: String,
    pub is_active: bool,
    pub technical_name: Option<String>, // Nombre técnico del dispositivo (de pactl)
}

// Mensajes de comunicación entre el hilo de audio y la UI
pub enum AudioMessage {
    Status(String),
    Transcription { text: String, name: String }, // Incluye el nombre del interlocutor
    Error(String),
}

// Enum para la navegación
#[derive(Debug, PartialEq, Eq)]
pub enum View {
    Transcription,
    Settings,
}

// Alias para el canal de comunicación de la UI
pub type UiSender = Sender<AudioMessage>;