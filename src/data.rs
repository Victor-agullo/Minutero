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
    pub technical_name: Option<String>,
}

// Perfil completo del Interlocutor
#[derive(Clone, Debug, PartialEq)]
pub struct InterlocutorProfile {
    pub id: usize,
    pub device_id: usize,
    pub source_type: SourceType,
    pub name: String,
    pub is_active: bool,
    pub technical_name: Option<String>,
}

// Configuración de idioma global para la sesión
#[derive(Clone, Debug, PartialEq)]
pub struct LanguageConfig {
    /// None = autodetección. Some("en"), Some("es"), etc.
    pub source_lang: Option<&'static str>,
    /// true = traducir a inglés (único destino que soporta Whisper nativamente)
    pub translate_to_english: bool,
}

impl Default for LanguageConfig {
    fn default() -> Self {
        Self {
            source_lang: Some("en"),
            translate_to_english: false,
        }
    }
}

impl LanguageConfig {
    pub fn source_label(&self) -> &'static str {
        match self.source_lang {
            None => "Auto",
            Some("en") => "English",
            Some("es") => "Español",
            Some("fr") => "Français",
            Some("de") => "Deutsch",
            Some("it") => "Italiano",
            Some("pt") => "Português",
            Some("zh") => "中文",
            Some("ja") => "日本語",
            Some(other) => other,
        }
    }

    pub fn dest_label(&self) -> &'static str {
        if self.translate_to_english {
            "English (traducir)"
        } else {
            "Original (sin traducción)"
        }
    }
}

pub const SOURCE_LANGUAGES: &[(&str, Option<&'static str>)] = &[
    ("Auto (detectar)", None),
    ("English",         Some("en")),
    ("Español",         Some("es")),
    ("Français",        Some("fr")),
    ("Deutsch",         Some("de")),
    ("Italiano",        Some("it")),
    ("Português",       Some("pt")),
    ("中文",            Some("zh")),
    ("日本語",          Some("ja")),
];

// Mensajes de comunicación entre el hilo de audio y la UI
pub enum AudioMessage {
    Status(String),
    Transcription { text: String, name: String },
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