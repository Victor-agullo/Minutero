use std::sync::mpsc::Sender;
pub const WHISPER_SAMPLE_RATE: u32 = 16000;
pub const CHUNK_DURATION_SECS: u32 = 5;
pub const SILENCE_THRESHOLD: f32 = 0.1;

// ── Tipos de fuente de audio ──────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq)]
pub enum SourceType {
    Input,
    Output,
}

// ── Dispositivos ──────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq)]
pub struct DeviceInfo {
    pub id: usize,
    pub name: String,
    pub technical_name: Option<String>,
}

// ── Perfil de interlocutor ────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq)]
pub struct InterlocutorProfile {
    pub id: usize,
    pub device_id: usize,
    pub source_type: SourceType,
    pub name: String,
    pub is_active: bool,
    pub technical_name: Option<String>,
}

// ── Configuración de idioma ───────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq)]
pub struct LanguageConfig {
    pub source_lang: Option<&'static str>,
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

// ── Configuración de diarización ──────────────────────────────────────────

#[derive(Clone, Debug, PartialEq)]
pub enum DiarizeMode {
    /// Auto-detección: clustering con umbral de similitud coseno.
    Auto,
    /// Manual: el usuario indica cuántos hablantes hay.
    Manual,
}

#[derive(Clone, Debug)]
pub struct DiarizeConfig {
    pub enabled: bool,
    pub mode: DiarizeMode,
    /// Nº de hablantes (solo en modo Manual).
    pub num_speakers: usize,
    /// Umbral de similitud coseno (solo en modo Auto, 0.0–1.0).
    pub threshold: f32,
}

impl Default for DiarizeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: DiarizeMode::Auto,
            num_speakers: 2,
            threshold: crate::diarize::DEFAULT_COSINE_THRESHOLD,
        }
    }
}

// ── Información de hablante (editable por el usuario) ─────────────────────

#[derive(Clone, Debug)]
pub struct SpeakerInfo {
    pub id: usize,
    /// Etiqueta editable ("Speaker 0" por defecto, renombrable).
    pub label: String,
    /// Color RGB.
    pub color: (u8, u8, u8),
}

/// Paleta de colores distinguibles para hablantes.
pub const SPEAKER_COLORS: [(u8, u8, u8); 8] = [
    ( 66, 133, 244), // Azul
    (234,  67,  53), // Rojo
    ( 52, 168,  83), // Verde
    (251, 188,   4), // Ámbar
    (171,  71, 188), // Púrpura
    (255, 112,  67), // Naranja
    (  0, 172, 193), // Cian
    (124, 179,  66), // Verde claro
];

impl SpeakerInfo {
    pub fn new(id: usize) -> Self {
        let color = SPEAKER_COLORS[id % SPEAKER_COLORS.len()];
        Self {
            id,
            label: format!("Speaker {}", id + 1),
            color,
        }
    }
}

// ── Segmento de transcripción (con speaker) ──────────────────────────────

#[derive(Clone, Debug)]
pub struct TranscriptSegment {
    /// Marca de tiempo formateada ("00:15", "01:23:45").
    pub timestamp: String,
    /// Tiempo de inicio en segundos (para timeline).
    pub time_secs: f64,
    /// Duración estimada en segundos.
    pub duration_secs: f64,
    /// Texto transcrito.
    pub text: String,
    /// ID de hablante (None si la diarización está desactivada).
    pub speaker_id: Option<usize>,
}

// ── Entrada de timeline de diarización ────────────────────────────────────

#[derive(Clone, Debug)]
pub struct TimelineEntry {
    pub start_secs: f64,
    pub end_secs: f64,
    pub speaker_id: usize,
}

// ── Mensajes de comunicación audio → UI ───────────────────────────────────

pub enum AudioMessage {
    Status(String),
    Transcription {
        text: String,
        name: String,
        speaker_id: Option<usize>,
    },
    Error(String),
}

// ── Mensajes de comunicación vídeo → UI ───────────────────────────────────

pub enum VideoMessage {
    Status(String),
    Progress(f32),
    /// Segmento transcrito con speaker asignado.
    Segment {
        idx: usize,
        timestamp: String,
        time_secs: f64,
        duration_secs: f64,
        text: String,
        speaker_id: Option<usize>,
    },
    /// Timeline de diarización (para la visualización).
    Timeline {
        entries: Vec<TimelineEntry>,
        num_speakers: usize,
        total_duration: f64,
    },
    Done,
    Error(String),
}

// ── Navegación ────────────────────────────────────────────────────────────

#[derive(Debug, PartialEq, Eq)]
pub enum View {
    Transcription,
    Video,
    Settings,
}

// ── Alias ─────────────────────────────────────────────────────────────────

pub type UiSender = Sender<AudioMessage>;