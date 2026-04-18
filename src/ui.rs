use anyhow::{Result, anyhow};
use cpal::default_host;
use eframe::egui;
use std::sync::mpsc::{Receiver, channel};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::path::{Path, PathBuf};
use std::thread;
use chrono::Local;
use crate::data::{
    AudioMessage, DeviceInfo, DiarizeConfig, DiarizeMode, InterlocutorProfile,
    LanguageConfig, SourceType, SpeakerInfo, TimelineEntry, TranscriptSegment,
    View, VideoMessage, SPEAKER_COLORS, SOURCE_LANGUAGES,
};
use crate::audio::{audio_thread_main, get_available_devices};
use crate::video::video_transcription_thread;
use crate::system_audio::{check_loopback_status, get_loopback_devices, LoopbackStatus, LoopbackInfo};

pub struct TranscriptorApp {
    // ── Navegación ─────────────────────────────────────────────────────────
    pub current_view: View,

    // ── Transcripción en tiempo real ───────────────────────────────────────
    pub transcription: String,
    pub status_message: String,
    pub model_name: String,
    pub is_running: bool,
    pub all_input_devices: Vec<DeviceInfo>,
    pub all_output_devices: Vec<DeviceInfo>,
    pub interlocutors: Vec<InterlocutorProfile>,
    pub output_dir: String,
    pub ui_rx: Option<Receiver<AudioMessage>>,
    pub stop_signal: Option<Arc<AtomicBool>>,

    // ── Configuración de idioma (global) ───────────────────────────────────
    pub lang_config: LanguageConfig,

    // ── Loopback ───────────────────────────────────────────────────────────
    pub loopback_info: Option<LoopbackInfo>,
    pub show_loopback_setup: bool,

    // ── Diarización ────────────────────────────────────────────────────────
    pub diarize_config: DiarizeConfig,
    /// Speakers detectados en tiempo real (para la pestaña Transcripción).
    pub rt_speakers: Vec<SpeakerInfo>,

    // ── Transcripción de vídeo ─────────────────────────────────────────────
    pub video_file_path: Option<String>,
    pub video_segments: Vec<TranscriptSegment>,
    pub video_status: String,
    pub video_progress: f32,
    pub video_is_running: bool,
    pub video_rx: Option<Receiver<VideoMessage>>,
    pub video_stop_signal: Option<Arc<AtomicBool>>,
    /// Speakers detectados en vídeo.
    pub video_speakers: Vec<SpeakerInfo>,
    /// Timeline de diarización para la visualización.
    pub video_timeline: Vec<TimelineEntry>,
    pub video_total_duration: f64,
    /// Aviso de diarización (persiste aunque Whisper sobreescriba video_status).
    pub video_diarize_warning: String,
}

impl Default for TranscriptorApp {
    fn default() -> Self {
        let host = default_host();
        let all_input_devices = get_available_devices(&host, true);
        let all_output_devices = get_loopback_devices();

        let mut app = Self {
            current_view: View::Transcription,
            transcription: String::from("El texto transcrito aparecerá aquí.\n"),
            status_message: String::from("Presiona 'Iniciar Captura' para comenzar."),
            model_name: String::from("large-v3"),
            is_running: false,
            all_input_devices,
            all_output_devices,
            interlocutors: Vec::new(),
            output_dir: String::from("./minutas"),
            ui_rx: None,
            stop_signal: None,
            lang_config: LanguageConfig::default(),
            loopback_info: None,
            show_loopback_setup: false,
            diarize_config: DiarizeConfig::default(),
            rt_speakers: Vec::new(),
            video_file_path: None,
            video_segments: Vec::new(),
            video_status: String::from("Selecciona un archivo de vídeo o audio."),
            video_progress: 0.0,
            video_is_running: false,
            video_rx: None,
            video_stop_signal: None,
            video_speakers: Vec::new(),
            video_timeline: Vec::new(),
            video_total_duration: 0.0,
            video_diarize_warning: String::new(),
        };

        if !app.all_input_devices.is_empty() {
            app.add_new_profile(SourceType::Input);
        }
        if app.all_output_devices.is_empty() {
            app.check_and_prompt_loopback();
        }

        app
    }
}

impl eframe::App for TranscriptorApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // ── Procesar mensajes de audio en tiempo real ──────────────────────
        // Recopilar mensajes primero para evitar conflicto de borrows
        let audio_msgs: Vec<AudioMessage> = self.ui_rx.as_ref()
            .map(|rx| std::iter::from_fn(|| rx.try_recv().ok()).collect())
            .unwrap_or_default();

        for msg in audio_msgs {
            match msg {
                AudioMessage::Status(s) => self.status_message = s,
                AudioMessage::Transcription { text, name, speaker_id } => {
                    if !text.trim().is_empty() {
                        let label = if let Some(sid) = speaker_id {
                            self.ensure_rt_speaker(sid);
                            let info = &self.rt_speakers[sid];
                            format!("{}/{}", name, info.label)
                        } else {
                            name.clone()
                        };
                        self.transcription.push_str(&format!("({}) {}\n", label, text));
                    }
                }
                AudioMessage::Error(e) => self.status_message = format!("❌ Error: {}", e),
            }
        }

        // ── Procesar mensajes de vídeo ─────────────────────────────────────
        let video_msgs: Vec<VideoMessage> = self.video_rx.as_ref()
            .map(|rx| std::iter::from_fn(|| rx.try_recv().ok()).collect())
            .unwrap_or_default();

        for msg in video_msgs {
            match msg {
                VideoMessage::Status(s) => self.video_status = s,
                VideoMessage::Progress(p) => self.video_progress = p,
                VideoMessage::Segment { timestamp, time_secs, duration_secs, text, speaker_id } => {
                    if let Some(sid) = speaker_id {
                        self.ensure_video_speaker(sid);
                    }
                    self.video_segments.push(TranscriptSegment {
                        timestamp,
                        time_secs,
                        duration_secs,
                        text,
                        speaker_id,
                    });
                }
                VideoMessage::Timeline { entries, num_speakers, total_duration } => {
                    self.video_timeline = entries;
                    self.video_total_duration = total_duration;
                    for i in 0..num_speakers {
                        self.ensure_video_speaker(i);
                    }
                }
                VideoMessage::Done => {
                    if self.video_is_running {
                        self.video_is_running = false;
                        self.video_status = "✅ Transcripción completada.".into();
                        if let Err(e) = self.save_video_transcript() {
                            self.video_status = format!("❌ Error al guardar: {:?}", e);
                        }
                    }
                }
                VideoMessage::Error(e) => {
                    self.video_is_running = false;
                    self.video_status = format!("❌ Error: {}", e);
                }
                VideoMessage::DiarizeWarning(w) => {
                    self.video_diarize_warning = w;
                }
            }
        }

        // ── UI ─────────────────────────────────────────────────────────────
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.selectable_value(&mut self.current_view, View::Transcription, "🎙 Transcripción");
                ui.selectable_value(&mut self.current_view, View::Video, "🎬 Vídeo");
                ui.selectable_value(&mut self.current_view, View::Settings, "⚙️ Configuración");

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.add_space(10.0);
                    ui.label(format!("Modelo: ggml-{}.bin", self.model_name));
                    if self.diarize_config.enabled {
                        ui.colored_label(egui::Color32::from_rgb(66, 133, 244), "🔊 Diarización");
                    }
                });
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            match self.current_view {
                View::Transcription => self.transcriber_ui(ui),
                View::Video => self.video_ui(ui),
                View::Settings => self.settings_ui(ui),
            }
        });

        if self.show_loopback_setup {
            self.show_loopback_dialog(ctx);
        }

        ctx.request_repaint();
    }
}

impl TranscriptorApp {
    // ── Helpers de speakers ───────────────────────────────────────────────

    fn ensure_rt_speaker(&mut self, id: usize) {
        while self.rt_speakers.len() <= id {
            self.rt_speakers.push(SpeakerInfo::new(self.rt_speakers.len()));
        }
    }

    fn ensure_video_speaker(&mut self, id: usize) {
        while self.video_speakers.len() <= id {
            self.video_speakers.push(SpeakerInfo::new(self.video_speakers.len()));
        }
    }

    fn speaker_color_egui(color: (u8, u8, u8)) -> egui::Color32 {
        egui::Color32::from_rgb(color.0, color.1, color.2)
    }

    // ── Pestaña: Transcripción en tiempo real ──────────────────────────────

    fn check_and_prompt_loopback(&mut self) {
        if let Ok(info) = check_loopback_status() {
            if info.status == LoopbackStatus::NeedsConfiguration
                || info.status == LoopbackStatus::RequiresSetup
            {
                self.loopback_info = Some(info);
                self.show_loopback_setup = true;
            }
        }
    }

    fn start_audio_capture(&mut self) {
        let active: Vec<InterlocutorProfile> = self.interlocutors
            .iter().filter(|p| p.is_active).cloned().collect();

        if active.is_empty() {
            self.status_message = "❌ Debe añadir y activar al menos una fuente.".into();
            return;
        }

        let (tx, rx) = channel::<AudioMessage>();
        self.ui_rx = Some(rx);

        let stop = Arc::new(AtomicBool::new(false));
        self.stop_signal = Some(stop.clone());

        let model = self.model_name.clone();
        let n = active.len();
        let lang = self.lang_config.clone();
        let diarize = self.diarize_config.clone();

        self.rt_speakers.clear();

        thread::spawn(move || {
            if let Err(e) = audio_thread_main(model, tx.clone(), stop, active, lang, diarize) {
                let _ = tx.send(AudioMessage::Error(format!("{:?}", e)));
            }
        });

        self.is_running = true;
        self.transcription.clear();
        self.status_message = format!("Iniciando {} fuentes de audio...", n);
    }

    fn transcriber_ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("🎙️ Transcripción en Tiempo Real");
        ui.separator();

        ui.horizontal(|ui| {
            ui.label("Modelo Whisper:");
            egui::ComboBox::from_label("")
                .selected_text(&self.model_name)
                .width(150.0)
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.model_name, "medium".into(), "Medium");
                    ui.selectable_value(&mut self.model_name, "large-v3".into(), "Large-v3");
                });
        });

        ui.add_space(10.0);

        let btn = if self.is_running { "⏹ Detener Captura" } else { "▶ Iniciar Captura" };
        let enabled = !self.is_running && !self.interlocutors.is_empty() || self.is_running;

        if ui.add_enabled(enabled, egui::Button::new(btn)).clicked() {
            if self.is_running {
                if let Some(sig) = self.stop_signal.take() {
                    sig.store(true, Ordering::SeqCst);
                }
                self.is_running = false;
                match self.save_transcript() {
                    Ok(path) => {
                        self.status_message = format!(
                            "✅ Captura detenida. Minuta guardada en: {}", path.display()
                        );
                    }
                    Err(e) => {
                        self.status_message = format!(
                            "⚠️ Captura detenida. Error al guardar minuta: {:?}", e
                        );
                    }
                }
            } else if self.interlocutors.iter().any(|p| p.is_active) {
                self.start_audio_capture();
            } else {
                self.status_message = "❌ Active al menos un interlocutor en Configuración.".into();
            }
        }

        ui.separator();

        ui.horizontal(|ui| {
            ui.label("Estado:");
            ui.colored_label(
                if self.is_running { egui::Color32::GREEN } else { egui::Color32::GRAY },
                &self.status_message,
            );
        });

        // Speaker legend (real-time)
        if !self.rt_speakers.is_empty() {
            ui.add_space(4.0);
            ui.horizontal_wrapped(|ui| {
                ui.label("Hablantes:");
                for sp in &self.rt_speakers {
                    let color = Self::speaker_color_egui(sp.color);
                    ui.colored_label(color, format!("● {}", sp.label));
                }
            });
        }

        ui.add_space(10.0);
        ui.label("📝 Minuta (Interlocutor) Texto:");

        egui::ScrollArea::vertical()
            .max_height(400.0)
            .stick_to_bottom(true)
            .show(ui, |ui| {
                ui.add(
                    egui::TextEdit::multiline(&mut self.transcription)
                        .desired_width(f32::INFINITY)
                        .font(egui::TextStyle::Monospace)
                        .interactive(false),
                );
            });

        if ui.button("🗑️ Limpiar").clicked() {
            self.transcription.clear();
            self.rt_speakers.clear();
        }
    }

    // ── Pestaña: Transcripción de vídeo ────────────────────────────────────

    fn video_ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("🎬 Transcripción de Vídeo / Audio");
        ui.separator();

        // Selector de archivo
        ui.horizontal(|ui| {
            ui.add_enabled_ui(!self.video_is_running, |ui| {
                if ui.button("📂 Seleccionar archivo").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter(
                            "Vídeo / Audio",
                            &["mp4", "mkv", "avi", "mov", "webm", "mp3", "wav", "flac", "ogg", "m4a"],
                        )
                        .pick_file()
                    {
                        self.video_file_path = Some(path.to_string_lossy().to_string());
                        self.video_segments.clear();
                        self.video_speakers.clear();
                        self.video_timeline.clear();
                        self.video_progress = 0.0;
                        self.video_total_duration = 0.0;
                        self.video_status = "Archivo seleccionado. Listo para transcribir.".into();
                    }
                }
            });

            match &self.video_file_path {
                Some(p) => {
                    let name = Path::new(p)
                        .file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_else(|| p.clone());
                    ui.label(egui::RichText::new(&name).strong());
                }
                None => { ui.label(egui::RichText::new("Sin archivo seleccionado").weak()); }
            }
        });

        ui.add_space(6.0);

        // Modelo + botón de inicio/parada
        ui.horizontal(|ui| {
            ui.label("Modelo:");
            ui.add_enabled_ui(!self.video_is_running, |ui| {
                egui::ComboBox::from_id_salt("video_model")
                    .selected_text(&self.model_name)
                    .width(150.0)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.model_name, "medium".into(), "Medium");
                        ui.selectable_value(&mut self.model_name, "large-v3".into(), "Large-v3");
                    });
            });

            ui.add_space(10.0);

            let can_start = self.video_file_path.is_some() && !self.video_is_running;

            if self.video_is_running {
                if ui.button("⏹ Cancelar").clicked() {
                    if let Some(sig) = self.video_stop_signal.take() {
                        sig.store(true, Ordering::SeqCst);
                    }
                    // Actualizar estado inmediatamente en la UI, sin esperar al hilo
                    self.video_is_running = false;
                    self.video_progress = 0.0;
                    self.video_status = "⛔ Cancelado por el usuario.".into();
                }
            } else if ui.add_enabled(can_start, egui::Button::new("▶ Transcribir")).clicked() {
                self.start_video_transcription();
            }
        });

        ui.add_space(6.0);

        // Barra de progreso
        if self.video_is_running || self.video_progress > 0.0 {
            let bar = egui::ProgressBar::new(self.video_progress)
                .show_percentage()
                .animate(self.video_is_running);
            ui.add(bar);
        }

        // ── Timeline de diarización ───────────────────────────────────────
        if !self.video_timeline.is_empty() {
            ui.add_space(4.0);
            self.draw_speaker_timeline(ui);
        }

        // Estado
        ui.horizontal(|ui| {
            ui.label("Estado:");
            ui.colored_label(
                if self.video_is_running { egui::Color32::GREEN } else { egui::Color32::GRAY },
                &self.video_status,
            );
        });

        // Estado de diarización (persiste aunque Whisper sobreescriba el status principal)
        if !self.video_diarize_warning.is_empty() {
            let color = if self.video_diarize_warning.starts_with('✅') {
                egui::Color32::from_rgb(52, 168, 83)  // verde
            } else {
                egui::Color32::YELLOW
            };
            ui.colored_label(color, &self.video_diarize_warning);
        }

        // ── Speaker labels editables ──────────────────────────────────────
        if !self.video_speakers.is_empty() && !self.video_is_running {
            ui.add_space(6.0);
            ui.horizontal_wrapped(|ui| {
                ui.label(egui::RichText::new("Hablantes:").strong());
                for sp in self.video_speakers.iter_mut() {
                    let color = Self::speaker_color_egui(sp.color);
                    ui.colored_label(color, "●");
                    ui.add(
                        egui::TextEdit::singleline(&mut sp.label)
                            .desired_width(100.0)
                            .hint_text(format!("Speaker {}", sp.id + 1)),
                    );
                    ui.add_space(8.0);
                }
            });
        }

        ui.separator();

        // Transcripción
        ui.label("📝 Transcripción [HH:MM:SS] (Speaker) texto:");

        egui::ScrollArea::vertical()
            .max_height(320.0)
            .stick_to_bottom(true)
            .show(ui, |ui| {
                for seg in &self.video_segments {
                    ui.horizontal_wrapped(|ui| {
                        ui.label(
                            egui::RichText::new(format!("[{}]", seg.timestamp))
                                .monospace()
                                .color(egui::Color32::GRAY),
                        );

                        if let Some(sid) = seg.speaker_id {
                            if sid < self.video_speakers.len() {
                                let sp = &self.video_speakers[sid];
                                let color = Self::speaker_color_egui(sp.color);
                                ui.colored_label(color, format!("({})", sp.label));
                            }
                        }

                        ui.label(&seg.text);
                    });
                }
            });

        ui.horizontal(|ui| {
            if ui.button("🗑️ Limpiar").clicked() {
                self.video_segments.clear();
                self.video_speakers.clear();
                self.video_timeline.clear();
                self.video_progress = 0.0;
                self.video_total_duration = 0.0;
            }
            if !self.video_segments.is_empty() && !self.video_is_running {
                if ui.button("💾 Guardar").clicked() {
                    match self.save_video_transcript() {
                        Ok(p) => self.video_status = format!("✅ Guardado en: {}", p.display()),
                        Err(e) => self.video_status = format!("❌ Error al guardar: {:?}", e),
                    }
                }
            }
        });
    }

    /// Dibuja la timeline visual de diarización (barra horizontal con colores por speaker).
    fn draw_speaker_timeline(&self, ui: &mut egui::Ui) {
        let available_width = ui.available_width();
        let height = 22.0;

        let (rect, _response) = ui.allocate_exact_size(
            egui::vec2(available_width, height),
            egui::Sense::hover(),
        );

        if self.video_total_duration <= 0.0 { return; }

        let painter = ui.painter_at(rect);

        // Fondo
        painter.rect_filled(rect, 3.0, egui::Color32::from_gray(50));

        // Segmentos coloreados por speaker
        for entry in &self.video_timeline {
            let x_start = rect.left()
                + (entry.start_secs / self.video_total_duration) as f32 * rect.width();
            let x_end = rect.left()
                + (entry.end_secs / self.video_total_duration) as f32 * rect.width();

            let color = if entry.speaker_id < self.video_speakers.len() {
                let c = self.video_speakers[entry.speaker_id].color;
                egui::Color32::from_rgba_premultiplied(c.0, c.1, c.2, 200)
            } else {
                let c = SPEAKER_COLORS[entry.speaker_id % SPEAKER_COLORS.len()];
                egui::Color32::from_rgba_premultiplied(c.0, c.1, c.2, 200)
            };

            let seg_rect = egui::Rect::from_min_max(
                egui::pos2(x_start, rect.top() + 1.0),
                egui::pos2(x_end, rect.bottom() - 1.0),
            );
            painter.rect_filled(seg_rect, 1.0, color);
        }

        // Borde
        painter.rect_stroke(rect, 3.0, egui::Stroke::new(1.0, egui::Color32::from_gray(80)), egui::StrokeKind::Outside);
    }

    fn start_video_transcription(&mut self) {
        let file_path = match &self.video_file_path {
            Some(p) => p.clone(),
            None => return,
        };

        let (tx, rx) = channel::<VideoMessage>();
        self.video_rx = Some(rx);

        let stop = Arc::new(AtomicBool::new(false));
        self.video_stop_signal = Some(stop.clone());

        let model = self.model_name.clone();
        let lang = self.lang_config.clone();
        let diarize = self.diarize_config.clone();

        thread::spawn(move || {
            if let Err(e) = video_transcription_thread(file_path, model, lang, diarize, tx.clone(), stop) {
                let _ = tx.send(VideoMessage::Error(format!("{:?}", e)));
            }
        });

        self.video_is_running = true;
        self.video_segments.clear();
        self.video_speakers.clear();
        self.video_timeline.clear();
        self.video_progress = 0.0;
        self.video_total_duration = 0.0;
        self.video_diarize_warning.clear();
        self.video_status = "Iniciando...".into();
    }

    fn save_video_transcript(&self) -> Result<PathBuf> {
        if self.video_segments.is_empty() {
            return Err(anyhow!("No hay transcripción para guardar."));
        }

        let stem = self.video_file_path
            .as_deref()
            .and_then(|p| Path::new(p).file_stem())
            .map(|s| s.to_string_lossy().replace(' ', "_"))
            .unwrap_or_else(|| "video".into());

        let timestamp = Local::now().format("%Y%m%d_%H%M%S").to_string();
        let filename = format!("{}_{}.md", stem, timestamp);
        let output_path = Path::new(&self.output_dir).join(filename);

        std::fs::create_dir_all(&self.output_dir)?;

        let mut content = format!(
            "# Transcripción: {}\n\nFecha: {}\n",
            stem,
            Local::now().format("%d-%m-%Y %H:%M:%S"),
        );

        // Leyenda de speakers
        if !self.video_speakers.is_empty() {
            content.push_str("\n## Hablantes\n\n");
            for sp in &self.video_speakers {
                content.push_str(&format!("- **{}** (Speaker {})\n", sp.label, sp.id + 1));
            }
        }

        content.push_str("\n---\n\n");

        for seg in &self.video_segments {
            let speaker_label = seg.speaker_id
                .and_then(|sid| self.video_speakers.get(sid))
                .map(|sp| format!(" ({})", sp.label))
                .unwrap_or_default();

            content.push_str(&format!("[{}]{} {}\n", seg.timestamp, speaker_label, seg.text));
        }

        std::fs::write(&output_path, content)?;
        Ok(output_path)
    }

    // ── Pestaña: Configuración ─────────────────────────────────────────────

    fn settings_ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("⚙️ Configuración de Interlocutores y Audio");
        ui.separator();

        // ── Idioma ────────────────────────────────────────────────────────
        ui.label(egui::RichText::new("🌐 Idioma").strong());
        ui.add_space(4.0);

        ui.add_enabled_ui(!self.is_running, |ui| {
            ui.horizontal(|ui| {
                ui.label("Idioma original:");
                egui::ComboBox::from_id_salt("lang_source")
                    .selected_text(self.lang_config.source_label())
                    .width(160.0)
                    .show_ui(ui, |ui| {
                        for (label, code) in SOURCE_LANGUAGES {
                            ui.selectable_value(&mut self.lang_config.source_lang, *code, *label);
                        }
                    });

                ui.add_space(16.0);

                ui.label("Idioma destino:");
                egui::ComboBox::from_id_salt("lang_dest")
                    .selected_text(self.lang_config.dest_label())
                    .width(200.0)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.lang_config.translate_to_english,
                            false,
                            "Original (sin traducción)",
                        );
                        ui.selectable_value(
                            &mut self.lang_config.translate_to_english,
                            true,
                            "English (traducir)",
                        );
                    });
            });

            ui.label(
                egui::RichText::new("ℹ Whisper solo puede traducir al inglés de forma nativa.")
                    .small()
                    .color(egui::Color32::GRAY),
            );
        });

        ui.add_space(10.0);
        ui.separator();

        // ── Diarización ───────────────────────────────────────────────────
        ui.label(egui::RichText::new("🔊 Diarización de Hablantes").strong());
        ui.add_space(4.0);

        ui.add_enabled_ui(!self.is_running && !self.video_is_running, |ui| {
            ui.checkbox(&mut self.diarize_config.enabled, "Activar diferenciación de voces");

            if self.diarize_config.enabled {
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.label("Modo:");
                    ui.radio_value(&mut self.diarize_config.mode, DiarizeMode::Auto, "Auto-detección");
                    ui.radio_value(&mut self.diarize_config.mode, DiarizeMode::Manual, "Nº manual");
                });

                match self.diarize_config.mode {
                    DiarizeMode::Auto => {
                        ui.horizontal(|ui| {
                            ui.label("Sensibilidad:");
                            ui.add(
                                egui::Slider::new(&mut self.diarize_config.threshold, 0.40..=0.80)
                                    .text("umbral coseno")
                                    .step_by(0.05),
                            );
                        });
                        ui.label(
                            egui::RichText::new(
                                "ℹ Más alto = más speakers detectados. Rango típico: 0.55–0.70."
                            )
                            .small()
                            .color(egui::Color32::GRAY),
                        );
                    }
                    DiarizeMode::Manual => {
                        ui.horizontal(|ui| {
                            ui.label("Nº de hablantes:");
                            ui.add(
                                egui::DragValue::new(&mut self.diarize_config.num_speakers)
                                    .range(2..=10)
                                    .speed(0.1),
                            );
                        });
                    }
                }

                ui.add_space(4.0);
                ui.label(
                    egui::RichText::new(
                        "Requiere el modelo ONNX de embeddings (~50 MB). Se descarga automáticamente\n\
                         en la primera ejecución, o puedes colocarlo manualmente en models/speaker_embedding.onnx"
                    )
                    .small()
                    .color(egui::Color32::GRAY),
                );
            }
        });

        ui.add_space(10.0);
        ui.separator();

        // ── Loopback ──────────────────────────────────────────────────────
        ui.horizontal(|ui| {
            if ui.button("📊 Configurar Captura de Salida").clicked() {
                self.loopback_info = check_loopback_status().ok();
                self.show_loopback_setup = true;
            }
            let n = self.all_output_devices.len();
            if n > 0 {
                ui.colored_label(egui::Color32::GREEN, format!("✅ {} dispositivos loopback", n));
            } else {
                ui.colored_label(egui::Color32::YELLOW, "⚠️ Sin dispositivos de salida");
            }
        });

        ui.add_space(10.0);

        // ── Interlocutores ────────────────────────────────────────────────
        ui.add_enabled_ui(!self.is_running, |ui| {
            ui.label("Añadir nueva fuente de audio:");
            ui.horizontal(|ui| {
                if ui.button("➕ Entrada (Micrófono)").clicked() {
                    self.add_new_profile(SourceType::Input);
                }
                if ui.button("➕ Salida (Loopback)").clicked() {
                    if self.all_output_devices.is_empty() {
                        self.status_message = "⚠️ Configure dispositivos loopback primero".into();
                        self.loopback_info = check_loopback_status().ok();
                        self.show_loopback_setup = true;
                    } else {
                        self.add_new_profile(SourceType::Output);
                    }
                }
            });
        });

        ui.add_space(10.0);
        ui.label("Perfiles Activos:");

        let input_devices = &self.all_input_devices;
        let output_devices = &self.all_output_devices;
        let mut to_remove: Option<usize> = None;

        egui::ScrollArea::vertical().show(ui, |ui| {
            for (idx, profile) in self.interlocutors.iter_mut().enumerate() {
                ui.horizontal(|ui| {
                    ui.checkbox(&mut profile.is_active, "");

                    let (devices_to_show, icon) = match profile.source_type {
                        SourceType::Input => (input_devices, "🎤"),
                        SourceType::Output => (output_devices, "📊"),
                    };
                    ui.label(icon);

                    let device_name = Self::get_device_name_static(
                        input_devices,
                        output_devices,
                        profile.source_type.clone(),
                        profile.device_id,
                    );

                    egui::ComboBox::from_id_salt(profile.id)
                        .selected_text(device_name)
                        .width(220.0)
                        .show_ui(ui, |ui| {
                            for device in devices_to_show {
                                let r = ui.selectable_value(
                                    &mut profile.device_id,
                                    device.id,
                                    &device.name,
                                );
                                if r.clicked() {
                                    profile.technical_name = device.technical_name.clone();
                                }
                            }
                        });

                    ui.add(
                        egui::TextEdit::singleline(&mut profile.name)
                            .desired_width(130.0)
                            .hint_text(format!("Interlocutor {}", profile.id)),
                    );

                    if ui.button("🗑").clicked() {
                        to_remove = Some(idx);
                    }
                });
            }
        });

        if let Some(idx) = to_remove {
            self.remove_profile(idx);
        }

        if self.is_running {
            ui.label(
                egui::RichText::new("⚠️ Detenga la captura para cambiar la configuración.")
                    .color(egui::Color32::YELLOW),
            );
        }

        ui.separator();
        ui.label("Ruta de guardado de minutas (Markdown):");
        ui.add_enabled(
            !self.is_running,
            egui::TextEdit::singleline(&mut self.output_dir).desired_width(300.0),
        );
    }

    fn show_loopback_dialog(&mut self, ctx: &egui::Context) {
        let mut close = false;

        egui::Window::new("📊 Configuración de Captura de Audio de Salida")
            .collapsible(false)
            .resizable(true)
            .default_width(650.0)
            .show(ctx, |ui| {
                if let Some(info) = &self.loopback_info {
                    ui.label(egui::RichText::new(&info.message).size(16.0).strong());
                    ui.add_space(10.0);

                    match info.status {
                        LoopbackStatus::Available => {
                            ui.colored_label(egui::Color32::GREEN, "✅ Sistema configurado correctamente");
                            for dev in &info.loopback_devices {
                                ui.label(format!("  • {}", dev.name));
                            }
                        }
                        _ => {
                            ui.label("Instrucciones:");
                            ui.separator();
                            egui::ScrollArea::vertical().max_height(250.0).show(ui, |ui| {
                                for line in &info.instructions {
                                    if line.is_empty() {
                                        ui.add_space(5.0);
                                    } else {
                                        ui.label(line);
                                    }
                                }
                            });
                        }
                    }

                    ui.add_space(10.0);
                    ui.separator();
                    ui.horizontal(|ui| {
                        if ui.button("🔄 Actualizar Dispositivos").clicked() {
                            let host = default_host();
                            self.all_input_devices = get_available_devices(&host, true);
                            self.all_output_devices = get_loopback_devices();
                            let n = self.all_output_devices.len();
                            self.status_message = if n > 0 {
                                format!("✅ {} dispositivos loopback detectados", n)
                            } else {
                                "⚠️ No se detectaron dispositivos loopback".into()
                            };
                            if n > 0 { close = true; }
                        }
                        if ui.button("Cerrar").clicked() { close = true; }
                    });
                }
            });

        if close { self.show_loopback_setup = false; }
    }

    // ── Helpers ────────────────────────────────────────────────────────────

    fn add_new_profile(&mut self, source_type: SourceType) {
        let raw = match source_type {
            SourceType::Input => &self.all_input_devices,
            SourceType::Output => &self.all_output_devices,
        };
        let device_id = raw.first().map(|d| d.id).unwrap_or(0);
        let new_id = self.interlocutors.len();
        self.interlocutors.push(InterlocutorProfile {
            id: new_id,
            device_id,
            source_type,
            name: format!("Interlocutor {}", new_id),
            is_active: true,
            technical_name: raw.first().and_then(|d| d.technical_name.clone()),
        });
    }

    fn remove_profile(&mut self, index: usize) {
        if index < self.interlocutors.len() {
            self.interlocutors.remove(index);
            for (i, p) in self.interlocutors.iter_mut().enumerate() {
                p.id = i;
                p.name = format!("Interlocutor {}", i);
            }
        }
    }

    fn save_transcript(&self) -> Result<PathBuf> {
        if self.transcription.trim().is_empty() {
            return Err(anyhow!("No hay transcripción para guardar."));
        }
        let timestamp = Local::now().format("%Y%m%d_%H%M%S").to_string();
        let names: String = self.interlocutors.iter()
            .filter(|p| p.is_active)
            .map(|p| p.name.replace(' ', "_"))
            .collect::<Vec<_>>()
            .join("_");
        let output_path = Path::new(&self.output_dir).join(format!("{}_{}.md", names, timestamp));
        std::fs::create_dir_all(&self.output_dir)?;

        let mut content = format!(
            "# Minuta de Transcripción\n\nFecha: {}\n",
            Local::now().format("%d-%m-%Y %H:%M:%S"),
        );

        // Leyenda de speakers (real-time)
        if !self.rt_speakers.is_empty() {
            content.push_str("\n## Hablantes detectados\n\n");
            for sp in &self.rt_speakers {
                content.push_str(&format!("- **{}**\n", sp.label));
            }
        }

        content.push_str(&format!("\n---\n\n{}", self.transcription));

        std::fs::write(&output_path, content)?;
        Ok(output_path)
    }

    fn get_device_name_static(
        inputs: &[DeviceInfo], outputs: &[DeviceInfo],
        source_type: SourceType, device_id: usize,
    ) -> String {
        let devices = match source_type {
            SourceType::Input => inputs,
            SourceType::Output => outputs,
        };
        devices.iter()
            .find(|d| d.id == device_id)
            .map(|d| d.name.clone())
            .unwrap_or_else(|| "Dispositivo no encontrado".into())
    }
}