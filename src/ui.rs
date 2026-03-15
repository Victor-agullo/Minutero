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
    AudioMessage, DeviceInfo, InterlocutorProfile, LanguageConfig, SourceType, View,
    SOURCE_LANGUAGES,
};
use crate::audio::{audio_thread_main, get_available_devices};
use crate::system_audio::{check_loopback_status, get_loopback_devices, LoopbackStatus, LoopbackInfo};

pub struct TranscriptorApp {
    pub current_view: View, 
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
    
    pub loopback_info: Option<LoopbackInfo>,
    pub show_loopback_setup: bool,

    /// Configuración de idioma global para la sesión
    pub lang_config: LanguageConfig,
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
            loopback_info: None,
            show_loopback_setup: false,
            lang_config: LanguageConfig::default(), // English → sin traducción
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
        if let Some(rx) = &self.ui_rx {
            while let Ok(msg) = rx.try_recv() {
                match msg {
                    AudioMessage::Status(status) => {
                        self.status_message = status;
                    }
                    AudioMessage::Transcription { text, name } => {
                        if !text.trim().is_empty() {
                            self.transcription.push_str(&format!("({}) {}\n", name, text)); 
                        }
                    }
                    AudioMessage::Error(err) => {
                        self.status_message = format!("❌ Error: {}", err);
                    }
                }
            }
        }
        
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.selectable_value(&mut self.current_view, View::Transcription, "🎙 Transcripción");
                ui.selectable_value(&mut self.current_view, View::Settings, "⚙️ Configuración");
                
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.add_space(10.0);
                    ui.label(format!("Modelo: ggml-{}.bin", self.model_name));
                });
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            match self.current_view {
                View::Transcription => self.transcriber_ui(ui),
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
    fn check_and_prompt_loopback(&mut self) {
        if let Ok(info) = check_loopback_status() {
            if info.status == LoopbackStatus::NeedsConfiguration || 
               info.status == LoopbackStatus::RequiresSetup {
                self.loopback_info = Some(info);
                self.show_loopback_setup = true;
            }
        }
    }

    fn start_audio_capture(&mut self) {
        let active_interlocutors: Vec<InterlocutorProfile> = self.interlocutors.iter()
            .filter(|p| p.is_active)
            .cloned()
            .collect();
            
        if active_interlocutors.is_empty() {
            self.status_message = "❌ Error: Debe añadir y activar al menos una fuente de audio.".to_string();
            return;
        }

        let (tx, rx) = channel::<AudioMessage>();
        self.ui_rx = Some(rx);
        
        let stop_signal = Arc::new(AtomicBool::new(false)); 
        self.stop_signal = Some(stop_signal.clone());
        
        let model_name = self.model_name.clone();
        let num_active = active_interlocutors.len();
        let lang_config = self.lang_config.clone();
        
        thread::spawn(move || {
            if let Err(e) = audio_thread_main(model_name, tx.clone(), stop_signal, active_interlocutors, lang_config) {
                let _ = tx.send(AudioMessage::Error(format!("{:?}", e)));
            }
        });
        
        self.is_running = true;
        self.transcription.clear();
        self.status_message = format!("Iniciando {} fuentes de audio...", num_active);
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
                    ui.selectable_value(&mut self.model_name, "medium".to_string(), "Medium");
                    ui.selectable_value(&mut self.model_name, "large-v3".to_string(), "Large-v3 (Turbo)");
                });
        });

        ui.add_space(10.0);

        let button_text = if self.is_running { "⏹ Detener Captura" } else { "▶ Iniciar Captura" };
        let can_run = !self.is_running && !self.interlocutors.is_empty();

        if ui.add_enabled(can_run || self.is_running, egui::Button::new(button_text)).clicked() {
            if self.is_running {
                if let Some(signal) = self.stop_signal.take() {
                    signal.store(true, Ordering::SeqCst);
                }
                
                self.is_running = false;
                self.status_message = "Captura detenida. Guardando minuta...".to_string();
                
                match self.save_transcript() {
                    Ok(path) => {
                        self.status_message = format!("✅ Minuta guardada en: {}", path.display());
                    }
                    Err(e) => {
                        self.status_message = format!("❌ Error al guardar minuta: {:?}", e);
                    }
                }
            } else {
                if !self.interlocutors.iter().any(|p| p.is_active) {
                     self.status_message = "❌ Error: Active al menos un interlocutor en Configuración.".to_string();
                } else {
                    self.start_audio_capture();
                }
            }
        }
        
        ui.separator();
        
        ui.horizontal(|ui| {
            ui.label("Estado:");
            ui.colored_label(
                if self.is_running { egui::Color32::GREEN } else { egui::Color32::GRAY },
                &self.status_message
            );
        });
        
        ui.add_space(10.0);
        ui.label("📝 Minuta (Formato: (Interlocutor) Texto):");
        
        egui::ScrollArea::vertical()
            .max_height(400.0)
            .stick_to_bottom(true)
            .show(ui, |ui| {
                ui.add(
                    egui::TextEdit::multiline(&mut self.transcription)
                        .desired_width(f32::INFINITY)
                        .font(egui::TextStyle::Monospace)
                        .interactive(false)
                );
            });
        
        if ui.button("🗑️ Limpiar Transcripciones").clicked() {
            self.transcription.clear();
        }
    }

    fn settings_ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("⚙️ Configuración de Interlocutores y Audio");
        ui.separator();

        // ── Sección de idioma ──────────────────────────────────────────────
        ui.label(egui::RichText::new("🌐 Idioma").strong());
        ui.add_space(4.0);

        ui.add_enabled_ui(!self.is_running, |ui| {
            ui.horizontal(|ui| {
                // Idioma original
                ui.label("Idioma original:");
                egui::ComboBox::from_id_salt("lang_source")
                    .selected_text(self.lang_config.source_label())
                    .width(160.0)
                    .show_ui(ui, |ui| {
                        for (label, code) in SOURCE_LANGUAGES {
                            ui.selectable_value(
                                &mut self.lang_config.source_lang,
                                *code,
                                *label,
                            );
                        }
                    });

                ui.add_space(16.0);

                // Idioma destino
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

            // Nota informativa
            ui.add_space(2.0);
            ui.label(
                egui::RichText::new(
                    "ℹ Whisper solo puede traducir al inglés de forma nativa. \
                     Para otros destinos se requeriría una API externa."
                )
                .small()
                .color(egui::Color32::GRAY),
            );
        });

        ui.add_space(10.0);
        ui.separator();

        // ── Loopback ───────────────────────────────────────────────────────
        ui.horizontal(|ui| {
            if ui.button("📊 Configurar Captura de Salida").clicked() {
                self.loopback_info = check_loopback_status().ok();
                self.show_loopback_setup = true;
            }
            
            let loopback_count = self.all_output_devices.len();
            if loopback_count > 0 {
                ui.colored_label(egui::Color32::GREEN, format!("✅ {} dispositivos loopback", loopback_count));
            } else {
                ui.colored_label(egui::Color32::YELLOW, "⚠️ Sin dispositivos de salida");
            }
        });
        
        ui.add_space(10.0);
        
        // ── Interlocutores ─────────────────────────────────────────────────
        ui.add_enabled_ui(!self.is_running, |ui| {
            ui.label("Añadir nueva fuente de audio:");
            ui.horizontal(|ui| {
                if ui.button("➕ Entrada (Micrófono)").clicked() {
                    self.add_new_profile(SourceType::Input);
                }
                if ui.button("➕ Salida (Loopback)").clicked() {
                    if self.all_output_devices.is_empty() {
                        self.status_message = "⚠️ Configure dispositivos loopback primero".to_string();
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

        let mut profile_to_remove: Option<usize> = None;

        egui::ScrollArea::vertical().show(ui, |ui| {
            for (idx, profile) in self.interlocutors.iter_mut().enumerate() { 
                ui.horizontal(|ui| {
                    ui.checkbox(&mut profile.is_active, "");
                    
                    let (devices_to_show, label) = match profile.source_type {
                        SourceType::Input => (input_devices, "🎤"),
                        SourceType::Output => (output_devices, "📊"),
                    };
                    
                    ui.label(label);
                    
                    let device_name = Self::get_device_name_static(
                        input_devices, 
                        output_devices, 
                        profile.source_type.clone(), 
                        profile.device_id
                    );

                    egui::ComboBox::from_id_salt(profile.id)
                        .selected_text(device_name)
                        .width(220.0)
                        .show_ui(ui, |ui| {
                            for device in devices_to_show {
                                let is_selected = ui.selectable_value(
                                    &mut profile.device_id, 
                                    device.id, 
                                    &device.name
                                );
                                if is_selected.clicked() {
                                    profile.technical_name = device.technical_name.clone();
                                }
                            }
                        });

                    ui.add(
                        egui::TextEdit::singleline(&mut profile.name)
                            .desired_width(130.0)
                            .hint_text(format!("Interlocutor {}", profile.id))
                    );

                    if ui.button("🗑").clicked() {
                        profile_to_remove = Some(idx);
                    }
                });
            }
        });

        if let Some(idx) = profile_to_remove {
            self.remove_profile(idx);
        }

        if self.is_running {
            ui.label(egui::RichText::new("⚠️ Detenga la captura para cambiar la configuración.").color(egui::Color32::YELLOW));
        }

        ui.separator();

        ui.label("Ruta de guardado de minutas (Markdown):");
        ui.horizontal(|ui| {
            ui.add_enabled(!self.is_running, egui::TextEdit::singleline(&mut self.output_dir).desired_width(300.0));
        });
    }

    fn show_loopback_dialog(&mut self, ctx: &egui::Context) {
        let mut close_dialog = false;
        
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
                            ui.add_space(5.0);
                            
                            if !info.loopback_devices.is_empty() {
                                ui.label("Dispositivos loopback detectados:");
                                for dev in &info.loopback_devices {
                                    ui.label(format!("  • {}", dev.name));
                                }
                            }
                        }
                        _ => {
                            ui.label("Instrucciones:");
                            ui.separator();
                            
                            egui::ScrollArea::vertical()
                                .max_height(250.0)
                                .show(ui, |ui| {
                                    for instruction in &info.instructions {
                                        if instruction.is_empty() {
                                            ui.add_space(5.0);
                                        } else {
                                            ui.label(instruction);
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
                            
                            let count = self.all_output_devices.len();
                            self.status_message = if count > 0 {
                                format!("✅ {} dispositivos loopback detectados", count)
                            } else {
                                "⚠️ No se detectaron dispositivos loopback".to_string()
                            };
                            
                            if count > 0 {
                                close_dialog = true;
                            }
                        }
                        
                        if ui.button("Cerrar").clicked() {
                            close_dialog = true;
                        }
                    });
                }
            });
        
        if close_dialog {
            self.show_loopback_setup = false;
        }
    }

    fn add_new_profile(&mut self, source_type: SourceType) {
        let raw_devices = match source_type {
            SourceType::Input => &self.all_input_devices,
            SourceType::Output => &self.all_output_devices,
        };
        
        let device_id = raw_devices.get(0).map(|d| d.id).unwrap_or(0); 
        
        let new_id = self.interlocutors.len();
        let new_profile = InterlocutorProfile {
            id: new_id,
            device_id,
            source_type,
            name: format!("Interlocutor {}", new_id),
            is_active: true,
            technical_name: raw_devices.get(0).and_then(|d| d.technical_name.clone()),
        };
        self.interlocutors.push(new_profile);
    }
    
    fn remove_profile(&mut self, index: usize) {
        if index < self.interlocutors.len() {
            self.interlocutors.remove(index);
            for (new_id, profile) in self.interlocutors.iter_mut().enumerate() {
                profile.id = new_id;
                profile.name = format!("Interlocutor {}", new_id);
            }
        }
    }
    
    fn save_transcript(&self) -> Result<PathBuf> { 
        if self.transcription.trim().is_empty() {
            return Err(anyhow!("No hay transcripción para guardar."));
        }

        let timestamp = Local::now().format("%Y%m%d_%H%M%S").to_string();
        
        let interlocutor_names: String = self.interlocutors.iter()
            .filter(|p| p.is_active)
            .map(|p| p.name.replace(' ', "_"))
            .collect::<Vec<_>>()
            .join("_");
            
        let filename = format!("{}_{}.md", interlocutor_names, timestamp);
        let output_path = Path::new(&self.output_dir).join(filename);

        std::fs::create_dir_all(&self.output_dir)?;
        
        let content = format!("# Minuta de Transcripción\n\nFecha: {}\n\n---\n\n{}", 
            Local::now().format("%d-%m-%Y %H:%M:%S"),
            self.transcription
        );
        
        std::fs::write(&output_path, content)?;
        
        Ok(output_path) 
    }

    fn get_device_name_static(input_devices: &[DeviceInfo], output_devices: &[DeviceInfo], source_type: SourceType, device_id: usize) -> String {
        let devices = match source_type {
            SourceType::Input => input_devices,
            SourceType::Output => output_devices,
        };
        
        devices.iter()
            .find(|d| d.id == device_id)
            .map(|d| d.name.clone())
            .unwrap_or_else(|| "Dispositivo no encontrado".to_string())
    }
}