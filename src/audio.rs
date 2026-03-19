use anyhow::{Result, anyhow};
use cpal::traits::{DeviceTrait, HostTrait};
use cpal::Host;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::path::Path;
use std::io::Write;
use std::thread;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext};
use tokio::runtime::Runtime;
use futures_util::StreamExt;
use reqwest::Client;
use std::process::Command;
use crate::data::{
    AudioMessage, InterlocutorProfile, LanguageConfig, SourceType, DeviceInfo, UiSender, 
    WHISPER_SAMPLE_RATE, CHUNK_DURATION_SECS, SILENCE_THRESHOLD
};

pub fn get_available_devices(host: &Host, is_input: bool) -> Vec<DeviceInfo> {
    #[cfg(target_os = "linux")]
    {
        if is_input {
            return get_linux_input_devices();
        }
    }
    
    let mut devices: Vec<DeviceInfo> = Vec::new();
    
    let device_iter = if is_input {
        host.input_devices()
    } else {
        host.output_devices()
    };

    if let Ok(device_list) = device_iter {
        let mut real_index = 0;
        for (i, device) in device_list.enumerate() {
            if let Ok(desc) = device.description() {
                let name = desc.name().to_string();
                println!("DEBUG: Dispositivo {} encontrado: {} (original idx: {})", 
                    if is_input { "INPUT" } else { "OUTPUT" }, name, i);
                
                if is_input && (name.contains(".monitor") || name.contains("Monitor of")) {
                    println!("  -> Saltado (es monitor)");
                    continue;
                }
                
                println!("  -> Agregado con ID {}", real_index);
                devices.push(DeviceInfo { 
                    id: real_index, 
                    name,
                    technical_name: None,
                });
                real_index += 1;
            }
        }
    }
    
    devices
}

#[cfg(target_os = "linux")]
fn get_linux_input_devices() -> Vec<DeviceInfo> {
    let mut devices = vec![];
    
    let output = Command::new("pactl")
        .args(&["list", "sources", "short"])
        .output();
    
    if let Ok(out) = output {
        let sources = String::from_utf8_lossy(&out.stdout);
        
        for line in sources.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                let tech_name = parts[1].to_string();
                
                if !tech_name.contains(".monitor") && tech_name.starts_with("alsa_input") {
                    let desc_output = Command::new("pactl")
                        .args(&["list", "sources"])
                        .output();
                    
                    let mut description = tech_name.clone();
                    if let Ok(desc_out) = desc_output {
                        let full_list = String::from_utf8_lossy(&desc_out.stdout);
                        let mut found_device = false;
                        
                        for desc_line in full_list.lines() {
                            if desc_line.contains(&format!("Name: {}", tech_name)) {
                                found_device = true;
                            } else if found_device && desc_line.trim().starts_with("Description:") {
                                description = desc_line.replace("Description:", "").trim().to_string();
                                break;
                            }
                        }
                    }
                    
                    devices.push(DeviceInfo { 
                        id: devices.len(), 
                        name: description,
                        technical_name: Some(tech_name),
                    });
                }
            }
        }
    }
    
    devices
}

pub fn audio_thread_main(
    model_name: String,
    tx_ui: UiSender,
    stop_signal: Arc<AtomicBool>,
    profiles: Vec<InterlocutorProfile>,
    lang_config: LanguageConfig,
) -> Result<()> {
    tx_ui.send(AudioMessage::Status("Verificando modelo...".to_string()))?;
    
    let model_path = Runtime::new()?
        .block_on(download_whisper_model(&model_name))?;
    
    for profile in profiles {
        let tx_ui_for_func = tx_ui.clone();
        let tx_ui_for_err = tx_ui.clone();
        let stop_signal_clone = stop_signal.clone(); 
        let model_path_clone = model_path.clone();
        let profile_name_err = profile.name.clone();
        let lang_config_clone = lang_config.clone();
        
        thread::spawn(move || {
            if let Err(e) = run_single_stream(profile, model_path_clone, tx_ui_for_func, stop_signal_clone, lang_config_clone) {
                let _ = tx_ui_for_err.send(AudioMessage::Error(format!("Error en {}: {:?}", profile_name_err, e)));
            }
        });
    }
    
    while !stop_signal.load(Ordering::SeqCst) {
        thread::sleep(std::time::Duration::from_millis(50));
    }
    
    tx_ui.send(AudioMessage::Status("Captura finalizada.".to_string()))?;
    Ok(())
}

fn run_single_stream(
    profile: InterlocutorProfile,
    model_path: String,
    tx_ui: UiSender,
    stop_signal: Arc<AtomicBool>,
    lang_config: LanguageConfig,
) -> Result<()> {
    #[cfg(target_os = "linux")]
    {
        return run_single_stream_linux_pipewire(profile, model_path, tx_ui, stop_signal, lang_config);
    }
    
    #[cfg(not(target_os = "linux"))]
    {
        run_single_stream_cpal(profile, model_path, tx_ui, stop_signal, lang_config)
    }
}

/// Aplica la configuración de idioma a los parámetros de Whisper.
/// 
/// Comportamiento:
/// - source_lang=None           → autodetección del idioma de entrada
/// - source_lang=Some("en")     → asume audio en inglés (mejor rendimiento si se sabe)
/// - translate_to_english=false → transcribe en el idioma de origen (sin traducción)
/// - translate_to_english=true  → traduce a inglés (única traducción nativa de Whisper)
fn apply_language_params(params: &mut FullParams, lang_config: &LanguageConfig) {
    params.set_language(lang_config.source_lang);
    params.set_translate(lang_config.translate_to_english);
}

#[cfg(target_os = "linux")]
fn run_single_stream_linux_pipewire(
    profile: InterlocutorProfile,
    model_path: String,
    tx_ui: UiSender,
    stop_signal: Arc<AtomicBool>,
    lang_config: LanguageConfig,
) -> Result<()> {
    use std::process::{Command, Stdio};
    use std::io::Read;
    
    let ctx = WhisperContext::new_with_params(&model_path, Default::default())
        .map_err(|e| anyhow!("Error cargando modelo: {:?}", e))?;
    
    let mut state = ctx.create_state()
        .map_err(|e| anyhow!("Error creando estado: {:?}", e))?;

    let device_name = profile.technical_name
        .ok_or_else(|| anyhow!("Dispositivo sin nombre técnico. Recarga la aplicación."))?;
    
    let check = Command::new("pactl")
        .args(&["list", "sources", "short"])
        .output()?;
    
    let sources = String::from_utf8_lossy(&check.stdout);
    if !sources.contains(&device_name) {
        return Err(anyhow!(
            "Dispositivo '{}' no encontrado.\n\nDispositivos disponibles:\n{}", 
            device_name, 
            sources
        ));
    }
    
    let source_icon = match profile.source_type {
        SourceType::Input => "🎤",
        SourceType::Output => "🔊",
    };
    
    tx_ui.send(AudioMessage::Status(format!(
        "{} {} - {} (16kHz mono) [{}→{}]", 
        source_icon, 
        profile.name,
        device_name,
        lang_config.source_label(),
        lang_config.dest_label(),
    )))?;
    
    let mut child = Command::new("parecord")
        .args(&[
            "--device", &device_name,
            "--rate", "16000",
            "--channels", "1",
            "--format", "s16le",
            "--raw",
        ])
        .stdout(Stdio::piped())
        .spawn()
        .map_err(|e| anyhow!("Error iniciando parecord: {:?}. ¿Está instalado?", e))?;
    
    let mut stdout = child.stdout.take()
        .ok_or_else(|| anyhow!("No se pudo obtener stdout de parecord"))?;
    
    let mut accumulated_audio: Vec<f32> = Vec::new();
    let target_samples = (WHISPER_SAMPLE_RATE * CHUNK_DURATION_SECS) as usize;
    let mut buffer = vec![0u8; 4096];
    
    loop {
        if stop_signal.load(Ordering::SeqCst) {
            let _ = child.kill();
            break;
        }
        
        match stdout.read(&mut buffer) {
            Ok(0) => break,
            Ok(n) => {
                for chunk in buffer[..n].chunks_exact(2) {
                    let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                    accumulated_audio.push(sample as f32 / 32768.0);
                }
                
                if accumulated_audio.len() >= target_samples {
                    let normalized_audio = normalize_audio(&accumulated_audio[..target_samples]);
                    let rms = calculate_rms(&normalized_audio);
                    
                    if rms >= SILENCE_THRESHOLD {
                        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
                        apply_language_params(&mut params, &lang_config);
                        params.set_print_special(false);
                        params.set_print_progress(false);
                        params.set_print_realtime(false);
                        params.set_print_timestamps(false);
                        params.set_suppress_blank(true);
                        params.set_suppress_nst(true);
                        params.set_no_speech_thold(0.6);
                        
                        match state.full(params, &normalized_audio) {
                            Ok(_) => {
                                let num_segments = state.full_n_segments();
                                if num_segments > 0 {
                                    let mut full_text = String::new();
                                    for i in 0..num_segments {
                                        if let Some(segment) = state.get_segment(i) {
                                            let text = segment.to_string().trim().to_string();
                                            if !text.is_empty() && text.len() > 1 {
                                                full_text.push_str(&text);
                                                full_text.push(' ');
                                            }
                                        }
                                    }
                                    
                                    let trimmed = full_text.trim();
                                    if !trimmed.is_empty() {
                                        tx_ui.send(AudioMessage::Transcription { 
                                            text: trimmed.to_string(), 
                                            name: profile.name.clone() 
                                        })?;
                                    }
                                }
                            }
                            Err(e) => {
                                eprintln!("Error en transcripción en hilo {}: {:?}", profile.name, e);
                            }
                        }
                    }
                    
                    let overlap = target_samples * 3 / 10;
                    accumulated_audio = accumulated_audio.split_off(
                        accumulated_audio.len().saturating_sub(overlap)
                    );
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                thread::sleep(std::time::Duration::from_millis(10));
                continue;
            }
            Err(e) => return Err(anyhow!("Error leyendo audio: {:?}", e)),
        }
    }
    
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn run_single_stream_cpal(
    profile: InterlocutorProfile,
    model_path: String,
    tx_ui: UiSender,
    stop_signal: Arc<AtomicBool>,
    lang_config: LanguageConfig,
) -> Result<()> {
    let host = cpal::default_host();
    
    let ctx = WhisperContext::new_with_params(&model_path, Default::default())
        .map_err(|e| anyhow!("Error cargando modelo: {:?}", e))?;
    
    let mut state = ctx.create_state()
        .map_err(|e| anyhow!("Error creando estado: {:?}", e))?;

    let device = match profile.source_type {
        SourceType::Input => {
            #[cfg(not(target_os = "linux"))]
            {
                let mut devices = host.input_devices()?;
                devices.nth(profile.device_id)
                    .ok_or_else(|| anyhow!("Input {} no encontrado", profile.device_id))?
            }
        }
        SourceType::Output => {
            #[cfg(not(target_os = "linux"))]
            {
                let mut devices = host.output_devices()?;
                devices.nth(profile.device_id)
                    .ok_or_else(|| anyhow!("Output {} no encontrado", profile.device_id))?
            }
        }
    };
    
    let config = device.default_input_config()?;
    let sample_rate = config.sample_rate().0;
    let channels = config.channels() as usize;
    
    let source_icon = match profile.source_type {
        SourceType::Input => "🎤",
        SourceType::Output => "🔊",
    };
    
    tx_ui.send(AudioMessage::Status(format!(
        "{} {} - {} ({}Hz, {} ch) [{}→{}]", 
        source_icon, 
        profile.name, 
        device.name()?, 
        sample_rate, 
        channels,
        lang_config.source_label(),
        lang_config.dest_label(),
    )))?;
    
    let (audio_tx, audio_rx) = std::sync::mpsc::channel::<Vec<f32>>();
    
    let data_callback = move |data: &[f32], _: &cpal::InputCallbackInfo| {
        let _ = audio_tx.send(data.to_vec());
    };
    
    let name_for_callback = profile.name.clone();

    let stream = device.build_input_stream(
        &config.into(),
        data_callback,
        move |err| eprintln!("Error en stream [{}]: {}", name_for_callback, err),
        None
    )?;
    
    stream.play()?;
    
    let mut accumulated_audio = Vec::new();
    let target_samples = (sample_rate * CHUNK_DURATION_SECS) as usize;
    
    loop {
        if stop_signal.load(Ordering::SeqCst) {
            break;
        }
        
        match audio_rx.recv_timeout(std::time::Duration::from_millis(100)) {
            Ok(buffer) => {
                let mono_buffer = if channels > 1 {
                    convert_to_mono(&buffer, channels)
                } else {
                    buffer
                };
                
                accumulated_audio.extend_from_slice(&mono_buffer);
                
                if accumulated_audio.len() >= target_samples {
                    let audio_to_process = if sample_rate != WHISPER_SAMPLE_RATE {
                        resample(&accumulated_audio, sample_rate, WHISPER_SAMPLE_RATE)
                    } else {
                        accumulated_audio.clone()
                    };
                    
                    let normalized_audio = normalize_audio(&audio_to_process);
                    let rms = calculate_rms(&normalized_audio);
                    
                    if rms < SILENCE_THRESHOLD {
                        accumulated_audio.clear();
                        continue;
                    }
                    
                    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
                    apply_language_params(&mut params, &lang_config);
                    params.set_print_special(false);
                    params.set_print_progress(false);
                    params.set_print_realtime(false);
                    params.set_print_timestamps(false);
                    params.set_suppress_blank(true);
                    params.set_suppress_nst(true);
                    params.set_no_speech_thold(0.6);
                    
                    match state.full(params, &normalized_audio) {
                        Ok(_) => {
                            let num_segments = state.full_n_segments();
                            if num_segments > 0 {
                                let mut full_text = String::new();
                                for i in 0..num_segments {
                                    if let Some(segment) = state.get_segment(i) {
                                        let text = segment.to_string().trim().to_string();
                                        if !text.is_empty() && text.len() > 1 {
                                            full_text.push_str(&text);
                                            full_text.push(' ');
                                        }
                                    }
                                }
                                
                                let trimmed = full_text.trim();
                                if !trimmed.is_empty() {
                                    tx_ui.send(AudioMessage::Transcription { 
                                        text: trimmed.to_string(), 
                                        name: profile.name.clone() 
                                    })?;
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("Error en transcripción en hilo {}: {:?}", profile.name, e);
                        }
                    }
                    
                    let overlap = target_samples * 3 / 10;
                    accumulated_audio = accumulated_audio.split_off(
                        accumulated_audio.len().saturating_sub(overlap)
                    );
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn convert_to_mono(buffer: &[f32], channels: usize) -> Vec<f32> {
    buffer.chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

#[cfg(not(target_os = "linux"))]
fn resample(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    let ratio = to_rate as f64 / from_rate as f64;
    let new_len = (input.len() as f64 * ratio) as usize;
    (0..new_len)
        .map(|i| {
            let src = i as f64 / ratio;
            let idx = src as usize;
            let frac = src - idx as f64;
            let a = input.get(idx).copied().unwrap_or(0.0);
            let b = input.get(idx + 1).copied().unwrap_or(0.0);
            a + (b - a) * frac as f32
        })
        .collect()
}

fn normalize_audio(input: &[f32]) -> Vec<f32> {
    let max_amplitude = input.iter()
        .map(|&s| s.abs())
        .fold(0.0f32, f32::max);
    
    if max_amplitude < 0.0001 {
        return input.to_vec();
    }
    
    let gain = 0.95 / max_amplitude;
    input.iter().map(|&s| s * gain).collect()
}

fn calculate_rms(audio: &[f32]) -> f32 {
    let sum_squares: f32 = audio.iter().map(|&s| s * s).sum();
    (sum_squares / audio.len() as f32).sqrt()
}

pub async fn download_whisper_model(model_name: &str) -> Result<String> {
    let models_dir = Path::new("models");
    let model_file = format!("ggml-{}.bin", model_name);
    let model_path = models_dir.join(&model_file);
    
    if !models_dir.exists() {
        std::fs::create_dir_all(models_dir)?;
    }
    
    if model_path.exists() {
        return Ok(model_path.to_string_lossy().to_string());
    }

    let url = format!(
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{}",
        model_file
    );
    
    println!("📥 Descargando modelo '{}' desde Hugging Face...", model_name);
    
    let client = Client::new();
    let response = client.get(&url).send().await?;
    
    if !response.status().is_success() {
        anyhow::bail!("Error al descargar: HTTP {}", response.status());
    }
    
    let total_size = response.content_length().unwrap_or(0);
    let mut downloaded: u64 = 0;
    let mut file = std::fs::File::create(&model_path)?;
    let mut stream = response.bytes_stream();
    
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk)?;
        downloaded += chunk.len() as u64;
        
        if total_size > 0 {
            print!("\r   Progreso: {:.1}% ({} MB / {} MB)", 
                (downloaded as f64 / total_size as f64) * 100.0,
                downloaded / 1_000_000,
                total_size / 1_000_000
            );
            std::io::stdout().flush()?;
        }
    }
    
    println!("\n✓ Modelo descargado");
    Ok(model_path.to_string_lossy().to_string())
}