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
    AudioMessage, InterlocutorProfile, SourceType, DeviceInfo, UiSender, 
    WHISPER_SAMPLE_RATE, CHUNK_DURATION_SECS, SILENCE_THRESHOLD
};

pub fn get_available_devices(host: &Host, is_input: bool) -> Vec<DeviceInfo> {
    // En Linux con PulseAudio/PipeWire, usar pactl para mayor confiabilidad
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
            if let Ok(name) = device.name() {
                println!("DEBUG: Dispositivo {} encontrado: {} (original idx: {})", 
                    if is_input { "INPUT" } else { "OUTPUT" }, name, i);
                
                // Filtrar dispositivos que claramente son monitores (para input)
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
    
    println!("DEBUG: Total dispositivos {}: {}", 
        if is_input { "INPUT" } else { "OUTPUT" }, devices.len());
    
    devices
}

#[cfg(target_os = "linux")]
fn get_linux_input_devices() -> Vec<DeviceInfo> {
    let mut devices = vec![];
    
    println!("DEBUG: Buscando dispositivos de entrada con pactl...");
    
    let output = Command::new("pactl")
        .args(&["list", "sources", "short"])
        .output();
    
    if let Ok(out) = output {
        let sources = String::from_utf8_lossy(&out.stdout);
        
        for line in sources.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                let tech_name = parts[1].to_string();
                
                // Solo inputs reales (no monitores)
                if !tech_name.contains(".monitor") && tech_name.starts_with("alsa_input") {
                    println!("DEBUG: Input encontrado: {}", tech_name);
                    
                    // Obtener descripción
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
                    
                    println!("  -> Agregado: {} (ID: {}, tech: {})", description, devices.len(), tech_name);
                    devices.push(DeviceInfo { 
                        id: devices.len(), 
                        name: description,
                        technical_name: Some(tech_name),
                    });
                }
            }
        }
    }
    
    println!("DEBUG: Total inputs encontrados: {}", devices.len());
    devices
}

pub fn audio_thread_main(model_name: String, tx_ui: UiSender, stop_signal: Arc<AtomicBool>, profiles: Vec<InterlocutorProfile>) -> Result<()> {
    tx_ui.send(AudioMessage::Status("Verificando modelo...".to_string()))?;
    
    let model_path = Runtime::new()?
        .block_on(download_whisper_model(&model_name))?;
    
    for profile in profiles {
        let tx_ui_for_func = tx_ui.clone();
        let tx_ui_for_err = tx_ui.clone();
        let stop_signal_clone = stop_signal.clone(); 
        let model_path_clone = model_path.clone();
        let profile_name_err = profile.name.clone();
        
        thread::spawn(move || {
            if let Err(e) = run_single_stream(profile, model_path_clone, tx_ui_for_func, stop_signal_clone) {
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

fn run_single_stream(profile: InterlocutorProfile, model_path: String, tx_ui: UiSender, stop_signal: Arc<AtomicBool>) -> Result<()> {
    #[cfg(target_os = "linux")]
    {
        return run_single_stream_linux_pipewire(profile, model_path, tx_ui, stop_signal);
    }
    
    #[cfg(not(target_os = "linux"))]
    {
        run_single_stream_cpal(profile, model_path, tx_ui, stop_signal)
    }
}

#[cfg(target_os = "linux")]
fn run_single_stream_linux_pipewire(profile: InterlocutorProfile, model_path: String, tx_ui: UiSender, stop_signal: Arc<AtomicBool>) -> Result<()> {
    use std::process::{Command, Stdio};
    use std::io::Read;
    
    let ctx = WhisperContext::new_with_params(&model_path, Default::default())
        .map_err(|e| anyhow!("Error cargando modelo: {:?}", e))?;
    
    let mut state = ctx.create_state()
        .map_err(|e| anyhow!("Error creando estado: {:?}", e))?;

    // Usar directamente el nombre técnico guardado en el perfil
    let device_name = profile.technical_name
        .ok_or_else(|| anyhow!("Dispositivo sin nombre técnico. Recarga la aplicación."))?;
    
    println!("DEBUG: Capturando desde: {}", device_name);
    
    // Verificar que el dispositivo existe
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
        "{} {} - {} (16kHz mono)", 
        source_icon, 
        profile.name,
        device_name
    )))?;
    
    // Usar parecord para capturar directamente
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
    
    // Buffer para leer audio (s16le = 2 bytes por sample)
    let mut buffer = vec![0u8; 4096];
    
    loop {
        if stop_signal.load(Ordering::SeqCst) {
            let _ = child.kill();
            break;
        }
        
        match stdout.read(&mut buffer) {
            Ok(0) => break, // EOF
            Ok(n) => {
                // Convertir s16le a f32
                for chunk in buffer[..n].chunks_exact(2) {
                    let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                    let normalized = sample as f32 / 32768.0;
                    accumulated_audio.push(normalized);
                }
                
                if accumulated_audio.len() >= target_samples {
                    let normalized_audio = normalize_audio(&accumulated_audio[..target_samples]);
                    let rms = calculate_rms(&normalized_audio);
                    
                    if rms >= SILENCE_THRESHOLD {
                        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
                        params.set_language(Some("es"));
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
                    
                    // Mantener overlap
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
fn run_single_stream_cpal(profile: InterlocutorProfile, model_path: String, tx_ui: UiSender, stop_signal: Arc<AtomicBool>) -> Result<()> {
    let host = cpal::default_host();
    
    let ctx = WhisperContext::new_with_params(&model_path, Default::default())
        .map_err(|e| anyhow!("Error cargando modelo: {:?}", e))?;
    
    let mut state = ctx.create_state()
        .map_err(|e| anyhow!("Error creando estado: {:?}", e))?;

    println!("DEBUG: Buscando dispositivo para perfil: {} (tipo: {:?}, ID: {})", 
        profile.name, profile.source_type, profile.device_id);
    let host = cpal::default_host();
    
    let ctx = WhisperContext::new_with_params(&model_path, Default::default())
        .map_err(|e| anyhow!("Error cargando modelo: {:?}", e))?;
    
    let mut state = ctx.create_state()
        .map_err(|e| anyhow!("Error creando estado: {:?}", e))?;

    println!("DEBUG: Buscando dispositivo para perfil: {} (tipo: {:?}, ID: {})", 
        profile.name, profile.source_type, profile.device_id);

    let device = match profile.source_type {
        SourceType::Input => {
            // Obtener lista de inputs de pactl
            #[cfg(target_os = "linux")]
            {
                let input_devices = get_linux_input_devices();
                println!("DEBUG: Inputs disponibles: {}", input_devices.len());
                
                let target_device = input_devices.get(profile.device_id)
                    .ok_or_else(|| anyhow!("Input ID {} no encontrado", profile.device_id))?;
                
                let tech_name = target_device.technical_name.as_ref()
                    .ok_or_else(|| anyhow!("Input sin nombre técnico"))?;
                
                println!("DEBUG: Buscando input en cpal: {}", tech_name);
                println!("DEBUG: Listando TODOS los dispositivos de cpal input_devices():");
                
                // Buscar en cpal por nombre técnico
                let mut found = None;
                for (idx, device) in host.input_devices()?.enumerate() {
                    if let Ok(name) = device.name() {
                        println!("  [{}] cpal device: '{}'", idx, name);
                        if name == *tech_name {
                            println!("    -> MATCH encontrado!");
                            found = Some(device);
                            break;
                        }
                    }
                }
                
                if found.is_none() {
                    println!("DEBUG: No se encontró match exacto, intentando match parcial...");
                    for device in host.input_devices()? {
                        if let Ok(name) = device.name() {
                            // Intentar match parcial
                            if name.contains("C-Media") || tech_name.contains(&name) || name.contains("USB") {
                                println!("    -> Match parcial encontrado: {}", name);
                                found = Some(device);
                                break;
                            }
                        }
                    }
                }
                
                found.ok_or_else(|| anyhow!("Input '{}' no encontrado en cpal.\n\nDispositivos cpal disponibles arriba.", tech_name))?
            }
            
            #[cfg(not(target_os = "linux"))]
            {
                let mut devices = host.input_devices()?;
                devices.nth(profile.device_id)
                    .ok_or_else(|| anyhow!("Input {} no encontrado", profile.device_id))?
            }
        }
        SourceType::Output => {
            // Para outputs (monitores)
            #[cfg(target_os = "linux")]
            {
                let output = Command::new("pactl")
                    .args(&["list", "sources"])
                    .output()?;
                
                let sources = String::from_utf8_lossy(&output.stdout);
                let mut monitors = Vec::new();
                let mut current_name = String::new();
                let mut current_desc = String::new();
                
                for line in sources.lines() {
                    let line = line.trim();
                    if line.starts_with("Source #") {
                        if !current_name.is_empty() && current_name.contains("monitor") {
                            monitors.push((current_name.clone(), current_desc.clone()));
                        }
                        current_name.clear();
                        current_desc.clear();
                    } else if line.starts_with("Name:") {
                        current_name = line.replace("Name:", "").trim().to_string();
                    } else if line.starts_with("Description:") {
                        current_desc = line.replace("Description:", "").trim().to_string();
                    }
                }
                
                if !current_name.is_empty() && current_name.contains("monitor") {
                    monitors.push((current_name, current_desc));
                }
                
                println!("DEBUG: Monitores disponibles: {}", monitors.len());
                for (i, (name, desc)) in monitors.iter().enumerate() {
                    println!("  [{}] {} ({})", i, desc, name);
                }
                
                let (monitor_name, monitor_desc) = monitors.get(profile.device_id)
                    .ok_or_else(|| anyhow!("Monitor con ID {} no encontrado. Total monitores: {}", profile.device_id, monitors.len()))?;
                
                println!("DEBUG: Buscando monitor en cpal: {}", monitor_name);
                println!("DEBUG: Listando TODOS los dispositivos de cpal input_devices():");
                
                let mut found_device = None;
                for (idx, device) in host.input_devices()?.enumerate() {
                    if let Ok(name) = device.name() {
                        println!("  [{}] cpal device: '{}'", idx, name);
                        if name == *monitor_name {
                            println!("    -> MATCH encontrado!");
                            found_device = Some(device);
                            break;
                        }
                    }
                }
                
                if found_device.is_none() {
                    println!("DEBUG: No se encontró match exacto, intentando match parcial...");
                    for device in host.input_devices()? {
                        if let Ok(name) = device.name() {
                            if name.contains(monitor_name) || monitor_name.contains(&name) {
                                println!("    -> Match parcial encontrado: {}", name);
                                found_device = Some(device);
                                break;
                            }
                        }
                    }
                }
                
                found_device.ok_or_else(|| anyhow!("Monitor '{}' no encontrado en cpal.\n\nAsegúrate de que el audio esté sonando.\nDispositivos cpal disponibles arriba.", monitor_desc))?
            }
            
            #[cfg(not(target_os = "linux"))]
            {
                let mut devices = host.output_devices()?;
                devices.nth(profile.device_id)
                    .ok_or_else(|| anyhow!("Output {} no encontrado", profile.device_id))?
            }
        }
    };
    
    println!("DEBUG: Dispositivo encontrado: {:?}", device.name());
    
    let config = device.default_input_config()?;
    let sample_rate = config.sample_rate().0;
    let channels = config.channels() as usize;
    
    let source_icon = match profile.source_type {
        SourceType::Input => "🎤",
        SourceType::Output => "🔊",
    };
    
    tx_ui.send(AudioMessage::Status(format!(
        "{} {} - {} ({}Hz, {} ch)", 
        source_icon, 
        profile.name, 
        device.name()?, 
        sample_rate, 
        channels
    )))?;
    
    let (audio_tx, audio_rx) = channel::<AudioBuffer>();
    
    let data_callback = move |data: &[f32], _: &cpal::InputCallbackInfo| {
        let _ = audio_tx.send(data.to_vec());
    };
    
    let name_for_callback = profile.name.clone();

    let stream: Stream = device.build_input_stream(
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
                    params.set_language(Some("es"));
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

fn normalize_audio(input: &[f32]) -> Vec<f32> {
    let max_amplitude = input.iter()
        .map(|&s| s.abs())
        .fold(0.0f32, f32::max);
    
    if max_amplitude < 0.0001 {
        return input.to_vec();
    }
    
    let target_peak = 0.95;
    let gain = target_peak / max_amplitude;
    
    input.iter().map(|&s| s * gain).collect()
}

fn calculate_rms(audio: &[f32]) -> f32 {
    let sum_squares: f32 = audio.iter().map(|&s| s * s).sum();
    (sum_squares / audio.len() as f32).sqrt()
}

async fn download_whisper_model(model_name: &str) -> Result<String> {
    let models_dir = Path::new("models");
    let model_file = format!("ggml-{}.bin", model_name);
    let model_path = models_dir.join(&model_file);
    
    if !models_dir.exists() {
        std::fs::create_dir_all(models_dir)?;
        println!("📁 Creado directorio 'models/'");
    }
    
    if model_path.exists() {
        println!("✓ Modelo '{}' ya existe", model_name);
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