use anyhow::{Result, anyhow};
use cpal::traits::{DeviceTrait, HostTrait};
#[cfg(not(target_os = "linux"))]
use cpal::traits::StreamTrait as _;
use cpal::Host;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::path::Path;
use std::io::Write;
use std::thread;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext};
use tokio::runtime::Runtime;
use futures_util::StreamExt;
use reqwest::Client;
#[cfg(target_os = "linux")]
use std::process::Command;
use crate::data::{
    AudioMessage, DiarizeConfig, InterlocutorProfile, LanguageConfig,
    SourceType, DeviceInfo, UiSender,
    WHISPER_SAMPLE_RATE, CHUNK_DURATION_SECS, SILENCE_THRESHOLD,
};
use crate::diarize::{
    DiarizeEngine, IncrementalClusterer, download_diarize_model,
    DEFAULT_DIARIZE_MODEL_URL,
};

// ── Enumeración de dispositivos ────────────────────────────────────────────

pub fn get_available_devices(host: &Host, is_input: bool) -> Vec<DeviceInfo> {
    #[cfg(target_os = "linux")]
    if is_input {
        return get_linux_input_devices();
    }

    let mut devices: Vec<DeviceInfo> = Vec::new();
    let iter = if is_input { host.input_devices() } else { host.output_devices() };

    if let Ok(device_list) = iter {
        let mut real_index = 0;
        for device in device_list {
            if let Ok(desc) = device.description() {
                let name = desc.name().to_string();

                #[cfg(target_os = "linux")]
                if is_input && (name.contains(".monitor") || name.contains("Monitor of")) {
                    continue;
                }

                devices.push(DeviceInfo {
                    id: real_index,
                    name: name.clone(),
                    technical_name: Some(name),
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
                    let mut description = tech_name.clone();

                    if let Ok(desc_out) = Command::new("pactl").args(&["list", "sources"]).output() {
                        let full_list = String::from_utf8_lossy(&desc_out.stdout);
                        let mut found = false;
                        for desc_line in full_list.lines() {
                            if desc_line.contains(&format!("Name: {}", tech_name)) {
                                found = true;
                            } else if found && desc_line.trim().starts_with("Description:") {
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

// ── Hilo principal de audio ────────────────────────────────────────────────

pub fn audio_thread_main(
    model_name: String,
    tx_ui: UiSender,
    stop_signal: Arc<AtomicBool>,
    profiles: Vec<InterlocutorProfile>,
    lang_config: LanguageConfig,
    diarize_config: DiarizeConfig,
) -> Result<()> {
    tx_ui.send(AudioMessage::Status("Verificando modelo Whisper...".to_string()))?;

    let model_path = Runtime::new()?
        .block_on(download_whisper_model(&model_name))?;

    // Cargar motor de diarización si está habilitado
    let diarize_engine: Option<Arc<Mutex<DiarizeEngine>>> = if diarize_config.enabled {
        tx_ui.send(AudioMessage::Status("Cargando modelo de diarización...".to_string()))?;
        match Runtime::new()?.block_on(download_diarize_model(DEFAULT_DIARIZE_MODEL_URL)) {
            Ok(diarize_path) => match DiarizeEngine::new(&diarize_path) {
                Ok(engine) => {
                    tx_ui.send(AudioMessage::Status("✅ Diarización lista.".to_string()))?;
                    Some(Arc::new(Mutex::new(engine)))
                }
                Err(e) => {
                    tx_ui.send(AudioMessage::Status(format!(
                        "⚠️ Diarización no disponible: {:?}", e
                    )))?;
                    None
                }
            },
            Err(e) => {
                tx_ui.send(AudioMessage::Status(format!(
                    "⚠️ Modelo diarización no descargado: {:?}", e
                )))?;
                None
            }
        }
    } else {
        None
    };

    let threshold = diarize_config.threshold;

    for profile in profiles {
        let tx_func = tx_ui.clone();
        let tx_err  = tx_ui.clone();
        let stop    = stop_signal.clone();
        let model   = model_path.clone();
        let lang    = lang_config.clone();
        let name    = profile.name.clone();
        let engine  = diarize_engine.clone();

        thread::spawn(move || {
            if let Err(e) = run_single_stream(profile, model, tx_func, stop, lang, engine, threshold) {
                let _ = tx_err.send(AudioMessage::Error(format!("Error en {}: {:?}", name, e)));
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
    diarize_engine: Option<Arc<Mutex<DiarizeEngine>>>,
    diarize_threshold: f32,
) -> Result<()> {
    #[cfg(target_os = "linux")]
    return run_single_stream_linux(
        profile, model_path, tx_ui, stop_signal, lang_config,
        diarize_engine, diarize_threshold,
    );

    #[cfg(not(target_os = "linux"))]
    run_single_stream_cpal(
        profile, model_path, tx_ui, stop_signal, lang_config,
        diarize_engine, diarize_threshold,
    )
}

// ── Captura Linux (parecord / PipeWire) ───────────────────────────────────

#[cfg(target_os = "linux")]
fn run_single_stream_linux(
    profile: InterlocutorProfile,
    model_path: String,
    tx_ui: UiSender,
    stop_signal: Arc<AtomicBool>,
    lang_config: LanguageConfig,
    diarize_engine: Option<Arc<Mutex<DiarizeEngine>>>,
    diarize_threshold: f32,
) -> Result<()> {
    use std::process::Stdio;
    use std::io::Read;

    let mut ctx_params = whisper_rs::WhisperContextParameters::default();
    ctx_params.use_gpu = true;   // Usa GPU si el binario fue compilado con CUDA/Metal/OpenCL
    let ctx = WhisperContext::new_with_params(&model_path, ctx_params)
        .map_err(|e| anyhow!("Error cargando modelo: {:?}", e))?;
    let mut state = ctx.create_state()
        .map_err(|e| anyhow!("Error creando estado: {:?}", e))?;

    let mut clusterer = diarize_engine.as_ref().map(|_| IncrementalClusterer::new(diarize_threshold));

    let device_name = profile.technical_name
        .ok_or_else(|| anyhow!("Dispositivo sin nombre técnico. Recarga la aplicación."))?;

    let check = Command::new("pactl").args(&["list", "sources", "short"]).output()?;
    let sources = String::from_utf8_lossy(&check.stdout);
    if !sources.contains(&device_name) {
        return Err(anyhow!(
            "Dispositivo '{}' no encontrado.\n\nDispositivos disponibles:\n{}",
            device_name, sources
        ));
    }

    let source_icon = match profile.source_type { SourceType::Input => "🎤", SourceType::Output => "🔊" };
    let diarize_tag = if diarize_engine.is_some() { " [diarización ON]" } else { "" };
    tx_ui.send(AudioMessage::Status(format!(
        "{} {} - {} (16kHz mono) [{}→{}]{}",
        source_icon, profile.name, device_name,
        lang_config.source_label(), lang_config.dest_label(), diarize_tag,
    )))?;

    let mut child = Command::new("parecord")
        .args(&["--device", &device_name, "--rate", "16000",
                "--channels", "1", "--format", "s16le", "--raw"])
        .stdout(Stdio::piped())
        .spawn()
        .map_err(|e| anyhow!("Error iniciando parecord: {:?}. ¿Está instalado?", e))?;

    let mut stdout = child.stdout.take()
        .ok_or_else(|| anyhow!("No se pudo obtener stdout de parecord"))?;

    let mut accumulated: Vec<f32> = Vec::new();
    let target = (WHISPER_SAMPLE_RATE * CHUNK_DURATION_SECS) as usize;
    let mut buf = vec![0u8; 4096];

    loop {
        if stop_signal.load(Ordering::SeqCst) { let _ = child.kill(); break; }

        match stdout.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                for chunk in buf[..n].chunks_exact(2) {
                    let s = i16::from_le_bytes([chunk[0], chunk[1]]);
                    accumulated.push(s as f32 / 32768.0);
                }
                if accumulated.len() >= target {
                    process_and_send(
                        &accumulated[..target], &mut state, &lang_config,
                        &profile.name, &tx_ui,
                        diarize_engine.as_ref(), clusterer.as_mut(),
                    )?;
                    let overlap = target * 3 / 10;
                    accumulated = accumulated.split_off(accumulated.len().saturating_sub(overlap));
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                thread::sleep(std::time::Duration::from_millis(10));
            }
            Err(e) => return Err(anyhow!("Error leyendo audio: {:?}", e)),
        }
    }

    Ok(())
}

// ── Captura multiplataforma (cpal / WASAPI / CoreAudio) ───────────────────

#[cfg(not(target_os = "linux"))]
fn run_single_stream_cpal(
    profile: InterlocutorProfile,
    model_path: String,
    tx_ui: UiSender,
    stop_signal: Arc<AtomicBool>,
    lang_config: LanguageConfig,
    diarize_engine: Option<Arc<Mutex<DiarizeEngine>>>,
    diarize_threshold: f32,
) -> Result<()> {
    let host = cpal::default_host();

    let mut ctx_params = whisper_rs::WhisperContextParameters::default();
    ctx_params.use_gpu = true;   // Usa GPU si el binario fue compilado con CUDA/Metal/OpenCL
    let ctx = WhisperContext::new_with_params(&model_path, ctx_params)
        .map_err(|e| anyhow!("Error cargando modelo: {:?}", e))?;
    let mut state = ctx.create_state()
        .map_err(|e| anyhow!("Error creando estado: {:?}", e))?;

    let mut clusterer = diarize_engine.as_ref().map(|_| IncrementalClusterer::new(diarize_threshold));

    let tech_name = profile.technical_name.clone()
        .ok_or_else(|| anyhow!(
            "Dispositivo sin nombre técnico. Reconfigura el perfil en Ajustes."
        ))?;

    let device = host.input_devices()?
        .find(|d| {
            d.description()
                .map(|desc| desc.name() == tech_name.as_str())
                .unwrap_or(false)
        })
        .ok_or_else(|| anyhow!(
            "Dispositivo '{}' no encontrado.\n\
             • Windows: comprueba que el dispositivo sigue conectado.\n\
             • Para captura de sistema: activa 'Stereo Mix' en el panel de sonido.",
            tech_name
        ))?;

    let config = device.default_input_config()?;
    let sample_rate = u32::from(config.sample_rate());
    let channels = config.channels() as usize;

    let source_icon = match profile.source_type { SourceType::Input => "🎤", SourceType::Output => "🔊" };
    let diarize_tag = if diarize_engine.is_some() { " [diarización ON]" } else { "" };
    tx_ui.send(AudioMessage::Status(format!(
        "{} {} - {} ({}Hz, {}ch) [{}→{}]{}",
        source_icon, profile.name, tech_name,
        sample_rate, channels,
        lang_config.source_label(), lang_config.dest_label(), diarize_tag,
    )))?;

    let (audio_tx, audio_rx) = std::sync::mpsc::channel::<Vec<f32>>();
    let name_cb = profile.name.clone();

    let stream = device.build_input_stream(
        &config.into(),
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            let _ = audio_tx.send(data.to_vec());
        },
        move |err| eprintln!("Error en stream [{}]: {}", name_cb, err),
        None,
    )?;
    stream.play()?;

    let mut accumulated: Vec<f32> = Vec::new();
    let target = (sample_rate * CHUNK_DURATION_SECS) as usize;

    loop {
        if stop_signal.load(Ordering::SeqCst) { break; }

        match audio_rx.recv_timeout(std::time::Duration::from_millis(100)) {
            Ok(buf) => {
                let mono = if channels > 1 { to_mono(&buf, channels) } else { buf };
                accumulated.extend_from_slice(&mono);

                if accumulated.len() >= target {
                    let audio = if sample_rate != WHISPER_SAMPLE_RATE {
                        resample(&accumulated[..target], sample_rate, WHISPER_SAMPLE_RATE)
                    } else {
                        accumulated[..target].to_vec()
                    };

                    process_and_send(
                        &audio, &mut state, &lang_config,
                        &profile.name, &tx_ui,
                        diarize_engine.as_ref(), clusterer.as_mut(),
                    )?;

                    let overlap = target * 3 / 10;
                    accumulated = accumulated.split_off(accumulated.len().saturating_sub(overlap));
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }

    Ok(())
}

// ── Helpers de audio compartidos ──────────────────────────────────────────

/// Normaliza, comprueba silencio, transcribe y diariza (si disponible).
fn process_and_send(
    audio: &[f32],
    state: &mut whisper_rs::WhisperState,
    lang_config: &LanguageConfig,
    name: &str,
    tx_ui: &UiSender,
    diarize_engine: Option<&Arc<Mutex<DiarizeEngine>>>,
    clusterer: Option<&mut IncrementalClusterer>,
) -> Result<()> {
    let normalized = normalize_audio(audio);
    if calculate_rms(&normalized) < SILENCE_THRESHOLD {
        return Ok(());
    }

    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_language(lang_config.source_lang);
    params.set_translate(lang_config.translate_to_english);
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    params.set_suppress_blank(true);
    params.set_suppress_nst(true);
    params.set_no_speech_thold(0.6);

    if let Ok(_) = state.full(params, &normalized) {
        let n = state.full_n_segments();
        if n > 0 {
            let mut text = String::new();
            for i in 0..n {
                if let Some(seg) = state.get_segment(i) {
                    let t = seg.to_string().trim().to_string();
                    if !t.is_empty() && t.len() > 1 {
                        text.push_str(&t);
                        text.push(' ');
                    }
                }
            }
            let trimmed = text.trim().to_string();
            if !trimmed.is_empty() {
                // Diarización en tiempo real: extraer embedding + asignar speaker
                let speaker_id = match (diarize_engine, clusterer) {
                    (Some(engine_arc), Some(clust)) => {
                        engine_arc.lock().ok()
                            .and_then(|mut engine| engine.extract_embedding(&normalized).ok())
                            .map(|emb| clust.assign(&emb))
                    }
                    _ => None,
                };

                tx_ui.send(AudioMessage::Transcription {
                    text: trimmed,
                    name: name.to_string(),
                    speaker_id,
                })?;
            }
        }
    }

    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn to_mono(buf: &[f32], channels: usize) -> Vec<f32> {
    buf.chunks(channels)
        .map(|f| f.iter().sum::<f32>() / channels as f32)
        .collect()
}

#[cfg(not(target_os = "linux"))]
fn resample(input: &[f32], from: u32, to: u32) -> Vec<f32> {
    let ratio = to as f64 / from as f64;
    let len = (input.len() as f64 * ratio) as usize;
    (0..len).map(|i| {
        let src = i as f64 / ratio;
        let idx = src as usize;
        let frac = (src - idx as f64) as f32;
        let a = input.get(idx).copied().unwrap_or(0.0);
        let b = input.get(idx + 1).copied().unwrap_or(0.0);
        a + (b - a) * frac
    }).collect()
}

fn normalize_audio(input: &[f32]) -> Vec<f32> {
    let max = input.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if max < 0.0001 { return input.to_vec(); }
    input.iter().map(|&s| s * (0.95 / max)).collect()
}

fn calculate_rms(audio: &[f32]) -> f32 {
    let sum: f32 = audio.iter().map(|&s| s * s).sum();
    (sum / audio.len() as f32).sqrt()
}

// ── Descarga del modelo ────────────────────────────────────────────────────

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

    println!("📥 Descargando modelo '{}'...", model_name);

    let client = Client::new();
    let response = client.get(&url).send().await?;

    if !response.status().is_success() {
        anyhow::bail!("Error al descargar: HTTP {}", response.status());
    }

    let total = response.content_length().unwrap_or(0);
    let mut downloaded: u64 = 0;
    let mut file = std::fs::File::create(&model_path)?;
    let mut stream = response.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk)?;
        downloaded += chunk.len() as u64;
        if total > 0 {
            print!("\r   {:.1}% ({}/{} MB)",
                (downloaded as f64 / total as f64) * 100.0,
                downloaded / 1_000_000, total / 1_000_000);
            std::io::stdout().flush()?;
        }
    }

    println!("\n✓ Modelo descargado");
    Ok(model_path.to_string_lossy().to_string())
}