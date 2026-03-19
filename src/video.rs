use anyhow::{Result, anyhow};
use std::io::Read;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::runtime::Runtime;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext};

use crate::audio::download_whisper_model;
use crate::data::{LanguageConfig, VideoMessage, WHISPER_SAMPLE_RATE};

/// Chunks de 30 segundos — ventana nativa de Whisper, calidad óptima.
const VIDEO_CHUNK_SECS: u32 = 30;

pub fn video_transcription_thread(
    file_path: String,
    model_name: String,
    lang_config: LanguageConfig,
    tx: std::sync::mpsc::Sender<VideoMessage>,
    stop_signal: Arc<AtomicBool>,
) -> Result<()> {
    // ── 1. Descargar / localizar modelo ────────────────────────────────────
    let _ = tx.send(VideoMessage::Status("Verificando modelo...".into()));
    let model_path = Runtime::new()?
        .block_on(download_whisper_model(&model_name))?;

    // ── 2. Extraer audio con ffmpeg ────────────────────────────────────────
    let _ = tx.send(VideoMessage::Status("Extrayendo audio con ffmpeg...".into()));

    let mut child = Command::new("ffmpeg")
        .args(&[
            "-i", &file_path,
            "-ar", &WHISPER_SAMPLE_RATE.to_string(),
            "-ac", "1",      // mono
            "-f", "f32le",   // float 32-bit little-endian, sin cabecera
            "-vn",           // descartar stream de vídeo
            "pipe:1",        // enviar a stdout
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null()) // silenciar output de ffmpeg
        .spawn()
        .map_err(|e| anyhow!("Error iniciando ffmpeg: {:?}\n¿Está ffmpeg instalado?", e))?;

    let mut stdout = child.stdout.take()
        .ok_or_else(|| anyhow!("No se pudo obtener stdout de ffmpeg"))?;

    let mut audio_bytes = Vec::new();
    stdout.read_to_end(&mut audio_bytes)?;
    let _ = child.wait();

    if audio_bytes.is_empty() {
        return Err(anyhow!("ffmpeg no produjo audio. ¿Es un archivo de vídeo/audio válido?"));
    }

    // Convertir bytes a muestras f32
    let audio: Vec<f32> = audio_bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    let total_samples = audio.len();
    let total_secs = total_samples as f64 / WHISPER_SAMPLE_RATE as f64;

    let _ = tx.send(VideoMessage::Status(format!(
        "Audio extraído: {} ({} muestras). Cargando modelo...",
        format_timestamp(total_secs),
        total_samples,
    )));

    // ── 3. Cargar modelo Whisper ───────────────────────────────────────────
    let ctx = WhisperContext::new_with_params(&model_path, Default::default())
        .map_err(|e| anyhow!("Error cargando modelo: {:?}", e))?;
    let mut state = ctx.create_state()
        .map_err(|e| anyhow!("Error creando estado: {:?}", e))?;

    // ── 4. Transcribir chunk a chunk ───────────────────────────────────────
    let chunk_samples = (WHISPER_SAMPLE_RATE * VIDEO_CHUNK_SECS) as usize;
    let starts: Vec<usize> = (0..total_samples).step_by(chunk_samples).collect();
    let total_chunks = starts.len();

    for (chunk_idx, &chunk_start) in starts.iter().enumerate() {
        if stop_signal.load(Ordering::SeqCst) {
            let _ = tx.send(VideoMessage::Status("Transcripción cancelada.".into()));
            return Ok(());
        }

        let chunk_end = (chunk_start + chunk_samples).min(total_samples);
        let chunk = &audio[chunk_start..chunk_end];
        let time_offset_secs = chunk_start as f64 / WHISPER_SAMPLE_RATE as f64;

        let progress = (chunk_idx + 1) as f32 / total_chunks as f32;
        let _ = tx.send(VideoMessage::Progress(progress));
        let _ = tx.send(VideoMessage::Status(format!(
            "Fragmento {}/{} [{}]",
            chunk_idx + 1,
            total_chunks,
            format_timestamp(time_offset_secs),
        )));

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

        match state.full(params, chunk) {
            Ok(_) => {
                let n = state.full_n_segments();
                for i in 0..n {
                    if let Some(segment) = state.get_segment(i) {
                        let text = segment.to_string().trim().to_string();
                        if text.is_empty() || text.len() <= 1 {
                            continue;
                        }

                        let _ = tx.send(VideoMessage::Segment {
                            timestamp: format_timestamp(time_offset_secs),
                            text,
                        });
                    }
                }
            }
            Err(e) => eprintln!("Error en chunk {}: {:?}", chunk_idx, e),
        }
    }

    let _ = tx.send(VideoMessage::Progress(1.0));
    let _ = tx.send(VideoMessage::Done);
    Ok(())
}

fn format_timestamp(secs: f64) -> String {
    let h = (secs / 3600.0) as u64;
    let m = ((secs % 3600.0) / 60.0) as u64;
    let s = (secs % 60.0) as u64;
    if h > 0 {
        format!("{:02}:{:02}:{:02}", h, m, s)
    } else {
        format!("{:02}:{:02}", m, s)
    }
}