use anyhow::{Result, anyhow};
use std::io::Read;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::runtime::Runtime;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext};

use crate::audio::download_whisper_model;
use crate::data::{
    DiarizeConfig, DiarizeMode, LanguageConfig, TimelineEntry,
    VideoMessage, WHISPER_SAMPLE_RATE,
};
use crate::diarize::{
    agglomerative_cluster, download_diarize_model, lookup_speaker,
    DiarizeEngine, DEFAULT_DIARIZE_MODEL_URL,
    DIARIZE_OVERLAP_RATIO, DIARIZE_WINDOW_SECS,
};

/// Chunks de 30 segundos — ventana nativa de Whisper, calidad óptima.
const VIDEO_CHUNK_SECS: u32 = 30;

pub fn video_transcription_thread(
    file_path: String,
    model_name: String,
    lang_config: LanguageConfig,
    diarize_config: DiarizeConfig,
    tx: std::sync::mpsc::Sender<VideoMessage>,
    stop_signal: Arc<AtomicBool>,
) -> Result<()> {
    // ── 1. Descargar modelos ───────────────────────────────────────────────
    let _ = tx.send(VideoMessage::Status("Verificando modelo Whisper...".into()));
    let model_path = Runtime::new()?.block_on(download_whisper_model(&model_name))?;

    let mut diarize_engine: Option<DiarizeEngine> = if diarize_config.enabled {
        let _ = tx.send(VideoMessage::Status("Verificando modelo de diarización...".into()));
        match Runtime::new()?.block_on(download_diarize_model(DEFAULT_DIARIZE_MODEL_URL)) {
            Ok(diarize_path) => match DiarizeEngine::new(&diarize_path) {
                Ok(engine) => Some(engine),
                Err(e) => {
                    let _ = tx.send(VideoMessage::Status(format!(
                        "⚠️ No se pudo cargar modelo de diarización: {:?}. Continuando sin diarización.",
                        e
                    )));
                    None
                }
            },
            Err(e) => {
                let _ = tx.send(VideoMessage::Status(format!(
                    "⚠️ No se pudo descargar modelo de diarización: {:?}. Continuando sin diarización.",
                    e
                )));
                None
            }
        }
    } else {
        None
    };

    // ── 2. Extraer audio con ffmpeg ────────────────────────────────────────
    let _ = tx.send(VideoMessage::Status("Extrayendo audio con ffmpeg...".into()));

    let mut child = Command::new("ffmpeg")
        .args(&[
            "-i", &file_path,
            "-ar", &WHISPER_SAMPLE_RATE.to_string(),
            "-ac", "1",
            "-f", "f32le",
            "-vn",
            "pipe:1",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
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

    let audio: Vec<f32> = audio_bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    let total_samples = audio.len();
    let total_secs = total_samples as f64 / WHISPER_SAMPLE_RATE as f64;

    let _ = tx.send(VideoMessage::Status(format!(
        "Audio extraído: {} ({} muestras)",
        format_timestamp(total_secs),
        total_samples,
    )));

    if stop_signal.load(Ordering::SeqCst) { return Ok(()); }

    // ── 3. Diarización (si está habilitada) ────────────────────────────────
    let diarize_result: Option<(Vec<f64>, Vec<usize>)> = if let Some(ref mut engine) = diarize_engine {
        let _ = tx.send(VideoMessage::Status("Analizando hablantes...".into()));
        let _ = tx.send(VideoMessage::Progress(0.01));

        match engine.extract_windowed_embeddings(&audio, DIARIZE_WINDOW_SECS, DIARIZE_OVERLAP_RATIO) {
            Ok((timestamps, embeddings)) => {
                if embeddings.is_empty() {
                    let _ = tx.send(VideoMessage::Status(
                        "⚠️ No se pudieron extraer embeddings. Continuando sin diarización.".into()
                    ));
                    None
                } else {
                    let _ = tx.send(VideoMessage::Progress(0.05));
                    let _ = tx.send(VideoMessage::Status(format!(
                        "Clustering {} ventanas de audio...", embeddings.len()
                    )));

                    let (threshold, max_speakers) = match diarize_config.mode {
                        DiarizeMode::Auto => (diarize_config.threshold, None),
                        DiarizeMode::Manual => (0.0, Some(diarize_config.num_speakers)),
                    };

                    let labels = agglomerative_cluster(&embeddings, threshold, max_speakers);
                    let num_speakers = labels.iter().copied().max().map(|m| m + 1).unwrap_or(0);

                    // Construir timeline
                    let window_duration = DIARIZE_WINDOW_SECS;
                    let entries: Vec<TimelineEntry> = timestamps.iter().zip(labels.iter())
                        .map(|(&t, &speaker_id)| TimelineEntry {
                            start_secs: t,
                            end_secs: (t + window_duration).min(total_secs),
                            speaker_id,
                        })
                        .collect();

                    let _ = tx.send(VideoMessage::Timeline {
                        entries,
                        num_speakers,
                        total_duration: total_secs,
                    });

                    let _ = tx.send(VideoMessage::Status(format!(
                        "✅ {} hablantes detectados. Transcribiendo...", num_speakers
                    )));

                    Some((timestamps, labels))
                }
            }
            Err(e) => {
                let _ = tx.send(VideoMessage::Status(format!(
                    "⚠️ Error en diarización: {:?}. Continuando sin speaker labels.", e
                )));
                None
            }
        }
    } else {
        None
    };

    if stop_signal.load(Ordering::SeqCst) { return Ok(()); }

    // ── 4. Cargar Whisper ──────────────────────────────────────────────────
    let _ = tx.send(VideoMessage::Status("Cargando modelo Whisper...".into()));
    let ctx = WhisperContext::new_with_params(&model_path, Default::default())
        .map_err(|e| anyhow!("Error cargando modelo: {:?}", e))?;
    let mut state = ctx.create_state()
        .map_err(|e| anyhow!("Error creando estado: {:?}", e))?;

    // ── 5. Transcribir chunk a chunk ──────────────────────────────────────
    let chunk_samples = (WHISPER_SAMPLE_RATE * VIDEO_CHUNK_SECS) as usize;
    let starts: Vec<usize> = (0..total_samples).step_by(chunk_samples).collect();
    let total_chunks = starts.len();
    let mut segment_idx: usize = 0;

    // Progreso: 0.10–1.0 para Whisper (0.0–0.10 fue diarización)
    let progress_base = if diarize_engine.is_some() { 0.10 } else { 0.0 };
    let progress_range = 1.0 - progress_base;

    for (chunk_idx, &chunk_start) in starts.iter().enumerate() {
        if stop_signal.load(Ordering::SeqCst) {
            let _ = tx.send(VideoMessage::Status("Transcripción cancelada.".into()));
            return Ok(());
        }

        let chunk_end = (chunk_start + chunk_samples).min(total_samples);
        let chunk = &audio[chunk_start..chunk_end];
        let time_offset_secs = chunk_start as f64 / WHISPER_SAMPLE_RATE as f64;

        let progress = progress_base + progress_range * (chunk_idx + 1) as f32 / total_chunks as f32;
        let _ = tx.send(VideoMessage::Progress(progress));
        let _ = tx.send(VideoMessage::Status(format!(
            "Fragmento {}/{} [{}]",
            chunk_idx + 1, total_chunks, format_timestamp(time_offset_secs),
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

                        // Estimar timestamps del segmento Whisper dentro del chunk.
                        // whisper-rs 0.16 no expone t0/t1 directamente; estimamos
                        // distribuyendo los segmentos uniformemente en el chunk.
                        let n_segs = n as f64;
                        let chunk_duration = (chunk_end - chunk_start) as f64 / WHISPER_SAMPLE_RATE as f64;
                        let seg_duration = chunk_duration / n_segs.max(1.0);
                        let seg_start_secs = time_offset_secs + (i as f64 * seg_duration);
                        let duration_secs = seg_duration.max(0.5);

                        // Buscar speaker en la timeline de diarización
                        let speaker_id = diarize_result.as_ref().and_then(|(timestamps, labels)| {
                            lookup_speaker(
                                seg_start_secs,
                                timestamps,
                                labels,
                                DIARIZE_WINDOW_SECS,
                            )
                        });

                        let _ = tx.send(VideoMessage::Segment {
                            idx: segment_idx,
                            timestamp: format_timestamp(seg_start_secs),
                            time_secs: seg_start_secs,
                            duration_secs,
                            text,
                            speaker_id,
                        });

                        segment_idx += 1;
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