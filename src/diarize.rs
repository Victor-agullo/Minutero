// ── diarize.rs ─────────────────────────────────────────────────────────────
//
// Speaker diarization via ONNX Runtime + speaker embedding model.
//
// Pipeline:
//   1. Mel-filterbank feature extraction (80-dim log fbank)
//   2. Speaker embedding via ECAPA-TDNN / ResNet (ONNX)
//   3. Agglomerative clustering (batch/vídeo) ó incremental (real-time)
//
// Modelo ONNX esperado:
//   - Input:  "feats" shape [1, T, 80]  (T = nº de frames fbank)
//   - Output: "embs"  shape [1, D]      (D = dimensión del embedding)
//
// Modelos compatibles (descargar el .onnx y colocar en models/):
//   · Wespeaker ECAPA-TDNN / ResNet34  (recomendado)
//   · 3D-Speaker ERes2Net / ECAPA
//   · NVIDIA TitaNet (exportado a ONNX)
//
// ────────────────────────────────────────────────────────────────────────────

use anyhow::{Result, anyhow};
use ndarray::{Array2, Axis};
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Tensor;
use rustfft::{FftPlanner, num_complex::Complex};
use std::io::Write;
use std::path::Path;

use futures_util::StreamExt;
use reqwest::Client;

// ── Constantes ─────────────────────────────────────────────────────────────

/// URL del modelo ONNX de embeddings de voz.
/// Wespeaker ResNet293-LM — entrenado en VoxCeleb (inglés/multilingüe europeo).
/// Mucho más preciso que el ResNet34 (~300 MB vs 45 MB).
///
/// Descarga manual si falla la automática:
///   https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet293-LM/tree/main
///   → descarga voxceleb_resnet293_LM.onnx → renombra a models/speaker_embedding.onnx
pub const DEFAULT_DIARIZE_MODEL_URL: &str =
    "https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet293-LM/resolve/main/voxceleb_resnet293_LM.onnx";
pub const DIARIZE_MODEL_FILE: &str = "speaker_embedding.onnx";

// Parámetros de extracción de features (estándar Kaldi/Wespeaker)
const N_MELS: usize = 80;
const FFT_SIZE: usize = 512;
const FRAME_LEN_SAMPLES: usize = 400;   // 25 ms @ 16 kHz
const FRAME_SHIFT_SAMPLES: usize = 160; // 10 ms @ 16 kHz
const PRE_EMPHASIS_COEFF: f32 = 0.97;
const MEL_LOW_FREQ: f32 = 20.0;
const MEL_HIGH_FREQ: f32 = 0.0; // 0 = Nyquist (8000 Hz @ 16 kHz)
const SAMPLE_RATE: u32 = 16000;

/// Ventana de diarización: 3 s por fragmento.
pub const DIARIZE_WINDOW_SECS: f64 = 3.0;
/// Solapamiento 50 % (1.5 s).
pub const DIARIZE_OVERLAP_RATIO: f64 = 0.5;
/// Umbral de similitud coseno por defecto para auto-detección.
pub const DEFAULT_COSINE_THRESHOLD: f32 = 0.60;
/// Mínimo de muestras para extraer embedding (~1 s).
pub const MIN_EMBEDDING_SAMPLES: usize = SAMPLE_RATE as usize;
/// Máximo de ventanas que se pasan al clustering aglomerativo.
/// Con n ventanas el clustering es O(n³); 1000 ventanas → ~1000M ops.
/// Las ventanas sobrantes se reasignan por similitud coseno al centroide más cercano.
pub const MAX_CLUSTER_WINDOWS: usize = 1000;

// ── Mel-filterbank features ───────────────────────────────────────────────

fn pre_emphasis(signal: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(signal.len());
    out.push(signal[0]);
    for i in 1..signal.len() {
        out.push(signal[i] - PRE_EMPHASIS_COEFF * signal[i - 1]);
    }
    out
}

fn hamming_window(size: usize) -> Vec<f32> {
    use std::f32::consts::PI;
    (0..size)
        .map(|n| 0.54 - 0.46 * (2.0 * PI * n as f32 / (size - 1) as f32).cos())
        .collect()
}

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

/// Construye un banco de filtros mel triangulares.
fn build_mel_filterbank(
    n_mels: usize,
    fft_size: usize,
    sample_rate: u32,
    low_freq: f32,
    high_freq: f32,
) -> Vec<Vec<f32>> {
    let n_fft_bins = fft_size / 2 + 1;
    let high = if high_freq <= 0.0 { sample_rate as f32 / 2.0 } else { high_freq };

    let mel_low = hz_to_mel(low_freq);
    let mel_high = hz_to_mel(high);

    let mel_points: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_low + i as f32 * (mel_high - mel_low) / (n_mels + 1) as f32)
        .collect();

    let bin_points: Vec<usize> = mel_points
        .iter()
        .map(|&m| ((fft_size as f32 + 1.0) * mel_to_hz(m) / sample_rate as f32).floor() as usize)
        .collect();

    let mut filters = vec![vec![0.0f32; n_fft_bins]; n_mels];

    for i in 0..n_mels {
        let (start, center, end) = (bin_points[i], bin_points[i + 1], bin_points[i + 2]);
        for j in start..center {
            if j < n_fft_bins && center > start {
                filters[i][j] = (j - start) as f32 / (center - start) as f32;
            }
        }
        for j in center..end {
            if j < n_fft_bins && end > center {
                filters[i][j] = (end - j) as f32 / (end - center) as f32;
            }
        }
    }
    filters
}

/// Calcula log mel-filterbank features a partir de audio PCM 16 kHz mono.
pub fn compute_fbank(audio: &[f32]) -> Array2<f32> {
    let signal = pre_emphasis(audio);
    let window = hamming_window(FRAME_LEN_SAMPLES);
    let filters = build_mel_filterbank(N_MELS, FFT_SIZE, SAMPLE_RATE, MEL_LOW_FREQ, MEL_HIGH_FREQ);

    let n_frames = if signal.len() > FRAME_LEN_SAMPLES {
        (signal.len() - FRAME_LEN_SAMPLES) / FRAME_SHIFT_SAMPLES + 1
    } else {
        1
    };

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);

    let mut fbank = Array2::<f32>::zeros((n_frames, N_MELS));

    for fi in 0..n_frames {
        let start = fi * FRAME_SHIFT_SAMPLES;

        // Ventana + zero-pad → FFT
        let mut buffer: Vec<Complex<f32>> = (0..FFT_SIZE)
            .map(|i| {
                if i < FRAME_LEN_SAMPLES && start + i < signal.len() {
                    Complex::new(signal[start + i] * window[i], 0.0)
                } else {
                    Complex::new(0.0, 0.0)
                }
            })
            .collect();

        fft.process(&mut buffer);

        // Espectro de potencia (mitad positiva)
        let power: Vec<f32> = buffer[..FFT_SIZE / 2 + 1]
            .iter()
            .map(|c| c.norm_sqr())
            .collect();

        // Aplicar filtros mel + log
        for (mel_idx, filter) in filters.iter().enumerate() {
            let energy: f32 = filter.iter().zip(power.iter()).map(|(f, p)| f * p).sum();
            fbank[[fi, mel_idx]] = energy.max(1e-10).ln();
        }
    }

    // CMVN: normalización de media cepstral
    if let Some(mean) = fbank.mean_axis(Axis(0)) {
        for mut row in fbank.rows_mut() {
            row -= &mean;
        }
    }

    fbank
}

// ── Motor de embeddings ONNX ──────────────────────────────────────────────

pub struct DiarizeEngine {
    session: Session,
    /// None = todavía no determinado (se detecta en la primera llamada).
    /// Some(true)  → el modelo espera [1, 80, T]  (Wespeaker ResNet, ECAPA…)
    /// Some(false) → el modelo espera [1, T, 80]  (algunos exports alternativos)
    mels_first: Option<bool>,
}

// Seguridad para uso en múltiples hilos
unsafe impl Send for DiarizeEngine {}
unsafe impl Sync for DiarizeEngine {}

/// Ejecuta el modelo ONNX con el tensor dado y devuelve el embedding como Vec<f32>.
/// Función libre para evitar conflictos de préstamo: toma &mut Session directamente,
/// de modo que el préstamo termina al retornar (antes de que el llamador
/// modifique otros campos de DiarizeEngine).
fn onnx_run_extract(session: &mut Session, tensor: Tensor<f32>) -> Option<Vec<f32>> {
    let outputs = session.run(ort::inputs![tensor]).ok()?;
    let emb = outputs[0].try_extract_tensor::<f32>().ok()?;
    Some(emb.1.iter().copied().collect())
}

impl DiarizeEngine {
    /// Carga el modelo ONNX desde disco.
    /// El formato de entrada ([1,80,T] ó [1,T,80]) se auto-detecta
    /// la primera vez que se llama a extract_embedding().
    pub fn new(model_path: &str) -> Result<Self> {
        let session = Session::builder()
            .map_err(|e| anyhow!("Error creando session builder: {:?}", e))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow!("Error configurando optimización: {:?}", e))?
            .commit_from_file(model_path)
            .map_err(|e| anyhow!("Error cargando modelo diarización ONNX: {:?}", e))?;
        Ok(Self { session, mels_first: None })
    }

    /// Extrae un vector de embedding de voz a partir de audio PCM 16 kHz mono.
    /// El audio debe tener al menos 1 segundo (~16 000 muestras).
    ///
    /// En la primera llamada auto-detecta el shape del tensor ([1,80,T] ó [1,T,80])
    /// probando ambos y cacheando el que funcione.
    pub fn extract_embedding(&mut self, audio: &[f32]) -> Result<Vec<f32>> {
        if audio.len() < MIN_EMBEDDING_SAMPLES {
            return Err(anyhow!(
                "Audio demasiado corto para embedding: {} muestras (mínimo {})",
                audio.len(),
                MIN_EMBEDDING_SAMPLES
            ));
        }

        let fbank = compute_fbank(audio);
        let (n_frames, n_mels) = fbank.dim();

        // ── Fase de auto-detección (solo primera llamada) ──────────────────
        if self.mels_first.is_none() {
            // Intento 1: [1, 80, T]  (Wespeaker ResNet/ECAPA, la mayoría)
            if let Ok(tensor) = Tensor::from_array(([1usize, n_mels, n_frames],
                fbank.t().to_owned().into_raw_vec_and_offset().0))
            {
                // run_and_extract es una función libre que toma &mut Session
                // y devuelve Vec<f32> ya copiado → el préstamo de session
                // termina antes de que modifiquemos self.mels_first.
                if let Some(emb) = onnx_run_extract(&mut self.session, tensor) {
                    eprintln!("[diarize] Shape auto-detectado: [1, 80, T]");
                    self.mels_first = Some(true);
                    return Ok(emb);
                }
            }
            // Intento 2: [1, T, 80]
            if let Ok(tensor) = Tensor::from_array(([1usize, n_frames, n_mels],
                fbank.clone().into_raw_vec_and_offset().0))
            {
                if let Some(emb) = onnx_run_extract(&mut self.session, tensor) {
                    eprintln!("[diarize] Shape auto-detectado: [1, T, 80]");
                    self.mels_first = Some(false);
                    return Ok(emb);
                }
            }
            return Err(anyhow!(
                "El modelo ONNX rechazó ambos shapes de entrada ([1,80,T] y [1,T,80]).                  Comprueba que el modelo sea compatible (Wespeaker ResNet/ECAPA)."
            ));
        }

        // ── Llamadas siguientes: shape ya conocido ─────────────────────────
        let tensor = if self.mels_first == Some(true) {
            Tensor::from_array(([1usize, n_mels, n_frames],
                fbank.t().to_owned().into_raw_vec_and_offset().0))
        } else {
            Tensor::from_array(([1usize, n_frames, n_mels],
                fbank.into_raw_vec_and_offset().0))
        }.map_err(|e| anyhow!("Error creando tensor ONNX: {:?}", e))?;

        onnx_run_extract(&mut self.session, tensor)
            .ok_or_else(|| anyhow!("Error ejecutando modelo ONNX o extrayendo embedding"))
    }

    /// Extrae embeddings para múltiples ventanas de audio.
    /// El paso se adapta automáticamente para que el total de ventanas nunca
    /// supere MAX_CLUSTER_WINDOWS, evitando el submuestreo posterior y
    /// manteniendo el clustering aglomerativo exacto sobre las ventanas reales.
    /// Devuelve (timestamps_secs, embeddings).
    pub fn extract_windowed_embeddings(
        &mut self,
        audio: &[f32],
        window_secs: f64,
        overlap_ratio: f64,
    ) -> Result<(Vec<f64>, Vec<Vec<f32>>)> {
        let window_samples = (window_secs * SAMPLE_RATE as f64) as usize;
        let total = audio.len();

        // Calcular cuántas ventanas produciría el paso por defecto
        let default_step = ((1.0 - overlap_ratio) * window_samples as f64) as usize;
        let default_count = if total > window_samples {
            (total - window_samples) / default_step + 1
        } else { 1 };

        // Si hay demasiadas, ampliar el paso para quedarnos en MAX_CLUSTER_WINDOWS
        let step_samples = if default_count > MAX_CLUSTER_WINDOWS {
            (total as f64 / MAX_CLUSTER_WINDOWS as f64) as usize
        } else {
            default_step
        };

        eprintln!(
            "[diarize] Audio {:.1}s → {} ventanas (paso {:.1}s)",
            total as f64 / SAMPLE_RATE as f64,
            default_count.min(MAX_CLUSTER_WINDOWS),
            step_samples as f64 / SAMPLE_RATE as f64,
        );

        let mut timestamps = Vec::new();
        let mut embeddings = Vec::new();
        let mut start = 0;

        while start < total {
            let end = (start + window_samples).min(total);
            if end - start < MIN_EMBEDDING_SAMPLES { break; }

            let chunk = &audio[start..end];
            let time_secs = start as f64 / SAMPLE_RATE as f64;

            match self.extract_embedding(chunk) {
                Ok(emb) => { timestamps.push(time_secs); embeddings.push(emb); }
                Err(e) => { eprintln!("Embedding fallido en t={:.1}s: {:?}", time_secs, e); }
            }

            start += step_samples;
        }

        Ok((timestamps, embeddings))
    }
}

// ── Similitud coseno ──────────────────────────────────────────────────────

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-8 || norm_b < 1e-8 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

// ── Clustering aglomerativo (batch — vídeo) ──────────────────────────────

/// Agrupa embeddings en clusters de hablantes.
///
/// - `threshold`: similitud coseno mínima para fusionar (0.55 – 0.70 típico).
/// - `max_speakers`: si es `Some(n)`, detiene el clustering al llegar a n clusters.
///
/// Para vídeos largos (> MAX_CLUSTER_WINDOWS ventanas) submuestrea antes de
/// agrupar y luego reasigna el resto por similitud al centroide más cercano.
/// Esto reduce la complejidad de O(n³) a O(MAX_CLUSTER_WINDOWS³ + n·k).
///
/// Devuelve un vector de etiquetas (0, 1, 2…) del mismo tamaño que `embeddings`.
pub fn agglomerative_cluster(
    embeddings: &[Vec<f32>],
    threshold: f32,
    max_speakers: Option<usize>,
) -> Vec<usize> {
    let n = embeddings.len();
    if n == 0 { return vec![]; }
    if n == 1 { return vec![0]; }

    // ── Submuestreo si hay demasiadas ventanas ────────────────────────────
    let (sample_indices, sample_embeddings): (Vec<usize>, Vec<Vec<f32>>) =
        if n > MAX_CLUSTER_WINDOWS {
            let step = n as f64 / MAX_CLUSTER_WINDOWS as f64;
            let idx: Vec<usize> = (0..MAX_CLUSTER_WINDOWS)
                .map(|i| ((i as f64 * step) as usize).min(n - 1))
                .collect();
            let emb = idx.iter().map(|&i| embeddings[i].clone()).collect();
            (idx, emb)
        } else {
            ((0..n).collect(), embeddings.to_vec())
        };

    let m = sample_embeddings.len();

    // ── Clustering aglomerativo sobre la muestra ──────────────────────────
    let mut labels: Vec<usize> = (0..m).collect();
    let mut centroids: Vec<Vec<f32>> = sample_embeddings.clone();
    let mut sizes: Vec<f32> = vec![1.0; m];
    let mut active: Vec<bool> = vec![true; m];

    loop {
        let n_active = active.iter().filter(|&&a| a).count();
        if n_active <= 1 { break; }
        if let Some(max) = max_speakers {
            if n_active <= max { break; }
        }

        let mut best_sim = f32::NEG_INFINITY;
        let mut best_i = 0;
        let mut best_j = 0;

        for i in 0..m {
            if !active[i] { continue; }
            for j in (i + 1)..m {
                if !active[j] { continue; }
                let sim = cosine_similarity(&centroids[i], &centroids[j]);
                if sim > best_sim {
                    best_sim = sim;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        if best_sim < threshold { break; }

        let si = sizes[best_i];
        let sj = sizes[best_j];
        let total = si + sj;
        for k in 0..centroids[best_i].len() {
            centroids[best_i][k] =
                (centroids[best_i][k] * si + centroids[best_j][k] * sj) / total;
        }
        sizes[best_i] = total;
        active[best_j] = false;
        let old_label = best_j;
        for l in labels.iter_mut() {
            if *l == old_label { *l = best_i; }
        }
    }

    // Renumerar etiquetas de la muestra a 0,1,2…
    let mut unique: Vec<usize> = labels.clone();
    unique.sort_unstable();
    unique.dedup();
    let sample_labels: Vec<usize> = labels.iter()
        .map(|&l| unique.iter().position(|&u| u == l).unwrap_or(0))
        .collect();

    // Centroides finales de cada cluster (para reasignación)
    let n_clusters = unique.len();
    let dim = centroids[0].len();
    let mut final_centroids = vec![vec![0.0f32; dim]; n_clusters];
    let mut final_counts = vec![0usize; n_clusters];
    for (i, &cl) in sample_labels.iter().enumerate() {
        for (d, v) in final_centroids[cl].iter_mut().zip(&sample_embeddings[i]) {
            *d += v;
        }
        final_counts[cl] += 1;
    }
    for (centroid, &count) in final_centroids.iter_mut().zip(&final_counts) {
        if count > 0 {
            for v in centroid.iter_mut() { *v /= count as f32; }
        }
    }

    // ── Si no hubo submuestreo, devolver ya ───────────────────────────────
    if n == m {
        return sample_labels;
    }

    // ── Reasignar TODAS las ventanas al centroide más cercano ─────────────
    // Primero construir mapa índice_muestra → label
    let mut sample_label_map = vec![0usize; n];
    for (pos, &orig_idx) in sample_indices.iter().enumerate() {
        sample_label_map[orig_idx] = sample_labels[pos];
    }

    let mut full_labels = vec![0usize; n];
    for i in 0..n {
        let best = final_centroids.iter().enumerate()
            .max_by(|(_, a), (_, b)| {
                cosine_similarity(&embeddings[i], a)
                    .partial_cmp(&cosine_similarity(&embeddings[i], b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        full_labels[i] = best;
    }
    full_labels
}

// ── Clustering incremental (real-time — audio en vivo) ────────────────────

pub struct IncrementalClusterer {
    centroids: Vec<Vec<f32>>,
    counts: Vec<usize>,
    threshold: f32,
}

impl IncrementalClusterer {
    pub fn new(threshold: f32) -> Self {
        Self {
            centroids: Vec::new(),
            counts: Vec::new(),
            threshold,
        }
    }

    /// Asigna un embedding al hablante más cercano o crea uno nuevo.
    /// Devuelve el ID del hablante (0, 1, 2…).
    pub fn assign(&mut self, embedding: &[f32]) -> usize {
        let mut best_sim = f32::NEG_INFINITY;
        let mut best_idx = 0;

        for (i, centroid) in self.centroids.iter().enumerate() {
            let sim = cosine_similarity(embedding, centroid);
            if sim > best_sim {
                best_sim = sim;
                best_idx = i;
            }
        }

        if self.centroids.is_empty() || best_sim < self.threshold {
            // Nuevo hablante
            let idx = self.centroids.len();
            self.centroids.push(embedding.to_vec());
            self.counts.push(1);
            idx
        } else {
            // Actualizar centroide (media acumulativa)
            let count = self.counts[best_idx] as f32;
            let new_count = count + 1.0;
            for (c, &e) in self.centroids[best_idx].iter_mut().zip(embedding) {
                *c = (*c * count + e) / new_count;
            }
            self.counts[best_idx] += 1;
            best_idx
        }
    }

    #[allow(dead_code)]
    pub fn num_speakers(&self) -> usize {
        self.centroids.len()
    }

    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.centroids.clear();
        self.counts.clear();
    }
}

// ── Mapeo de segmentos Whisper → hablantes ────────────────────────────────

/// Dado un instante (en segundos), busca el hablante dominante en la
/// timeline de diarización (lista de ventanas con speaker asignado).
pub fn lookup_speaker(
    time_secs: f64,
    window_times: &[f64],
    window_labels: &[usize],
    window_secs: f64,
) -> Option<usize> {
    for (i, &wt) in window_times.iter().enumerate() {
        let wend = wt + window_secs;
        if time_secs >= wt && time_secs < wend {
            return Some(window_labels[i]);
        }
    }
    // Fallback: ventana más cercana
    window_times
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            let da = (time_secs - **a).abs();
            let db = (time_secs - **b).abs();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| window_labels[i])
}

// ── Descarga del modelo ───────────────────────────────────────────────────

/// Descarga el modelo ONNX de embeddings de voz.
/// Reutiliza el mismo directorio `models/` que Whisper.
pub async fn download_diarize_model(url: &str) -> Result<String> {
    let models_dir = Path::new("models");
    let model_path = models_dir.join(DIARIZE_MODEL_FILE);

    if !models_dir.exists() {
        std::fs::create_dir_all(models_dir)?;
    }

    if model_path.exists() {
        return Ok(model_path.to_string_lossy().to_string());
    }

    println!("📥 Descargando modelo de diarización...");

    let client = Client::new();
    let response = client.get(url).send().await?;

    if !response.status().is_success() {
        return Err(anyhow!(
            "Error descargando modelo de diarización: HTTP {}\n\
             URL: {}\n\n\
             Descarga el modelo manualmente y colócalo en:\n  {}\n\n\
             Modelos compatibles:\n\
             · Wespeaker:   https://huggingface.co/Wespeaker\n\
             · 3D-Speaker:  https://github.com/alibaba-damo-academy/3D-Speaker\n\
             · NVIDIA NeMo: https://catalog.ngc.nvidia.com (TitaNet)",
            response.status(),
            url,
            model_path.display()
        ));
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
            print!(
                "\r   Diarización: {:.1}% ({}/{} MB)",
                (downloaded as f64 / total as f64) * 100.0,
                downloaded / 1_000_000,
                total / 1_000_000
            );
            std::io::stdout().flush()?;
        }
    }

    println!("\n✓ Modelo de diarización descargado");
    Ok(model_path.to_string_lossy().to_string())
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-5);
    }

    #[test]
    fn test_fbank_shape() {
        // 1 segundo de silencio @ 16 kHz
        let audio = vec![0.001; 16000];
        let fbank = compute_fbank(&audio);
        let (frames, mels) = fbank.dim();
        assert_eq!(mels, 80);
        // ~100 frames por segundo (10 ms de hop)
        assert!(frames > 90 && frames < 110, "frames = {}", frames);
    }

    #[test]
    fn test_agglomerative_cluster_identical() {
        let emb = vec![1.0, 0.0, 0.0];
        let embeddings = vec![emb.clone(), emb.clone(), emb.clone()];
        let labels = agglomerative_cluster(&embeddings, 0.5, None);
        assert!(labels.iter().all(|&l| l == labels[0]));
    }

    #[test]
    fn test_agglomerative_cluster_distinct() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let embeddings = vec![a.clone(), b.clone(), a.clone(), b.clone()];
        let labels = agglomerative_cluster(&embeddings, 0.5, None);
        assert_eq!(labels[0], labels[2]); // a-a mismo cluster
        assert_eq!(labels[1], labels[3]); // b-b mismo cluster
        assert_ne!(labels[0], labels[1]); // a≠b
    }

    #[test]
    fn test_incremental_clusterer() {
        let mut c = IncrementalClusterer::new(0.5);
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];

        let s0 = c.assign(&a);
        let s1 = c.assign(&b);
        let s2 = c.assign(&a); // debería reasignar a s0

        assert_eq!(s0, 0);
        assert_eq!(s1, 1);
        assert_eq!(s2, 0);
        assert_eq!(c.num_speakers(), 2);
    }

    #[test]
    fn test_lookup_speaker() {
        let times = vec![0.0, 1.5, 3.0, 4.5];
        let labels = vec![0, 1, 0, 1];
        assert_eq!(lookup_speaker(0.5, &times, &labels, 1.5), Some(0));
        assert_eq!(lookup_speaker(2.0, &times, &labels, 1.5), Some(1));
        assert_eq!(lookup_speaker(3.5, &times, &labels, 1.5), Some(0));
    }
}