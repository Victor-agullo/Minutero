mod data;
mod audio;
mod diarize;
mod ui;
mod video;
mod system_audio;
use anyhow::Result;
use eframe::egui;
use crate::ui::TranscriptorApp;
use std::env;

fn main() -> Result<()> {
    // Silenciar mensajes de ALSA que no son errores reales.
    // En Rust edition 2024, env::set_var es unsafe porque puede causar UB si
    // se llama concurrentemente desde varios hilos. Aquí es seguro: se llama
    // al inicio de main(), antes de que se cree ningún hilo.
    // SAFETY: single-threaded at this point — no threads exist yet.
    unsafe {
        env::set_var("ALSA_CONFIG_PATH", "/dev/null");
        env::set_var("ALSA_PLUGIN_DIR", "/dev/null");
    }

    // NO suprimimos stderr globalmente: eso escondía los eprintln! de depuración
    // de diarize.rs y video.rs, haciendo imposible diagnosticar errores.
    // Si el spam de ALSA molesta en producción, compilar con:
    //   RUST_LOG=error cargo run --release
    // o filtrar con: cargo run 2>&1 | grep -v ALSA

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([700.0, 600.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Minutador de Transcripción Multicanal",
        options,
        Box::new(|_cc| Ok(Box::new(TranscriptorApp::default()))),
    ).map_err(|e| anyhow::anyhow!("Error en eframe: {:?}", e))
}