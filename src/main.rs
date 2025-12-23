mod data;
mod audio;
mod ui;
mod system_audio;
use anyhow::Result;
use eframe::egui;
use crate::ui::TranscriptorApp;
use std::env;

fn main() -> Result<()> {
    // Silenciar warnings de ALSA sobre backends no disponibles
    env::set_var("ALSA_CONFIG_PATH", "/dev/null");
    
    // Alternativamente, redirigir stderr de ALSA a /dev/null en sistemas Unix
    #[cfg(target_os = "linux")]
    {
        use std::fs::OpenOptions;
        use std::os::unix::io::AsRawFd;
        
        if let Ok(null) = OpenOptions::new().write(true).open("/dev/null") {
            unsafe {
                libc::dup2(null.as_raw_fd(), libc::STDERR_FILENO);
            }
        }
    }
    
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