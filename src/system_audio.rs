use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait};
use crate::data::DeviceInfo;

#[derive(Debug, Clone, PartialEq)]
pub enum LoopbackStatus {
    Available,
    NeedsConfiguration,
    RequiresSetup,
    Unsupported,
}

#[derive(Debug, Clone)]
pub struct LoopbackInfo {
    pub status: LoopbackStatus,
    pub message: String,
    pub instructions: Vec<String>,
    pub loopback_devices: Vec<DeviceInfo>,
}

pub fn detect_os() -> &'static str {
    if cfg!(target_os = "windows") { "windows" }
    else if cfg!(target_os = "linux") { "linux" }
    else if cfg!(target_os = "macos") { "macos" }
    else { "unknown" }
}

pub fn check_loopback_status() -> Result<LoopbackInfo> {
    match detect_os() {
        "windows" => Ok(check_windows_loopback()),
        "linux"   => Ok(check_linux_loopback()),
        "macos"   => Ok(check_macos_loopback()),
        _ => Ok(LoopbackInfo {
            status: LoopbackStatus::Unsupported,
            message: "Sistema operativo no soportado".to_string(),
            instructions: vec![],
            loopback_devices: vec![],
        }),
    }
}

// ── Windows ───────────────────────────────────────────────────────────────
//
// En WASAPI, los dispositivos loopback (Stereo Mix, What U Hear…)
// aparecen directamente en la lista de inputs de cpal cuando están
// habilitados en el Panel de Sonido. No necesitamos PowerShell.

fn check_windows_loopback() -> LoopbackInfo {
    let devices = get_windows_loopback_devices();

    if !devices.is_empty() {
        LoopbackInfo {
            status: LoopbackStatus::Available,
            message: format!("✅ {} dispositivos de captura de sistema detectados", devices.len()),
            instructions: vec![
                "Dispositivos disponibles para capturar audio del sistema:".into(),
                "".into(),
                "Úsalos en Configuración como fuentes de tipo SALIDA.".into(),
            ],
            loopback_devices: devices,
        }
    } else {
        LoopbackInfo {
            status: LoopbackStatus::NeedsConfiguration,
            message: "⚠️ No se detecta 'Stereo Mix' u otro dispositivo loopback".to_string(),
            instructions: vec![
                "Para capturar el audio del sistema en Windows:".into(),
                "".into(),
                "1. Click derecho en el icono de volumen (barra de tareas)".into(),
                "2. Selecciona 'Configuración de sonido' → 'Más opciones de sonido'".into(),
                "3. Pestaña 'Grabación' → click derecho → 'Mostrar dispositivos deshabilitados'".into(),
                "4. Busca 'Mezcla estéreo' o 'Stereo Mix' → click derecho → 'Habilitar'".into(),
                "".into(),
                "Si no aparece Stereo Mix, tu tarjeta de sonido no lo soporta.".into(),
                "Alternativa: instala 'VB-Audio Virtual Cable' (gratuito):".into(),
                "  https://vb-audio.com/Cable/".into(),
            ],
            loopback_devices: vec![],
        }
    }
}

fn get_windows_loopback_devices() -> Vec<DeviceInfo> {
    // WASAPI expone Stereo Mix / What U Hear / Wave Out Mix como inputs normales
    enumerate_loopback_inputs(&[
        "stereo mix", "mezcla estéreo", "what u hear",
        "wave out mix", "loopback", "virtual cable", "vb-audio",
        "cable output", // VB-Audio Cable
    ])
}

// ── Linux ─────────────────────────────────────────────────────────────────

fn check_linux_loopback() -> LoopbackInfo {
    use std::process::Command;

    let audio_sys = {
        let out = Command::new("pactl").args(&["info"]).output().ok();
        out.and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| if s.to_lowercase().contains("pipewire") { "PipeWire" } else { "PulseAudio" })
            .unwrap_or("PulseAudio")
            .to_string()
    };

    let devices = get_linux_loopback_devices();

    if !devices.is_empty() {
        LoopbackInfo {
            status: LoopbackStatus::Available,
            message: format!("✅ {} — {} dispositivos monitor detectados", audio_sys, devices.len()),
            instructions: vec![
                "Los dispositivos '.monitor' capturan el audio de salida del sistema.".into(),
                "Úsalos en Configuración como fuentes de tipo SALIDA.".into(),
            ],
            loopback_devices: devices,
        }
    } else {
        LoopbackInfo {
            status: LoopbackStatus::RequiresSetup,
            message: format!("⚠️ {}: no se detectan dispositivos monitor", audio_sys),
            instructions: vec![
                "Verifica que tu tarjeta de audio esté activa:".into(),
                "  pactl list sinks short".into(),
                "".into(),
                "Los dispositivos '.monitor' deberían aparecer automáticamente.".into(),
            ],
            loopback_devices: vec![],
        }
    }
}

pub fn get_linux_loopback_devices() -> Vec<DeviceInfo> {
    use std::process::Command;

    let mut devices = vec![];

    let output = Command::new("pactl").args(&["list", "sources", "short"]).output();

    if let Ok(out) = output {
        let sources = String::from_utf8_lossy(&out.stdout);

        for line in sources.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 2 { continue; }
            let tech_name = parts[1].to_string();

            if tech_name.contains(".monitor") || tech_name.contains("Monitor") {
                let mut description = tech_name.clone();

                if let Ok(desc_out) = Command::new("pactl").args(&["list", "sources"]).output() {
                    let full = String::from_utf8_lossy(&desc_out.stdout);
                    let mut found = false;
                    for dl in full.lines() {
                        if dl.contains(&format!("Name: {}", tech_name)) {
                            found = true;
                        } else if found && dl.trim().starts_with("Description:") {
                            description = dl.replace("Description:", "").trim().to_string();
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

    devices
}

// ── macOS ─────────────────────────────────────────────────────────────────
//
// CoreAudio no tiene loopback nativo. BlackHole o Soundflower se instalan
// como drivers de audio virtuales y aparecen como inputs en cpal.

fn check_macos_loopback() -> LoopbackInfo {
    let devices = get_macos_loopback_devices();

    if !devices.is_empty() {
        LoopbackInfo {
            status: LoopbackStatus::Available,
            message: format!("✅ {} dispositivos de audio virtual detectados", devices.len()),
            instructions: vec![
                "Asegúrate de que las aplicaciones envíen el audio a este dispositivo virtual.".into(),
                "En la mayoría de casos se configura con un 'Dispositivo Agregado' en Audio MIDI Setup.".into(),
            ],
            loopback_devices: devices,
        }
    } else {
        LoopbackInfo {
            status: LoopbackStatus::RequiresSetup,
            message: "⚠️ macOS requiere software de audio virtual".to_string(),
            instructions: vec![
                "macOS no tiene captura de salida nativa.".into(),
                "".into(),
                "Opción recomendada — BlackHole (gratuito):".into(),
                "  https://github.com/ExistentialAudio/BlackHole".into(),
                "".into(),
                "Instalación:".into(),
                "  1. Descarga e instala BlackHole 2ch".into(),
                "  2. Abre 'Audio MIDI Setup' (en /Aplicaciones/Utilidades/)".into(),
                "  3. Crea un 'Dispositivo Agregado' con tu salida habitual + BlackHole".into(),
                "  4. Úsalo como salida del sistema en Preferencias de Sonido".into(),
                "  5. BlackHole aparecerá aquí como dispositivo disponible".into(),
                "".into(),
                "Alternativa de pago — Loopback (Rogue Amoeba):".into(),
                "  https://rogueamoeba.com/loopback/".into(),
            ],
            loopback_devices: vec![],
        }
    }
}

fn get_macos_loopback_devices() -> Vec<DeviceInfo> {
    enumerate_loopback_inputs(&[
        "blackhole", "soundflower", "loopback",
        "virtual", "aggregate", // Dispositivo Agregado de Audio MIDI Setup
    ])
}

// ── Helper compartido ─────────────────────────────────────────────────────

/// Busca en los inputs de cpal dispositivos cuyo nombre (en minúsculas)
/// contenga alguna de las palabras clave dadas.
fn enumerate_loopback_inputs(keywords: &[&str]) -> Vec<DeviceInfo> {
    let host = cpal::default_host();
    let mut devices = vec![];

    if let Ok(inputs) = host.input_devices() {
        for device in inputs {
            if let Ok(desc) = device.description() {
                let name = desc.name().to_string();
                let lower = name.to_lowercase();
                if keywords.iter().any(|kw| lower.contains(kw)) {
                    devices.push(DeviceInfo {
                        id: devices.len(),
                        name: name.clone(),
                        technical_name: Some(name),
                    });
                }
            }
        }
    }

    devices
}

// ── API pública ───────────────────────────────────────────────────────────

pub fn get_loopback_devices() -> Vec<DeviceInfo> {
    match detect_os() {
        "linux"   => get_linux_loopback_devices(),
        "windows" => get_windows_loopback_devices(),
        "macos"   => get_macos_loopback_devices(),
        _         => vec![],
    }
}