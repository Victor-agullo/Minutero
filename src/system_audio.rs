use anyhow::{Result};
use std::process::Command;
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

pub fn detect_os() -> String {
    if cfg!(target_os = "windows") {
        "windows".to_string()
    } else if cfg!(target_os = "linux") {
        "linux".to_string()
    } else if cfg!(target_os = "macos") {
        "macos".to_string()
    } else {
        "unknown".to_string()
    }
}

fn detect_audio_system() -> String {
    let output = Command::new("pactl")
        .args(&["info"])
        .output();
    
    if let Ok(out) = output {
        let info = String::from_utf8_lossy(&out.stdout);
        if info.to_lowercase().contains("pipewire") {
            return "pipewire".to_string();
        }
        return "pulseaudio".to_string();
    }
    
    "unknown".to_string()
}

pub fn check_loopback_status() -> Result<LoopbackInfo> {
    let os = detect_os();
    
    match os.as_str() {
        "windows" => check_windows_loopback(),
        "linux" => check_linux_loopback(),
        "macos" => check_macos_loopback(),
        _ => Ok(LoopbackInfo {
            status: LoopbackStatus::Unsupported,
            message: "Sistema operativo no soportado".to_string(),
            instructions: vec![],
            loopback_devices: vec![],
        }),
    }
}

fn check_windows_loopback() -> Result<LoopbackInfo> {
    let output = Command::new("powershell")
        .args(&[
            "-Command",
            "Get-AudioDevice -List | Where-Object {$_.Type -eq 'Recording' -and ($_.Name -like '*Stereo Mix*' -or $_.Name -like '*Mezcla*' -or $_.Name -like '*Wave*' -or $_.Name -like '*Loopback*')} | Select-Object -Property Name"
        ])
        .output();
    
    let has_stereo_mix = output
        .as_ref()
        .map(|o| !o.stdout.is_empty())
        .unwrap_or(false);
    
    let loopback_devices = if has_stereo_mix {
        vec![DeviceInfo { 
            id: 999, 
            name: "Stereo Mix (detectado)".to_string(),
            technical_name: None,
        }]
    } else {
        vec![]
    };
    
    if has_stereo_mix {
        Ok(LoopbackInfo {
            status: LoopbackStatus::Available,
            message: "✅ Dispositivo de captura de salida detectado".to_string(),
            instructions: vec![
                "El dispositivo está disponible como INPUT.".to_string(),
                "Busca 'Stereo Mix' o 'Mezcla estéreo' en dispositivos de Entrada.".to_string(),
            ],
            loopback_devices,
        })
    } else {
        Ok(LoopbackInfo {
            status: LoopbackStatus::NeedsConfiguration,
            message: "⚠️ Necesita habilitar 'Mezcla estéreo' en Windows".to_string(),
            instructions: vec![
                "1. Click derecho en el icono de volumen (barra de tareas)".to_string(),
                "2. Selecciona 'Configuración de sonido'".to_string(),
                "3. Baja y click en 'Más opciones de sonido'".to_string(),
                "4. Ve a la pestaña 'Grabación'".to_string(),
                "5. Click derecho → 'Mostrar dispositivos deshabilitados'".to_string(),
                "6. Busca 'Mezcla estéreo' o 'Stereo Mix'".to_string(),
                "7. Click derecho → 'Habilitar'".to_string(),
            ],
            loopback_devices: vec![],
        })
    }
}

fn check_linux_loopback() -> Result<LoopbackInfo> {
    let audio_system = detect_audio_system();
    let loopback_devices = get_linux_loopback_devices();
    
    if !loopback_devices.is_empty() {
        Ok(LoopbackInfo {
            status: LoopbackStatus::Available,
            message: format!("✅ {} - {} dispositivos monitor detectados", 
                if audio_system == "pipewire" { "PipeWire" } else { "PulseAudio" },
                loopback_devices.len()
            ),
            instructions: vec![
                "Dispositivos monitor disponibles (capturan audio de salida).".to_string(),
                "".to_string(),
                "Los dispositivos 'Monitor' capturan lo que suena en tu sistema.".to_string(),
                "Úsalos en la configuración como fuentes de SALIDA.".to_string(),
            ],
            loopback_devices,
        })
    } else {
        Ok(LoopbackInfo {
            status: LoopbackStatus::RequiresSetup,
            message: format!("⚠️ {}: No se detectan dispositivos monitor", 
                if audio_system == "pipewire" { "PipeWire" } else { "PulseAudio" }
            ),
            instructions: vec![
                "Los dispositivos monitor permiten capturar audio de salida.".to_string(),
                "".to_string(),
                "SOLUCIÓN:".to_string(),
                "1. Abre el control de volumen de PulseAudio/PipeWire".to_string(),
                "2. Ve a la pestaña 'Dispositivos de entrada'".to_string(),
                "3. Deberías ver dispositivos con 'Monitor' en el nombre".to_string(),
                "".to_string(),
                "Si no aparecen, verifica que tu tarjeta de audio esté activa:".to_string(),
                "  pactl list sinks short".to_string(),
            ],
            loopback_devices: vec![],
        })
    }
}

fn check_macos_loopback() -> Result<LoopbackInfo> {
    Ok(LoopbackInfo {
        status: LoopbackStatus::RequiresSetup,
        message: "⚠️ macOS requiere software adicional".to_string(),
        instructions: vec![
            "macOS no tiene captura de salida nativa.".to_string(),
            "".to_string(),
            "Opciones recomendadas:".to_string(),
            "".to_string(),
            "1. BlackHole (GRATIS):".to_string(),
            "   https://github.com/ExistentialAudio/BlackHole".to_string(),
            "   • Descarga e instala BlackHole 2ch".to_string(),
            "   • Crea un dispositivo agregado en Audio MIDI Setup".to_string(),
            "".to_string(),
            "2. Loopback by Rogue Amoeba (PAGO):".to_string(),
            "   https://rogueamoeba.com/loopback/".to_string(),
        ],
        loopback_devices: vec![],
    })
}

pub fn get_loopback_devices() -> Vec<DeviceInfo> {
    match detect_os().as_str() {
        "linux" => get_linux_loopback_devices(),
        "windows" => get_windows_loopback_devices(),
        "macos" => get_macos_loopback_devices(),
        _ => vec![],
    }
}

fn get_linux_loopback_devices() -> Vec<DeviceInfo> {
    let mut devices = vec![];
    
    println!("DEBUG: Buscando dispositivos monitor...");
    
    // Método 1: pactl list sources short (más simple y directo)
    let output = Command::new("pactl")
        .args(&["list", "sources", "short"])
        .output();
    
    if let Ok(out) = output {
        let sources = String::from_utf8_lossy(&out.stdout);
        println!("DEBUG: Salida de 'pactl list sources short':");
        println!("{}", sources);
        
        for (idx, line) in sources.lines().enumerate() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                let tech_name = parts[1].to_string();
                println!("DEBUG: Analizando source {}: {}", idx, tech_name);
                
                if tech_name.contains(".monitor") || tech_name.contains("Monitor") {
                    println!("  -> Es monitor, obteniendo descripción...");
                    
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
    } else {
        println!("DEBUG: Error ejecutando pactl");
    }
    
    println!("DEBUG: Total monitores encontrados: {}", devices.len());
    devices
}

fn get_windows_loopback_devices() -> Vec<DeviceInfo> {
    let output = Command::new("powershell")
        .args(&[
            "-Command",
            "Get-AudioDevice -List | Where-Object {$_.Type -eq 'Recording' -and ($_.Name -like '*Stereo Mix*' -or $_.Name -like '*Mezcla*')} | Select-Object -ExpandProperty Name"
        ])
        .output();
    
    let mut devices = vec![];
    if let Ok(out) = output {
        let names = String::from_utf8_lossy(&out.stdout);
        for (idx, name) in names.lines().enumerate() {
            devices.push(DeviceInfo { 
                id: idx, 
                name: name.trim().to_string(),
                technical_name: None,
            });
        }
    }
    devices
}

fn get_macos_loopback_devices() -> Vec<DeviceInfo> {
    vec![]
}