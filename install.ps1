# ============================================================
#  Transcriptor — Instalador para Windows
#  Detecta GPU y descarga CUDA (NVIDIA), DirectML (AMD/Intel)
#  o CPU según el hardware disponible.
# ============================================================

param(
    [string]$InstallDir = "",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$ProgressPreference    = "SilentlyContinue"

$REPO     = "tu-usuario/transcriptor"   # <── cambia esto
$APP_NAME = "transcriptor"

# ── Helpers ───────────────────────────────────────────────────────────────
function Write-Step { param($msg) Write-Host "`n[*] $msg" -ForegroundColor Cyan }
function Write-OK   { param($msg) Write-Host "    [OK] $msg" -ForegroundColor Green }
function Write-Warn { param($msg) Write-Host "    [!]  $msg" -ForegroundColor Yellow }
function Write-Fail { param($msg) Write-Host "    [X]  $msg" -ForegroundColor Red; exit 1 }
function Write-Info { param($msg) Write-Host "         $msg" -ForegroundColor Gray }

# ── Banner ────────────────────────────────────────────────────────────────
Clear-Host
Write-Host @"

  ████████╗██████╗  █████╗ ███╗   ██╗███████╗ ██████╗██████╗ ██╗██████╗ ████████╗ ██████╗ ██████╗
     ██║   ██╔══██╗██╔══██╗████╗  ██║██╔════╝██╔════╝██╔══██╗██║██╔══██╗╚══██╔══╝██╔═══██╗██╔══██╗
     ██║   ██████╔╝███████║██╔██╗ ██║███████╗██║     ██████╔╝██║██████╔╝   ██║   ██║   ██║██████╔╝
     ██║   ██╔══██╗██╔══██║██║╚██╗██║╚════██║██║     ██╔══██╗██║██╔═══╝    ██║   ██║   ██║██╔══██╗
     ██║   ██║  ██║██║  ██║██║ ╚████║███████║╚██████╗██║  ██║██║██║        ██║   ╚██████╔╝██║  ██║
     ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝╚═╝        ╚═╝    ╚═════╝ ╚═╝  ╚═╝

  Instalador para Windows
"@ -ForegroundColor Magenta


# ════════════════════════════════════════════════════════════════
#  1. DETECTAR GPU
# ════════════════════════════════════════════════════════════════
Write-Step "Analizando hardware gráfico..."

$gpu = @{
    hasNvidia    = $false
    hasAmd       = $false
    hasIntel     = $false
    cudaOk       = $false
    directmlOk   = $false
    names        = @()
}

try {
    $controllers = Get-WmiObject Win32_VideoController |
        Where-Object { $_.Name -notmatch "Remote|Virtual|Basic|Microsoft" }

    foreach ($c in $controllers) {
        $name  = $c.Name
        $lower = $name.ToLower()
        $gpu.names += $name
        Write-Info "GPU: $name"

        if ($lower -match "nvidia|geforce|quadro|rtx|gtx|tesla") {
            $gpu.hasNvidia = $true
        } elseif ($lower -match "amd|radeon|rx \d|vega|navi|rdna|polaris|rx5|rx6|rx7") {
            $gpu.hasAmd = $true
        } elseif ($lower -match "intel|iris|uhd|hd graphics|arc|xe") {
            $gpu.hasIntel = $true
        }
    }
} catch {
    Write-Warn "No se pudo leer la GPU via WMI. Se usará CPU."
}

# ── Comprobar CUDA Runtime ────────────────────────────────────
if ($gpu.hasNvidia) {
    $cudaDlls  = @("cublas64_12.dll", "cudart64_120.dll")
    $cudaPaths = @(
        "$env:SystemRoot\System32",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin"
    )
    $allFound = $true
    foreach ($dll in $cudaDlls) {
        $found = $false
        foreach ($path in $cudaPaths) {
            if (Test-Path "$path\$dll") { $found = $true; break }
        }
        if (-not $found) { $allFound = $false; break }
    }
    $gpu.cudaOk = $allFound
}

# ── Comprobar DirectML (AMD / Intel) ─────────────────────────
# DirectML viene con Windows 10 1903+ / Windows 11.
# Verifica que el sistema sea compatible (build >= 18362).
if ($gpu.hasAmd -or $gpu.hasIntel) {
    try {
        $build = [int](Get-ItemProperty "HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion").CurrentBuildNumber
        if ($build -ge 18362) {
            $gpu.directmlOk = $true
        }
    } catch { }
}


# ════════════════════════════════════════════════════════════════
#  2. ELEGIR VARIANTE
# ════════════════════════════════════════════════════════════════
Write-Step "Seleccionando variante..."

$variant        = "cpu"
$artifactSuffix = "windows-x86_64-cpu"
$variantLabel   = "CPU (sin aceleración GPU)"

if ($gpu.hasNvidia) {
    if ($gpu.cudaOk) {
        ok "GPU NVIDIA + CUDA Runtime 12.x disponibles → variante CUDA"
        $variant        = "cuda"
        $artifactSuffix = "windows-x86_64-cuda"
        $variantLabel   = "CUDA (GPU NVIDIA)"
    } else {
        Write-Warn "GPU NVIDIA detectada, pero CUDA Runtime 12.x NO está instalado."
        Write-Info "Descarga CUDA: https://developer.nvidia.com/cuda-downloads"
        Write-Info ""
        Write-Info "Opciones:"
        Write-Info "  [1] Variante CUDA     — máximo rendimiento, necesita CUDA 12.x después"
        Write-Info "  [2] Variante CPU      — funciona ya mismo, más lenta"
        Write-Host ""
        $ans = Read-Host "    Elige [1/2] (Enter = 1)"
        if ($ans -ne "2") {
            $variant        = "cuda"
            $artifactSuffix = "windows-x86_64-cuda"
            $variantLabel   = "CUDA (GPU NVIDIA)"
            Write-Warn "Instala CUDA Runtime 12.x antes de ejecutar el programa."
        } else {
            $variantLabel = "CPU"
        }
    }
} elseif ($gpu.hasAmd -or $gpu.hasIntel) {
    $gpuType = if ($gpu.hasAmd) { "AMD Radeon" } else { "Intel" }

    if ($gpu.directmlOk) {
        Write-OK "GPU $gpuType + Windows 10/11 detectados → variante DirectML"
        Write-Info "DirectML acelera Whisper en GPUs AMD e Intel sin drivers adicionales."
        $variant        = "directml"
        $artifactSuffix = "windows-x86_64-directml"
        $variantLabel   = "DirectML (GPU $gpuType)"
    } else {
        Write-Warn "GPU $gpuType detectada, pero DirectML requiere Windows 10 v1903 o superior."
        Write-Info "Tu versión de Windows es demasiado antigua. Se usará CPU."
        $variantLabel = "CPU"
    }
} else {
    Write-OK "Sin GPU dedicada → variante CPU"
}

Write-Info "Variante seleccionada: $variantLabel"


# ════════════════════════════════════════════════════════════════
#  3. DIRECTORIO DE INSTALACIÓN
# ════════════════════════════════════════════════════════════════
Write-Step "Directorio de instalación..."

if ($InstallDir -eq "") {
    $default = "$env:LOCALAPPDATA\$APP_NAME"
    Write-Host ""
    Write-Host "    Por defecto: $default" -ForegroundColor White
    $input = Read-Host "    Pulsa Enter para aceptar o escribe otra ruta"
    $InstallDir = if ($input -eq "") { $default } else { $input.Trim() }
}

if (-not (Test-Path $InstallDir)) {
    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
}
Write-OK "Directorio: $InstallDir"


# ════════════════════════════════════════════════════════════════
#  4. DESCARGAR BINARIO
# ════════════════════════════════════════════════════════════════
Write-Step "Buscando última versión en GitHub..."

$headers  = @{ "User-Agent" = "transcriptor-installer/1.0" }
$apiUrl   = "https://api.github.com/repos/$REPO/releases"

try {
    $releases = Invoke-RestMethod -Uri $apiUrl -Headers $headers -TimeoutSec 15
} catch {
    Write-Fail "No se pudo contactar GitHub API: $_"
}

# Buscar asset con el sufijo correcto en cualquier release
$asset = $null
foreach ($r in $releases) {
    $asset = $r.assets | Where-Object { $_.name -match [regex]::Escape($artifactSuffix) -and $_.name -match "\.zip$" } | Select-Object -First 1
    if ($asset) { Write-OK "Versión: $($r.tag_name)"; break }
}

if (-not $asset) {
    Write-Fail "No se encontró '$artifactSuffix' en ninguna release. Visita: https://github.com/$REPO/releases"
}

Write-Info "Archivo: $($asset.name) ($([math]::Round($asset.size/1MB,1)) MB)"

$zipPath = "$env:TEMP\transcriptor-install.zip"
Write-Step "Descargando..."
try {
    Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $zipPath -Headers $headers
    Write-OK "Descarga completada"
} catch {
    Write-Fail "Error al descargar: $_"
}


# ════════════════════════════════════════════════════════════════
#  5. INSTALAR
# ════════════════════════════════════════════════════════════════
Write-Step "Instalando..."

try {
    Expand-Archive -Path $zipPath -DestinationPath $InstallDir -Force
    Remove-Item $zipPath -ErrorAction SilentlyContinue
} catch {
    Write-Fail "Error al descomprimir: $_"
}

$exePath = "$InstallDir\transcriptor.exe"
if (-not (Test-Path $exePath)) {
    $exePath = Get-ChildItem -Path $InstallDir -Filter "transcriptor.exe" -Recurse |
        Select-Object -First 1 -ExpandProperty FullName
}
if (-not $exePath -or -not (Test-Path $exePath)) {
    Write-Fail "No se encontró transcriptor.exe tras la instalación."
}
Write-OK "Instalado: $exePath"


# ════════════════════════════════════════════════════════════════
#  6. PATH + ACCESO DIRECTO
# ════════════════════════════════════════════════════════════════
Write-Step "Configurando PATH..."

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath -notmatch [regex]::Escape($InstallDir)) {
    [Environment]::SetEnvironmentVariable("Path", "$userPath;$InstallDir", "User")
    Write-OK "Añadido al PATH de usuario"
} else {
    Write-OK "Ya estaba en el PATH"
}

Write-Step "Acceso directo..."
$ans = Read-Host "    ¿Crear acceso directo en el Escritorio? [S/n]"
if ($ans -notmatch "^[nN]$") {
    $wsh = New-Object -ComObject WScript.Shell
    $lnk = $wsh.CreateShortcut("$([Environment]::GetFolderPath('Desktop'))\Transcriptor.lnk")
    $lnk.TargetPath       = $exePath
    $lnk.WorkingDirectory = $InstallDir
    $lnk.Description      = "Transcriptor Multicanal"
    $lnk.Save()
    Write-OK "Acceso directo creado en el Escritorio"
}


# ════════════════════════════════════════════════════════════════
#  7. DEPENDENCIAS
# ════════════════════════════════════════════════════════════════
Write-Step "Comprobando dependencias..."

# ffmpeg
if (Get-Command ffmpeg -ErrorAction SilentlyContinue) {
    Write-OK "ffmpeg encontrado"
} else {
    Write-Warn "ffmpeg NO está instalado."
    Write-Info "Necesario para transcribir vídeos/audio de archivo."
    Write-Info ""
    $ans = Read-Host "    ¿Instalar ffmpeg ahora con winget? [S/n]"
    if ($ans -notmatch "^[nN]$") {
        try {
            winget install --id Gyan.FFmpeg -e --silent
            Write-OK "ffmpeg instalado (reinicia el terminal para que esté disponible)"
        } catch {
            Write-Warn "No se pudo instalar automáticamente."
            Write-Info "Instálalo manualmente: https://ffmpeg.org/download.html"
        }
    }
}

# Avisos específicos por variante
if ($variant -eq "cuda" -and -not $gpu.cudaOk) {
    Write-Warn "Recuerda instalar CUDA Runtime 12.x:"
    Write-Info "  https://developer.nvidia.com/cuda-downloads"
}

if ($variant -eq "directml") {
    Write-OK "DirectML: nativo en Windows 10/11, no requiere instalación adicional."
    Write-Info "Si el rendimiento es bajo, actualiza los drivers de tu GPU:"
    if ($gpu.hasAmd) {
        Write-Info "  AMD Software (Adrenalin): https://www.amd.com/es/support"
    } elseif ($gpu.hasIntel) {
        Write-Info "  Intel Arc Control / Driver: https://www.intel.com/content/www/us/en/download-center/home.html"
    }
}


# ════════════════════════════════════════════════════════════════
#  RESUMEN
# ════════════════════════════════════════════════════════════════
Write-Host ""
Write-Host "  ═══════════════════════════════════════════" -ForegroundColor Magenta
Write-Host "   Instalación completada" -ForegroundColor Green
Write-Host "  ═══════════════════════════════════════════" -ForegroundColor Magenta
Write-Host ""
Write-Host "   Variante  : " -NoNewline; Write-Host $variantLabel -ForegroundColor Cyan
Write-Host "   Ubicación : $exePath"
Write-Host ""
Write-Host "   Para lanzar: transcriptor" -ForegroundColor White
Write-Host "   (o usa el acceso directo del Escritorio)" -ForegroundColor Gray
Write-Host ""

Pause