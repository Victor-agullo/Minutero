#!/usr/bin/env bash
# ============================================================
#  Transcriptor — Instalador para Linux y macOS
#  Detecta GPU: NVIDIA (CUDA), AMD (ROCm), Intel (SYCL),
#  Apple Silicon (Metal) o CPU fallback.
# ============================================================

set -euo pipefail

REPO="tu-usuario/transcriptor"   # <── cambia esto
APP_NAME="transcriptor"
DEFAULT_INSTALL_DIR="$HOME/.local/bin"

CYAN='\033[0;36m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
RED='\033[0;31m'; GRAY='\033[0;90m'; MAGENTA='\033[0;35m'; NC='\033[0m'

step() { echo -e "\n${CYAN}[*] $*${NC}"; }
ok()   { echo -e "    ${GREEN}[OK]${NC} $*"; }
warn() { echo -e "    ${YELLOW}[!] ${NC} $*"; }
fail() { echo -e "    ${RED}[X] ${NC} $*"; exit 1; }
info() { echo -e "         ${GRAY}$*${NC}"; }

clear
echo -e "${MAGENTA}"
cat << 'EOF'
  ████████╗██████╗  █████╗ ███╗   ██╗███████╗ ██████╗██████╗ ██╗██████╗ ████████╗ ██████╗ ██████╗
     ██║   ██╔══██╗██╔══██╗████╗  ██║██╔════╝██╔════╝██╔══██╗██║██╔══██╗╚══██╔══╝██╔═══██╗██╔══██╗
     ██║   ██████╔╝███████║██╔██╗ ██║███████╗██║     ██████╔╝██║██████╔╝   ██║   ██║   ██║██████╔╝
     ██║   ██╔══██╗██╔══██║██║╚██╗██║╚════██║██║     ██╔══██╗██║██╔═══╝    ██║   ██║   ██║██╔══██╗
     ██║   ██║  ██║██║  ██║██║ ╚████║███████║╚██████╗██║  ██║██║██║        ██║   ╚██████╔╝██║  ██║
     ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝╚═╝        ╚═╝    ╚═════╝ ╚═╝  ╚═╝
  Instalador para Linux / macOS
EOF
echo -e "${NC}"


# ════════════════════════════════════════════════════════════════
#  1. DETECTAR OS Y ARQUITECTURA
# ════════════════════════════════════════════════════════════════
step "Detectando sistema..."

OS="$(uname -s)"
ARCH="$(uname -m)"
info "OS: $OS  |  Arch: $ARCH"

case "$OS" in
    Linux)  PLATFORM="linux"  ;;
    Darwin) PLATFORM="macos"  ;;
    *)      fail "Sistema '$OS' no soportado." ;;
esac


# ════════════════════════════════════════════════════════════════
#  2. DETECTAR GPU
# ════════════════════════════════════════════════════════════════
step "Analizando hardware gráfico..."

HAS_NVIDIA=false; HAS_AMD=false; HAS_INTEL=false; IS_APPLE_SILICON=false
CUDA_OK=false;    ROCM_OK=false; INTEL_SYCL_OK=false
VARIANT="cpu"
VARIANT_LABEL="CPU (sin aceleración GPU)"

if [[ "$PLATFORM" == "linux" ]]; then

    # ── lspci ────────────────────────────────────────────────
    if command -v lspci &>/dev/null; then
        while IFS= read -r line; do
            lower="${line,,}"
            if   [[ "$lower" =~ nvidia|geforce|quadro|rtx|gtx|tesla ]]; then HAS_NVIDIA=true; info "GPU NVIDIA: $line"
            elif [[ "$lower" =~ amd|radeon|"rx "[0-9]|vega|navi|rdna|polaris ]]; then HAS_AMD=true; info "GPU AMD: $line"
            elif [[ "$lower" =~ intel|"iris xe"|arc|"uhd graphics" ]]; then HAS_INTEL=true; info "GPU Intel: $line"
            fi
        done < <(lspci 2>/dev/null | grep -iE "VGA|3D|Display")
    fi

    # ── /dev/kfd (AMD ROCm device node) ──────────────────────
    [[ -e /dev/kfd ]] && HAS_AMD=true

    # ── /dev/dri/render* con nombre del driver ────────────────
    if ls /sys/class/drm/*/device/driver/module/drivers/ 2>/dev/null | grep -qi "amdgpu"; then
        HAS_AMD=true
    fi
    if ls /sys/class/drm/*/device/driver/module/drivers/ 2>/dev/null | grep -qi "i915\|xe"; then
        HAS_INTEL=true
    fi

    # ── Comprobar CUDA Runtime en sistema ────────────────────
    if [[ "$HAS_NVIDIA" == true ]]; then
        if ldconfig -p 2>/dev/null | grep -q "libcublas.so.12\|libcudart.so.12"; then
            CUDA_OK=true
        elif command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
            # Driver presente pero runtime podría estar en ruta no estándar
            for p in /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu /usr/lib64; do
                if [[ -f "$p/libcudart.so.12" ]]; then CUDA_OK=true; break; fi
            done
        fi
    fi

    # ── Comprobar ROCm ────────────────────────────────────────
    if [[ "$HAS_AMD" == true ]]; then
        if ldconfig -p 2>/dev/null | grep -q "libhipblas\|librocblas"; then
            ROCM_OK=true
        elif [[ -d /opt/rocm/lib ]] && ls /opt/rocm/lib/libhipblas* &>/dev/null 2>&1; then
            ROCM_OK=true
        fi
    fi

    # ── Comprobar Intel oneAPI ────────────────────────────────
    if [[ "$HAS_INTEL" == true ]]; then
        if ldconfig -p 2>/dev/null | grep -q "libOpenCL\|libsycl"; then
            INTEL_SYCL_OK=true
        elif [[ -d /opt/intel/oneapi ]]; then
            INTEL_SYCL_OK=true
        fi
    fi

elif [[ "$PLATFORM" == "macos" ]]; then

    GPU_INFO=$(system_profiler SPDisplaysDataType 2>/dev/null || true)

    if [[ "$ARCH" == "arm64" ]]; then
        IS_APPLE_SILICON=true
        info "Apple Silicon detectado (Metal + CoreML disponibles)"
    fi

    if echo "$GPU_INFO" | grep -iq "AMD\|Radeon"; then
        HAS_AMD=true
        info "GPU AMD: $(echo "$GPU_INFO" | grep -i "Chipset Model" | head -1 | awk -F': ' '{print $2}')"
    fi
    if echo "$GPU_INFO" | grep -iq "Intel"; then
        HAS_INTEL=true
    fi

fi


# ════════════════════════════════════════════════════════════════
#  3. ELEGIR VARIANTE
# ════════════════════════════════════════════════════════════════
step "Seleccionando variante..."

# ── macOS Apple Silicon → Metal + CoreML ─────────────────────
if [[ "$IS_APPLE_SILICON" == true ]]; then
    ok "Apple Silicon → Metal GPU + CoreML (Neural Engine)"
    VARIANT="macos-apple-silicon-metal"
    VARIANT_LABEL="Metal + CoreML (Apple Silicon)"

# ── Linux NVIDIA ──────────────────────────────────────────────
elif [[ "$HAS_NVIDIA" == true && "$PLATFORM" == "linux" ]]; then
    if [[ "$CUDA_OK" == true ]]; then
        ok "NVIDIA + CUDA 12.x disponible → variante CUDA"
        VARIANT="cuda"
        VARIANT_LABEL="CUDA (GPU NVIDIA)"
    else
        warn "GPU NVIDIA detectada pero CUDA Runtime 12.x NO está instalado."
        info "  Instalar CUDA: https://developer.nvidia.com/cuda-downloads"
        info ""
        info "  [1] Variante CUDA    — máximo rendimiento, requiere CUDA 12.x después"
        info "  [2] Variante CPU     — funciona ya mismo"
        echo ""
        read -rp "    Elige [1/2] (Enter = 1): " ans
        if [[ "${ans:-1}" != "2" ]]; then
            VARIANT="cuda"
            VARIANT_LABEL="CUDA (GPU NVIDIA)"
            warn "Instala CUDA Runtime 12.x antes de ejecutar: https://developer.nvidia.com/cuda-downloads"
        else
            VARIANT_LABEL="CPU"
        fi
    fi

# ── Linux AMD ─────────────────────────────────────────────────
elif [[ "$HAS_AMD" == true && "$PLATFORM" == "linux" ]]; then
    if [[ "$ROCM_OK" == true ]]; then
        ok "AMD + ROCm disponible → variante ROCm"
        VARIANT="rocm"
        VARIANT_LABEL="ROCm (GPU AMD)"
    else
        warn "GPU AMD detectada pero ROCm NO está instalado."
        info ""
        info "  Para instalar ROCm en Ubuntu/Debian:"
        info "    wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | sudo gpg --dearmor -o /etc/apt/keyrings/rocm.gpg"
        info "    echo 'deb [signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.1 jammy main' | sudo tee /etc/apt/sources.list.d/rocm.list"
        info "    sudo apt-get update && sudo apt-get install -y rocm-dev"
        info ""
        info "  [1] Variante ROCm    — aceleración GPU AMD, requiere instalar ROCm después"
        info "  [2] Variante CPU     — funciona ya mismo"
        echo ""
        read -rp "    Elige [1/2] (Enter = 2): " ans
        if [[ "${ans:-2}" == "1" ]]; then
            VARIANT="rocm"
            VARIANT_LABEL="ROCm (GPU AMD)"
            warn "Instala ROCm antes de ejecutar el programa."
        else
            VARIANT_LABEL="CPU"
        fi
    fi

# ── Linux Intel ───────────────────────────────────────────────
elif [[ "$HAS_INTEL" == true && "$PLATFORM" == "linux" ]]; then
    if [[ "$INTEL_SYCL_OK" == true ]]; then
        ok "Intel GPU + oneAPI SYCL disponible → variante Intel"
        VARIANT="intel"
        VARIANT_LABEL="Intel SYCL (GPU Intel Arc/Xe)"
    else
        warn "GPU Intel detectada pero Intel oneAPI Base Toolkit NO está instalado."
        info ""
        info "  Para instalar oneAPI en Ubuntu/Debian:"
        info "    wget -qO - https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \\"
        info "      | sudo gpg --dearmor -o /usr/share/keyrings/oneapi-archive-keyring.gpg"
        info "    echo 'deb [signed-by=...] https://apt.repos.intel.com/oneapi all main' | sudo tee /etc/apt/sources.list.d/oneAPI.list"
        info "    sudo apt-get update && sudo apt-get install intel-oneapi-compiler-dpcpp-cpp intel-oneapi-mkl-devel"
        info ""
        info "  [1] Variante Intel SYCL — aceleración GPU Intel, requiere oneAPI después"
        info "  [2] Variante CPU        — funciona ya mismo"
        echo ""
        read -rp "    Elige [1/2] (Enter = 2): " ans
        if [[ "${ans:-2}" == "1" ]]; then
            VARIANT="intel"
            VARIANT_LABEL="Intel SYCL (GPU Intel)"
            warn "Instala Intel oneAPI Base Toolkit antes de ejecutar el programa."
        else
            VARIANT_LABEL="CPU"
        fi
    fi

# ── macOS Intel / sin GPU dedicada ───────────────────────────
else
    ok "Sin GPU con aceleración disponible → variante CPU"
    VARIANT_LABEL="CPU"
fi

# Determinar sufijo del artifact
if [[ "$PLATFORM" == "linux" ]]; then
    case "$ARCH" in
        x86_64)  ARTIFACT_SUFFIX="linux-x86_64-$VARIANT" ;;
        aarch64) ARTIFACT_SUFFIX="linux-aarch64-$VARIANT" ;;
        *)        fail "Arquitectura '$ARCH' no soportada en Linux." ;;
    esac
elif [[ "$PLATFORM" == "macos" ]]; then
    case "$ARCH" in
        x86_64)        ARTIFACT_SUFFIX="macos-intel-cpu" ;;
        arm64|aarch64) ARTIFACT_SUFFIX="macos-apple-silicon-metal" ;;
    esac
fi

info "Artifact: transcriptor-$ARTIFACT_SUFFIX"


# ════════════════════════════════════════════════════════════════
#  4. DIRECTORIO DE INSTALACIÓN
# ════════════════════════════════════════════════════════════════
step "Directorio de instalación..."

echo ""
echo -e "    Por defecto: ${GREEN}$DEFAULT_INSTALL_DIR${NC}"
read -rp "    Pulsa Enter para aceptar o escribe otra ruta: " INSTALL_INPUT
INSTALL_DIR="${INSTALL_INPUT:-$DEFAULT_INSTALL_DIR}"
mkdir -p "$INSTALL_DIR"
ok "Directorio: $INSTALL_DIR"


# ════════════════════════════════════════════════════════════════
#  5. DESCARGAR
# ════════════════════════════════════════════════════════════════
step "Buscando última versión en GitHub..."

if ! command -v curl &>/dev/null && ! command -v wget &>/dev/null; then
    fail "Necesitas 'curl' o 'wget'."
fi

API_URL="https://api.github.com/repos/$REPO/releases"
if command -v curl &>/dev/null; then
    RELEASES_JSON=$(curl -fsSL --max-time 15 -H "User-Agent: transcriptor-installer/1.0" "$API_URL")
else
    RELEASES_JSON=$(wget -qO- --timeout=15 --header="User-Agent: transcriptor-installer/1.0" "$API_URL")
fi

# Buscar URL del artifact correcto
DOWNLOAD_URL=$(echo "$RELEASES_JSON" \
    | grep -o '"browser_download_url": *"[^"]*'"$ARTIFACT_SUFFIX"'[^"]*\.tar\.gz"' \
    | head -1 \
    | sed 's/.*"\(https[^"]*\)"/\1/')

if [[ -z "$DOWNLOAD_URL" ]]; then
    warn "No encontrado '$ARTIFACT_SUFFIX'. Buscando cualquier tarball para $PLATFORM..."
    DOWNLOAD_URL=$(echo "$RELEASES_JSON" \
        | grep -o '"browser_download_url": *"[^"]*'"$PLATFORM"'[^"]*\.tar\.gz"' \
        | head -1 \
        | sed 's/.*"\(https[^"]*\)"/\1/')
fi

[[ -z "$DOWNLOAD_URL" ]] && fail "No se encontró ningún binario. Visita: https://github.com/$REPO/releases"
ok "URL: $DOWNLOAD_URL"

TARBALL="/tmp/transcriptor-install.tar.gz"
step "Descargando..."
if command -v curl &>/dev/null; then
    curl -fL --progress-bar -o "$TARBALL" "$DOWNLOAD_URL"
else
    wget -q --show-progress -O "$TARBALL" "$DOWNLOAD_URL"
fi
ok "Descarga completada"


# ════════════════════════════════════════════════════════════════
#  6. INSTALAR
# ════════════════════════════════════════════════════════════════
step "Instalando..."

tar -xzf "$TARBALL" -C "$INSTALL_DIR"
rm -f "$TARBALL"

BINARY="$INSTALL_DIR/$APP_NAME"
if [[ ! -f "$BINARY" ]]; then
    BINARY=$(find "$INSTALL_DIR" -name "$APP_NAME" -type f | head -1)
fi
[[ -z "$BINARY" || ! -f "$BINARY" ]] && fail "No se encontró el binario tras la instalación."

chmod +x "$BINARY"
ok "Instalado: $BINARY"


# ════════════════════════════════════════════════════════════════
#  7. PATH
# ════════════════════════════════════════════════════════════════
step "Configurando PATH..."

add_to_path() {
    local dir="$1" rc="$2"
    local line="export PATH=\"$dir:\$PATH\""
    [[ -f "$rc" ]] && grep -qF "$dir" "$rc" && return
    { echo ""; echo "# Transcriptor"; echo "$line"; } >> "$rc"
    ok "Añadido a $rc"
}

if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
    add_to_path "$INSTALL_DIR" "$HOME/.bashrc"
    add_to_path "$INSTALL_DIR" "$HOME/.zshrc"
    add_to_path "$INSTALL_DIR" "$HOME/.profile"
    warn "Reinicia el terminal o ejecuta: source ~/.bashrc"
else
    ok "Ya estaba en el PATH"
fi


# ════════════════════════════════════════════════════════════════
#  8. DEPENDENCIAS
# ════════════════════════════════════════════════════════════════
step "Comprobando dependencias..."

# ffmpeg
if command -v ffmpeg &>/dev/null; then
    ok "ffmpeg $(ffmpeg -version 2>&1 | head -1 | awk '{print $3}')"
else
    warn "ffmpeg NO instalado (necesario para transcribir archivos de vídeo/audio)."
    if [[ "$PLATFORM" == "linux" ]]; then
        info "  sudo apt install ffmpeg      (Debian/Ubuntu)"
        info "  sudo dnf install ffmpeg      (Fedora)"
        info "  sudo pacman -S ffmpeg         (Arch)"
    elif [[ "$PLATFORM" == "macos" ]]; then
        info "  brew install ffmpeg"
    fi
fi

# PulseAudio/PipeWire (Linux)
if [[ "$PLATFORM" == "linux" ]]; then
    if command -v pactl &>/dev/null; then
        ok "PulseAudio/PipeWire disponible"
    else
        warn "pactl no encontrado (necesario para captura en tiempo real)."
        info "  sudo apt install pulseaudio-utils"
    fi
fi

# BlackHole (macOS)
if [[ "$PLATFORM" == "macos" ]]; then
    if system_profiler SPAudioDataType 2>/dev/null | grep -iq "blackhole\|soundflower\|loopback"; then
        ok "Dispositivo de audio virtual detectado"
    else
        warn "Para capturar audio del sistema en macOS instala BlackHole:"
        info "  https://github.com/ExistentialAudio/BlackHole"
    fi
fi

# Avisos específicos por variante
if [[ "$VARIANT" == "rocm" && "$ROCM_OK" == "false" ]]; then
    warn "Recuerda instalar ROCm antes de ejecutar:"
    info "  https://rocm.docs.amd.com/projects/install-on-linux/en/latest/"
fi

if [[ "$VARIANT" == "intel" && "$INTEL_SYCL_OK" == "false" ]]; then
    warn "Recuerda instalar Intel oneAPI Base Toolkit antes de ejecutar:"
    info "  https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html"
fi

if [[ "$VARIANT" == "cuda" && "$CUDA_OK" == "false" ]]; then
    warn "Recuerda instalar CUDA Runtime 12.x antes de ejecutar:"
    info "  https://developer.nvidia.com/cuda-downloads"
fi


# ════════════════════════════════════════════════════════════════
#  RESUMEN
# ════════════════════════════════════════════════════════════════
echo ""
echo -e "${MAGENTA}  ═══════════════════════════════════════════${NC}"
echo -e "${GREEN}   Instalación completada${NC}"
echo -e "${MAGENTA}  ═══════════════════════════════════════════${NC}"
echo ""
echo -e "   Variante  : ${CYAN}${VARIANT_LABEL}${NC}"
echo    "   Plataforma: $PLATFORM / $ARCH"
echo    "   Binario   : $BINARY"
echo ""
echo    "   Para lanzar: transcriptor"
echo -e "   ${GRAY}(puede requerir reiniciar el terminal)${NC}"
echo ""