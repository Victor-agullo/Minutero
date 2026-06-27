; ============================================================
;  Transcriptor — Instalador para Windows
;  Compilar: ISCC.exe /DMyAppVersion=1.0.0 /DRepo=Victor-agullo/Minutero installer.iss
;
;  Crea Install-Transcriptor.exe con asistente de instalación que:
;   · Detecta la GPU automáticamente
;   · Descarga la variante correcta de GitHub Releases
;   · Crea acceso directo, añade al PATH, instala ffmpeg opcionales
; ============================================================

; Parámetros inyectados desde CI (con /D)
#ifndef MyAppVersion
  #define MyAppVersion "dev"
#endif
#ifndef Repo
  #define Repo "Victor-agullo/Minutero"
#endif

#define MyAppName      "Transcriptor"
#define MyAppPublisher "Victor Agullo"
#define MyAppURL       "https://github.com/{#Repo}"
#define MyAppExeName   "transcriptor.exe"

; ── Configuración general ─────────────────────────────────────────────────

[Setup]
AppId={9E7F8A3C-B2D4-4E8F-A1C3-7D5E2F9B0A8C}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
DefaultDirName={localappdata}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputBaseFilename=Install-Transcriptor
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
; No requiere admin — instala en %LOCALAPPDATA% del usuario
PrivilegesRequired=lowest
SetupMutex=TranscriptorInstaller
VersionInfoVersion=1.0.0.0
VersionInfoDescription=Instalador de {#MyAppName}
UninstallDisplayIcon={app}\{#MyAppExeName}
CreateUninstallRegKey=yes
; Mostrar checkbox para lanzar al finalizar
FinishMessage=Transcriptor se ha instalado correctamente.

[Languages]
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

; ── Tareas opcionales ─────────────────────────────────────────────────────

[Tasks]
Name: "desktopicon"; Description: "Crear icono en el &Escritorio"; \
  GroupDescription: "{cm:AdditionalIcons}"

; ── Iconos y menú inicio ─────────────────────────────────────────────────

[Icons]
Name: "{userdesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; \
  Tasks: desktopicon
Name: "{group}\{#MyAppName}";       Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Desinstalar {#MyAppName}"; Filename: "{uninstallexe}"

; ── Ejecutar al finalizar ─────────────────────────────────────────────────

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Iniciar {#MyAppName}"; \
  Flags: nowait postinstall skipifsilent

; ── Código Pascal ─────────────────────────────────────────────────────────

[Code]

// ── Imports de Windows API ────────────────────────────────────────────────

// Descarga síncrona de un archivo desde una URL (urlmon.dll, disponible en
// cualquier Windows XP+). Devuelve 0 (S_OK) si tiene éxito.
function URLDownloadToFile(pCaller: IUnknown; szURL: WideString;
  szFileName: WideString; dwReserved: DWORD; lpfnCB: IUnknown): HRESULT;
  external 'URLDownloadToFileW@urlmon.dll stdcall';

// Notifica al shell que el PATH ha cambiado (para que se propague sin reiniciar)
function SendMessageTimeout(hWnd: HWND; Msg: UINT; wParam: WPARAM; lParam: string;
  fuFlags: UINT; uTimeout: UINT; out lpdwResult: DWORD): LRESULT;
  external 'SendMessageTimeoutW@user32.dll stdcall';

const
  SMTO_ABORTIFHUNG = $0002;
  WM_SETTINGCHANGE  = $001A;
  HWND_BROADCAST    = $FFFF;

// ── Variables globales ────────────────────────────────────────────────────

var
  GPUType:       string;  // 'nvidia' | 'amd' | 'intel' | 'none'
  ChosenVariant: string;  // 'cuda' | 'directml' | 'cpu'
  VariantLabel:  string;

  // Controles de la página de selección de GPU
  GPUPage:       TWizardPage;
  GPUInfoLabel:  TNewStaticText;
  RBCuda:        TRadioButton;
  RBDirectml:    TRadioButton;
  RBCpu:         TRadioButton;
  GPUNoteLabel:  TNewStaticText;

// ── Detección de GPU ──────────────────────────────────────────────────────

// Ejecuta un script PowerShell que detecta la GPU y escribe el resultado
// ('nvidia', 'amd', 'intel' o 'none') en un fichero temporal.
function DetectGPU: string;
var
  TmpScript, TmpOutput: string;
  Lines: TArrayOfString;
  Code:  Integer;
begin
  TmpScript := ExpandConstant('{tmp}\detect_gpu.ps1');
  TmpOutput := ExpandConstant('{tmp}\gpu_type.txt');

  SaveStringToFile(TmpScript,
    '$result = "none"' + #13#10 +
    'try {' + #13#10 +
    '  Get-CimInstance Win32_VideoController |' + #13#10 +
    '    Where-Object { $_.Name -notmatch "Remote|Virtual|Basic|Microsoft" } |' + #13#10 +
    '    ForEach-Object {' + #13#10 +
    '      $n = $_.Name.ToLower()' + #13#10 +
    '      if ($n -match "nvidia|geforce|rtx|gtx|quadro|tesla") { $result = "nvidia"; return }' + #13#10 +
    '      if ($n -match "amd|radeon")                           { $result = "amd" }' + #13#10 +
    '      if ($n -match "intel|arc|iris|uhd" -and $result -eq "none") { $result = "intel" }' + #13#10 +
    '    }' + #13#10 +
    '} catch {}' + #13#10 +
    '$result | Set-Content -LiteralPath ''' + TmpOutput + ''' -NoNewline',
    False);

  Exec('powershell.exe',
    '-NonInteractive -ExecutionPolicy Bypass -File "' + TmpScript + '"',
    '', SW_HIDE, ewWaitUntilTerminated, Code);

  DeleteFile(TmpScript);

  if LoadStringsFromFile(TmpOutput, Lines) and (GetArrayLength(Lines) > 0) then
    Result := Trim(Lines[0])
  else
    Result := 'none';

  DeleteFile(TmpOutput);
end;

// ── Obtener URL de descarga desde GitHub API ──────────────────────────────

// Consulta la API de GitHub, busca el asset que coincide con el sufijo dado
// y escribe su URL de descarga en un fichero temporal.
function FetchDownloadURL(ArtifactSuffix: string): string;
var
  TmpScript, TmpOutput: string;
  Lines: TArrayOfString;
  Code:  Integer;
begin
  TmpScript := ExpandConstant('{tmp}\get_url.ps1');
  TmpOutput := ExpandConstant('{tmp}\dl_url.txt');

  SaveStringToFile(TmpScript,
    'try {' + #13#10 +
    '  $releases = Invoke-RestMethod "https://api.github.com/repos/{#Repo}/releases"' + #13#10 +
    '    -Headers @{ ''User-Agent'' = ''transcriptor-installer'' } -TimeoutSec 20' + #13#10 +
    '  $asset = $releases | ForEach-Object { $_.assets }' + #13#10 +
    '    | Where-Object { $_.name -like ''*' + ArtifactSuffix + '*'' -and $_.name -like ''*.zip'' }' + #13#10 +
    '    | Select-Object -First 1' + #13#10 +
    '  if ($asset) { $asset.browser_download_url }' + #13#10 +
    '  else        { "" }' + #13#10 +
    '  | Set-Content -LiteralPath ''' + TmpOutput + ''' -NoNewline' + #13#10 +
    '} catch {' + #13#10 +
    '  "" | Set-Content -LiteralPath ''' + TmpOutput + ''' -NoNewline' + #13#10 +
    '}',
    False);

  Exec('powershell.exe',
    '-NonInteractive -ExecutionPolicy Bypass -File "' + TmpScript + '"',
    '', SW_HIDE, ewWaitUntilTerminated, Code);

  DeleteFile(TmpScript);

  if LoadStringsFromFile(TmpOutput, Lines) and (GetArrayLength(Lines) > 0) then
    Result := Trim(Lines[0])
  else
    Result := '';

  DeleteFile(TmpOutput);
end;

// ── Comprobación de PATH ──────────────────────────────────────────────────

function IsInUserPath(Dir: string): Boolean;
var OldPath: string;
begin
  if not RegQueryStringValue(HKEY_CURRENT_USER, 'Environment', 'Path', OldPath) then
    OldPath := '';
  Result := Pos(Uppercase(Dir), Uppercase(OldPath)) > 0;
end;

procedure AddToUserPath(Dir: string);
var
  OldPath: DWORD;
begin
  if IsInUserPath(Dir) then Exit;

  RegQueryStringValue(HKEY_CURRENT_USER, 'Environment', 'Path', '');
  // Append via registry; use ExpandConstant so %LOCALAPPDATA% type vars survive
  if RegQueryStringValue(HKEY_CURRENT_USER, 'Environment', 'Path', '') then
    RegWriteExpandStringValue(HKEY_CURRENT_USER, 'Environment', 'Path',
      RegQueryStringValue(HKEY_CURRENT_USER, 'Environment', 'Path', '') + ';' + Dir)
  else
    RegWriteExpandStringValue(HKEY_CURRENT_USER, 'Environment', 'Path', Dir);

  // Broadcast WM_SETTINGCHANGE so Explorer picks up the new PATH without reboot
  SendMessageTimeout(HWND_BROADCAST, WM_SETTINGCHANGE, 0, 'Environment',
    SMTO_ABORTIFHUNG, 2000, OldPath);
end;

// ── Página personalizada de selección de variante ─────────────────────────

procedure CreateGPUPage;
var
  Desc: string;
begin
  GPUPage := CreateCustomPage(wpSelectDir,
    'Selección de versión',
    'Elige la versión optimizada para tu hardware gráfico');

  GPUInfoLabel := TNewStaticText.Create(GPUPage);
  GPUInfoLabel.Parent   := GPUPage.Surface;
  GPUInfoLabel.Left     := 0;
  GPUInfoLabel.Top      := 0;
  GPUInfoLabel.Width    := GPUPage.SurfaceWidth;
  GPUInfoLabel.AutoSize := False;
  GPUInfoLabel.Height   := 44;
  GPUInfoLabel.WordWrap := True;

  case GPUType of
    'nvidia': Desc := 'GPU NVIDIA detectada. Se recomienda CUDA para máximo rendimiento. ' +
                      'Requiere CUDA Runtime 12.x instalado.';
    'amd':    Desc := 'GPU AMD Radeon detectada. Se recomienda DirectML. ' +
                      'No necesita drivers adicionales en Windows 10/11.';
    'intel':  Desc := 'GPU Intel detectada. Se recomienda DirectML. ' +
                      'No necesita drivers adicionales en Windows 10/11.';
  else        Desc := 'No se detectó GPU dedicada. Se instalará la versión CPU, ' +
                      'compatible con cualquier equipo.';
  end;
  GPUInfoLabel.Caption := Desc;

  RBCuda := TRadioButton.Create(GPUPage);
  RBCuda.Parent  := GPUPage.Surface;
  RBCuda.Left    := 4;
  RBCuda.Top     := 54;
  RBCuda.Width   := GPUPage.SurfaceWidth - 4;
  RBCuda.Caption := 'CUDA  —  GPU NVIDIA  (máximo rendimiento, requiere CUDA Runtime 12.x)';
  RBCuda.Enabled := (GPUType = 'nvidia');

  RBDirectml := TRadioButton.Create(GPUPage);
  RBDirectml.Parent  := GPUPage.Surface;
  RBDirectml.Left    := 4;
  RBDirectml.Top     := 82;
  RBDirectml.Width   := GPUPage.SurfaceWidth - 4;
  RBDirectml.Caption := 'DirectML  —  GPU AMD / Intel  (Windows 10/11, sin instalación extra)';
  RBDirectml.Enabled := (GPUType = 'amd') or (GPUType = 'intel');

  RBCpu := TRadioButton.Create(GPUPage);
  RBCpu.Parent  := GPUPage.Surface;
  RBCpu.Left    := 4;
  RBCpu.Top     := 110;
  RBCpu.Width   := GPUPage.SurfaceWidth - 4;
  RBCpu.Caption := 'CPU  —  Compatible con cualquier equipo (transcripción más lenta)';

  // Preselección según GPU detectada
  if GPUType = 'nvidia'                            then RBCuda.Checked    := True
  else if (GPUType = 'amd') or (GPUType = 'intel') then RBDirectml.Checked := True
  else                                                   RBCpu.Checked      := True;

  GPUNoteLabel := TNewStaticText.Create(GPUPage);
  GPUNoteLabel.Parent      := GPUPage.Surface;
  GPUNoteLabel.Left        := 4;
  GPUNoteLabel.Top         := 148;
  GPUNoteLabel.Width       := GPUPage.SurfaceWidth - 4;
  GPUNoteLabel.AutoSize    := False;
  GPUNoteLabel.Height      := 28;
  GPUNoteLabel.WordWrap    := True;
  GPUNoteLabel.Font.Color  := clGray;
  GPUNoteLabel.Caption     :=
    'Puedes reinstalar en cualquier momento para cambiar de versión.';
end;

// ── InitializeWizard — punto de entrada del asistente ────────────────────

procedure InitializeWizard;
begin
  GPUType := DetectGPU;
  CreateGPUPage;
end;

// ── NextButtonClick — capturar selección al avanzar ───────────────────────

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;

  if CurPageID = GPUPage.ID then
  begin
    if RBCuda.Checked then
    begin ChosenVariant := 'cuda';     VariantLabel := 'CUDA (GPU NVIDIA)'; end
    else if RBDirectml.Checked then
    begin ChosenVariant := 'directml'; VariantLabel := 'DirectML (GPU AMD/Intel)'; end
    else
    begin ChosenVariant := 'cpu';      VariantLabel := 'CPU'; end;
  end;
end;

// ── UpdateReadyMemo — resumen antes de instalar ───────────────────────────

function UpdateReadyMemo(Space, NewLine, MemoUserInfoInfo, MemoDirInfo,
  MemoTypeInfo, MemoComponentsInfo, MemoGroupInfo, MemoTasksInfo: string): string;
begin
  Result :=
    'Versión a instalar:' + NewLine +
    Space + VariantLabel + NewLine + NewLine +
    MemoDirInfo;
  if MemoTasksInfo <> '' then
    Result := Result + NewLine + NewLine + MemoTasksInfo;
end;

// ── CurStepChanged — lógica principal de instalación ─────────────────────

procedure CurStepChanged(CurStep: TSetupStep);
var
  ArtifactSuffix, DownloadURL, ZipPath, AppDir: string;
  Res, Code: Integer;
begin
  if CurStep <> ssInstall then Exit;

  ArtifactSuffix := 'windows-x86_64-' + ChosenVariant;
  AppDir  := ExpandConstant('{app}');
  ZipPath := ExpandConstant('{tmp}\transcriptor-setup.zip');

  // ── 1. Resolver URL ────────────────────────────────────────
  WizardForm.StatusLabel.Caption :=
    'Buscando la versión más reciente en GitHub...';
  Application.ProcessMessages;

  DownloadURL := FetchDownloadURL(ArtifactSuffix);

  if DownloadURL = '' then
  begin
    MsgBox(
      'No se encontró el archivo para la variante «' + VariantLabel + '».' + #13#10#13#10 +
      'Comprueba que existe una release en:' + #13#10 +
      '  https://github.com/{#Repo}/releases' + #13#10#13#10 +
      'También puedes descargarlo manualmente y copiar' + #13#10 +
      'transcriptor.exe a la carpeta de instalación.',
      mbError, MB_OK);
    Abort;
  end;

  // ── 2. Descargar ────────────────────────────────────────────
  WizardForm.StatusLabel.Caption :=
    'Descargando Transcriptor (' + VariantLabel + ')... esto puede tardar varios minutos.';
  Application.ProcessMessages;

  Res := URLDownloadToFile(nil, DownloadURL, ZipPath, 0, nil);
  if Res <> 0 then
  begin
    MsgBox(
      'Error al descargar el archivo (código ' + IntToStr(Res) + ').' + #13#10 +
      'Comprueba tu conexión a internet e inténtalo de nuevo.',
      mbError, MB_OK);
    Abort;
  end;

  // ── 3. Extraer ──────────────────────────────────────────────
  WizardForm.StatusLabel.Caption := 'Extrayendo archivos...';
  Application.ProcessMessages;
  ForceDirectories(AppDir);

  Exec('powershell.exe',
    '-NonInteractive -ExecutionPolicy Bypass -Command ' +
    '"Expand-Archive -LiteralPath ''' + ZipPath + ''' -DestinationPath ''' + AppDir + ''' -Force"',
    '', SW_HIDE, ewWaitUntilTerminated, Code);

  DeleteFile(ZipPath);

  if not FileExists(AppDir + '\transcriptor.exe') then
  begin
    MsgBox('No se encontró transcriptor.exe tras la extracción.', mbError, MB_OK);
    Abort;
  end;

  // Guardar variante instalada (útil para actualizaciones futuras)
  SaveStringToFile(AppDir + '\variant.txt', ChosenVariant, False);

  // ── 4. Añadir al PATH ───────────────────────────────────────
  AddToUserPath(AppDir);
end;

// ── CurPageChanged — comprobar ffmpeg al llegar a la pantalla final ───────

procedure CurPageChanged(CurPageID: Integer);
var
  Code: Integer;
  FfmpegOK: Boolean;
begin
  if CurPageID <> wpFinished then Exit;

  // Comprobar si ffmpeg está en PATH
  Exec('powershell.exe',
    '-NonInteractive -ExecutionPolicy Bypass -Command "Get-Command ffmpeg -ErrorAction Stop | Out-Null"',
    '', SW_HIDE, ewWaitUntilTerminated, Code);
  FfmpegOK := (Code = 0);

  if not FfmpegOK then
  begin
    if MsgBox(
      'ffmpeg no está instalado en este equipo.' + #13#10 +
      'ffmpeg es necesario para transcribir archivos de vídeo y audio.' + #13#10#13#10 +
      '¿Quieres instalarlo ahora con winget? (recomendado)',
      mbConfirmation, MB_YESNO) = IDYES then
    begin
      WizardForm.StatusLabel.Caption := 'Instalando ffmpeg con winget...';
      Application.ProcessMessages;

      Exec('cmd.exe',
        '/c winget install --id Gyan.FFmpeg -e --silent ' +
        '--accept-package-agreements --accept-source-agreements',
        '', SW_HIDE, ewWaitUntilTerminated, Code);

      if Code = 0 then
        MsgBox(
          'ffmpeg instalado correctamente.' + #13#10 +
          'Puede ser necesario reiniciar el terminal para que esté disponible.',
          mbInformation, MB_OK)
      else
        MsgBox(
          'No se pudo instalar ffmpeg automáticamente.' + #13#10 +
          'Instálalo manualmente desde: https://ffmpeg.org/download.html' + #13#10 +
          '  o con: winget install Gyan.FFmpeg',
          mbError, MB_OK);
    end;
  end;
end;
