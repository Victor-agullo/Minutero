@echo off
:: ============================================================
::  Transcriptor — Lanzador del instalador para Windows
::
::  Resuelve la política de ejecución de PowerShell que impide
::  ejecutar scripts .ps1 directamente con doble clic.
::  Este .bat actúa como puente y no requiere ningún permiso
::  especial (instala en %LOCALAPPDATA% del usuario).
::
::  Uso: doble clic en Install.bat
:: ============================================================
chcp 65001 >nul 2>&1
setlocal EnableDelayedExpansion

echo.
echo   Transcriptor ^— Instalador
echo   ══════════════════════════
echo.

:: Comprobar que PowerShell está disponible (cualquier Windows moderno lo tiene)
where powershell.exe >nul 2>&1
if errorlevel 1 (
    echo   ERROR: PowerShell no encontrado.
    echo   Este instalador requiere PowerShell 5.1 o superior.
    echo   Descarga PowerShell desde: https://aka.ms/powershell
    echo.
    pause
    exit /b 1
)

:: Lanzar install.ps1 con:
::   -NoProfile          : inicio más rápido, sin cargar el perfil del usuario
::   -ExecutionPolicy Bypass : omitir restricción de scripts no firmados
::   -File               : ruta al script (mismo directorio que este .bat)
echo   Iniciando instalador...
echo.

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"

:: Si PowerShell terminó con código de error, mantener la ventana abierta
if errorlevel 1 (
    echo.
    echo   ══════════════════════════════════════════
    echo   La instalacion termino con errores ^(codigo %errorlevel%^).
    echo   Revisa los mensajes anteriores para más información.
    echo   ══════════════════════════════════════════
    echo.
    pause
)

endlocal
