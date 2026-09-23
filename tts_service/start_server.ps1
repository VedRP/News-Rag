# Starts the IndicF5 TTS service with all the fixes from README.md applied:
# - FFmpeg shared-build bin/ on PATH (torchcodec needs its DLLs)
# - Offline mode (once weights are cached -- see README.md for why this also
#   avoids a real hang, not just saves time)
#
# Run from anywhere: powershell -File tts_service\start_server.ps1

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

$ffmpegSharedBin = Get-ChildItem -Path "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" `
    -Filter "avcodec-*.dll" -Recurse -ErrorAction SilentlyContinue |
    Select-Object -First 1 -ExpandProperty DirectoryName

if ($ffmpegSharedBin) {
    $env:PATH = "$ffmpegSharedBin;$env:PATH"
} else {
    Write-Warning "FFmpeg shared build not found -- torchcodec audio loading may fail. See README.md step 4."
}

$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

Write-Host "Starting IndicF5 TTS service on http://localhost:8100 (offline mode -- uses cached weights only)"
Set-Location $scriptDir
& "$scriptDir\.venv\Scripts\python.exe" -m uvicorn server:app --port 8100
