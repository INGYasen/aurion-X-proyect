# AURION X 4.0 — activa el venv aislado en D:\AURION_DATA\venv_v4
# y redirige TODOS los caches (pip, torch, hf, tmp, xdg) a disco D.
#
# Uso (desde la raíz del proyecto):
#
#   . .\scripts\activate_v4.ps1
#
# Después podrás ejecutar tests, entrenamientos y scripts con todas las
# dependencias v4 disponibles y sin escribir caches en C:.

$venvPath = 'D:\AURION_DATA\venv_v4'
$activate = Join-Path $venvPath 'Scripts\Activate.ps1'

if (-not (Test-Path $activate)) {
    Write-Error "venv v4 no encontrado en $venvPath. Ejecuta primero la instalación (ver requirements-v4.txt)."
    return
}

# Redirección de caches a D:
$env:PIP_CACHE_DIR    = 'D:\AURION_DATA\pip_cache'
$env:TMP              = 'D:\AURION_DATA\tmp'
$env:TEMP             = 'D:\AURION_DATA\tmp'
$env:TORCH_HOME       = 'D:\AURION_DATA\torch_hub'
$env:HF_HOME          = 'D:\AURION_DATA\torch_hub\hf'
$env:XDG_CACHE_HOME   = 'D:\AURION_DATA\cache'
$env:AURION_DATA_DIR  = 'D:\AURION_DATA'
$env:AURION_MODELS    = 'D:\AURION_DATA\models'
$env:AURION_REPORTS   = 'D:\AURION_DATA\reports'

# Activa el venv
. $activate

Write-Host ''
Write-Host '════════════════════════════════════════════════════════' -ForegroundColor Cyan
Write-Host ' AURION X 4.0 HÍBRIDO — venv v4 activado' -ForegroundColor Green
Write-Host '════════════════════════════════════════════════════════' -ForegroundColor Cyan
Write-Host " venv         : $venvPath"
Write-Host " python       : $(python --version 2>&1)"
Write-Host " pip cache    : $env:PIP_CACHE_DIR"
Write-Host " torch home   : $env:TORCH_HOME"
Write-Host " models dir   : $env:AURION_MODELS"
Write-Host " reports dir  : $env:AURION_REPORTS"
Write-Host ''
Write-Host ' Comandos útiles:' -ForegroundColor Yellow
Write-Host '   pytest tests/                                # suite completa'
Write-Host '   python scripts/train_patchtst_xauusd.py      # entrena PatchTST'
Write-Host '   python scripts/train_anomaly_autoencoder.py  # entrena Anomaly AE'
Write-Host '   python scripts/train_meta_stacker.py         # entrena MetaStacker'
Write-Host '   python scripts/train_drl_exit_xauusd.py      # entrena PPO Exit'
Write-Host ''
