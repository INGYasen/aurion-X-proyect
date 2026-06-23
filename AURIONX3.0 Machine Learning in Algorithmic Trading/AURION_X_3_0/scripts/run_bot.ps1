# AURION X 4.0 HIBRIDO — Arranque del bot con venv v4 activado.
# Equivalente a:
#   . .\scripts\activate_v4.ps1
#   python core/main.py
#
# Uso (desde la raíz del proyecto AURION_X_3_0):
#
#   .\scripts\run_bot.ps1                  # log INFO, config default
#   .\scripts\run_bot.ps1 -LogLevel DEBUG  # log DEBUG
#   .\scripts\run_bot.ps1 -ConfigPath "config/otro.yaml"
#
# El script:
#   1) Verifica que el venv v4 exista en D:\AURION_DATA\venv_v4.
#   2) Activa el venv y exporta variables de cache hacia D:.
#   3) Cambia al directorio raíz del proyecto (donde está core/main.py).
#   4) Lanza el bot.
#   5) Devuelve el exit code original al PowerShell.

param(
    [string]$LogLevel  = "INFO",
    [string]$ConfigPath = ""
)

$ErrorActionPreference = "Stop"

$scriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
$venvPath   = "D:\AURION_DATA\venv_v4"
$activate   = Join-Path $venvPath "Scripts\Activate.ps1"

if (-not (Test-Path $activate)) {
    Write-Error "venv v4 no encontrado en $venvPath. Crea el entorno primero (requirements-v4.txt)."
    exit 1
}

# Redirección de caches a D:
$env:PIP_CACHE_DIR    = "D:\AURION_DATA\pip_cache"
$env:TMP              = "D:\AURION_DATA\tmp"
$env:TEMP             = "D:\AURION_DATA\tmp"
$env:TORCH_HOME       = "D:\AURION_DATA\torch_hub"
$env:HF_HOME          = "D:\AURION_DATA\torch_hub\hf"
$env:XDG_CACHE_HOME   = "D:\AURION_DATA\cache"
$env:AURION_DATA_DIR  = "D:\AURION_DATA"
$env:AURION_MODELS    = "D:\AURION_DATA\models"
$env:AURION_REPORTS   = "D:\AURION_DATA\reports"

# Activa el venv
. $activate

# Cambia a la raíz del proyecto (donde están core/, market/, ml/, config/)
Set-Location $projectRoot

Write-Host ""
Write-Host "════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host " AURION X 4.0 HIBRIDO — arrancando bot"             -ForegroundColor Green
Write-Host "════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host " python  : $(python --version 2>&1)"
Write-Host " cwd     : $(Get-Location)"
Write-Host " log     : $LogLevel"
if ($ConfigPath) { Write-Host " config  : $ConfigPath" }
Write-Host ""

# Construye los argumentos para core/main.py
$mainArgs = @("core/main.py", "--log-level", $LogLevel)
if ($ConfigPath) {
    $mainArgs += @("--config", $ConfigPath)
}

# Arranca el bot (Ctrl+C cierra limpiamente)
python @mainArgs
$exitCode = $LASTEXITCODE

Write-Host ""
Write-Host "Bot terminó con exit code $exitCode" -ForegroundColor Yellow
exit $exitCode
