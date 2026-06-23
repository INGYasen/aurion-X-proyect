# AURION X 4.0 HIBRIDO — Reentrena los 4 modelos v4 en el orden correcto.
#
# Orden de entrenamiento (las dependencias importan):
#   1) PatchTST          — independiente, requiere M1 desde MT5
#   2) Anomaly AE        — independiente, requiere snapshots de microestructura
#   3) MetaStacker       — requiere PatchTST entrenado (genera prob_patchtst)
#   4) PPO Exit Agent    — requiere trades cerrados + M1 alineadas
#
# Uso (desde la raíz del proyecto):
#
#   .\scripts\retrain_all_v4.ps1                       # entrenamiento estándar
#   .\scripts\retrain_all_v4.ps1 -PatchtstEpochs 40    # epochs PatchTST custom
#   .\scripts\retrain_all_v4.ps1 -DRLSteps 500000      # más timesteps PPO
#   .\scripts\retrain_all_v4.ps1 -Symbol XAUUSD -Months 6
#   .\scripts\retrain_all_v4.ps1 -SkipPatchtst         # saltar un componente
#
# Al terminar, los .pt/.pkl/.zip viven en D:\AURION_DATA\models\ y los reportes
# JSON en D:\AURION_DATA\reports\.

param(
    [string]$Symbol           = "XAUUSD",
    [int]   $Months           = 6,
    [int]   $PatchtstEpochs   = 20,
    [int]   $PatchtstBatchSize= 256,
    [int]   $PatchtstFolds    = 5,
    [int]   $AnomalyDays      = 8,
    [int]   $AnomalyEpochs    = 80,
    [int]   $MinTrades        = 150,
    [int]   $DRLSteps         = 300000,
    [int]   $DRLEvalInterval  = 50000,
    [switch]$SkipPatchtst,
    [switch]$SkipAnomaly,
    [switch]$SkipMetaStacker,
    [switch]$SkipDRL,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$scriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
$activate    = "D:\AURION_DATA\venv_v4\Scripts\Activate.ps1"

if (-not (Test-Path $activate)) {
    Write-Error "venv v4 no encontrado. Crea el entorno primero (requirements-v4.txt)."
    exit 1
}

# Caches a D:
$env:PIP_CACHE_DIR    = "D:\AURION_DATA\pip_cache"
$env:TMP              = "D:\AURION_DATA\tmp"
$env:TEMP             = "D:\AURION_DATA\tmp"
$env:TORCH_HOME       = "D:\AURION_DATA\torch_hub"
$env:XDG_CACHE_HOME   = "D:\AURION_DATA\cache"

. $activate
Set-Location $projectRoot

$dryArg = if ($DryRun) { "--dry-run" } else { "" }
$started = Get-Date
$failed  = @()

function Run-Step {
    param([string]$Name, [scriptblock]$Block)
    Write-Host ""
    Write-Host "════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host " [$Name] $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Green
    Write-Host "════════════════════════════════════════════════════════" -ForegroundColor Cyan
    try {
        & $Block
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "[$Name] exit code $LASTEXITCODE (no promovido o gates no pasados)"
            $script:failed += $Name
        }
    } catch {
        Write-Warning "[$Name] excepción: $_"
        $script:failed += $Name
    }
}

# ============================================================================
# 1) PatchTST
# ============================================================================
if (-not $SkipPatchtst) {
    Run-Step "PatchTST" {
        $args = @(
            "scripts/train_patchtst_xauusd.py",
            "--config", "config/parameters.yaml",
            "--symbol", $Symbol,
            "--months", $Months,
            "--epochs", $PatchtstEpochs,
            "--batch-size", $PatchtstBatchSize,
            "--folds", $PatchtstFolds
        )
        if ($DryRun) { $args += "--dry-run" }
        python @args
    }
} else { Write-Host "[skip] PatchTST" -ForegroundColor DarkGray }

# ============================================================================
# 2) Anomaly Autoencoder
# ============================================================================
if (-not $SkipAnomaly) {
    Run-Step "Anomaly AE" {
        python scripts/train_anomaly_autoencoder.py `
            --config config/parameters.yaml `
            --symbol $Symbol `
            --days $AnomalyDays `
            --epochs $AnomalyEpochs `
            --batch-size 512
    }
} else { Write-Host "[skip] Anomaly AE" -ForegroundColor DarkGray }

# ============================================================================
# 3) MetaStacker (requiere PatchTST entrenado para producir prob_patchtst)
# ============================================================================
if (-not $SkipMetaStacker) {
    Run-Step "MetaStacker" {
        python scripts/train_meta_stacker.py `
            --config config/parameters.yaml `
            --symbol $Symbol `
            --min-trades $MinTrades `
            --folds 5
    }
} else { Write-Host "[skip] MetaStacker" -ForegroundColor DarkGray }

# ============================================================================
# 4) PPO Exit Agent
# ============================================================================
if (-not $SkipDRL) {
    Run-Step "PPO Exit Agent" {
        $args = @(
            "scripts/train_drl_exit_xauusd.py",
            "--config", "config/parameters.yaml",
            "--symbol", $Symbol,
            "--months", $Months,
            "--min-trades", $MinTrades,
            "--total-timesteps", $DRLSteps,
            "--eval-interval", $DRLEvalInterval,
            "--device", "cpu"
        )
        if ($DryRun) { $args += "--dry-run" }
        python @args
    }
} else { Write-Host "[skip] PPO Exit Agent" -ForegroundColor DarkGray }

$dur = (Get-Date) - $started
Write-Host ""
Write-Host "════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host " RESUMEN reentrenamiento v4" -ForegroundColor Yellow
Write-Host "════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host (" duración total : {0:N1} min" -f $dur.TotalMinutes)
Write-Host (" símbolo        : {0}" -f $Symbol)
Write-Host (" meses M1       : {0}" -f $Months)
Write-Host (" fallos         : {0}" -f $(if ($failed.Count) { $failed -join ", " } else { "ninguno" }))

# Listar modelos generados
Write-Host ""
Write-Host " Modelos en D:\AURION_DATA\models\ :" -ForegroundColor Yellow
Get-ChildItem 'D:\AURION_DATA\models' -File |
    Where-Object { $_.Name -like "*v4*" } |
    Sort-Object LastWriteTime -Descending |
    Select-Object Name, @{n="MB"; e={[math]::Round($_.Length/1MB, 2)}}, LastWriteTime |
    Format-Table -AutoSize

exit ($failed.Count)
