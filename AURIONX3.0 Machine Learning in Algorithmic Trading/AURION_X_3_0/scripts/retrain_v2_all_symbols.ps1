# =====================================================================
# AURION X 3.0 - Reentrena modelos ML por simbolo con schema v2
# (25 features + ensemble RF+LR+XGB/LGBM, walk-forward, OOS gating)
#
# Uso:
#   .\scripts\retrain_v2_all_symbols.ps1
#   .\scripts\retrain_v2_all_symbols.ps1 -Symbols "XAUUSD,NDX100"
#   .\scripts\retrain_v2_all_symbols.ps1 -Limit 8000 -MinTrades 20
# =====================================================================

param(
    [int]    $Limit      = 5000,
    [string] $Symbols    = "",
    [int]    $MinTrades  = 12,
    [string] $Config     = ""
)

$ErrorActionPreference = "Stop"
$proj = Split-Path -Parent $PSScriptRoot
Set-Location $proj

$venvPy = Join-Path $proj ".venv\Scripts\python.exe"
if (Test-Path $venvPy) {
    $py = $venvPy
} else {
    $py = "python"
}

$pyArgs = @("-m", "scripts.retrain_v2_all_symbols", "--limit", $Limit, "--min-trades", $MinTrades)
if ($Symbols -ne "") { $pyArgs += @("--symbols", $Symbols) }
if ($Config  -ne "") { $pyArgs += @("--config",  $Config)  }

Write-Host ""
Write-Host "[AURION X] Lanzando reentrenamiento v2 con: $py $($pyArgs -join ' ')"
Write-Host ""

& $py @pyArgs
$code = $LASTEXITCODE

Write-Host ""
if ($code -eq 0) {
    Write-Host "[AURION X] Reentrenamiento finalizado. Reinicia el bot para cargar los .pkl v2."
} else {
    Write-Host "[AURION X] Reentrenamiento terminado con codigo $code." -ForegroundColor Yellow
}
exit $code
