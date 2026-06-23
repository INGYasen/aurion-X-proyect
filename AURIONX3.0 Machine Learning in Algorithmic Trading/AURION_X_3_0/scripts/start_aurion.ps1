# =============================================================================
# AURION X 4.2.5 — Launcher robusto
# =============================================================================
# Inicia (o reinicia) toda la pila en orden:
#   1) Verifica/levanta PostgreSQL (Docker contenedor `aurionx_db2`)
#   2) Lanza dashboard_server (necesario para visualizar SMC en MT5)
#   3) Lanza el bot principal (core.main)
#
# Uso:
#   .\scripts\start_aurion.ps1                # arranca todo
#   .\scripts\start_aurion.ps1 -RestartBot    # solo reinicia bot
#   .\scripts\start_aurion.ps1 -StopAll       # detiene bot y dashboard
# =============================================================================

[CmdletBinding()]
param(
    [switch]$RestartBot,
    [switch]$StopAll,
    [int]$DbWaitSec = 60
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$logs = Join-Path $root "logs"
$venvPython = "D:\AURION_DATA\venv_v4\Scripts\python.exe"

if (-not (Test-Path $logs)) { New-Item -ItemType Directory -Path $logs | Out-Null }
if (-not (Test-Path $venvPython)) {
    Write-Host "❌ Python venv no encontrado en $venvPython" -ForegroundColor Red
    exit 1
}

function Stop-AurionPython {
    Write-Host "🛑 Deteniendo procesos Python existentes..." -ForegroundColor Yellow
    Get-Process python -ErrorAction SilentlyContinue | ForEach-Object {
        try {
            $cmd = (Get-CimInstance Win32_Process -Filter "ProcessId=$($_.Id)" -ErrorAction SilentlyContinue).CommandLine
            if ($cmd -match "core\.main|dashboard_server") {
                Stop-Process -Id $_.Id -Force
                Write-Host "   killed PID $($_.Id) ($($cmd -replace '.*python.exe[^a-z]*', ''))"
            }
        } catch {}
    }
    Start-Sleep -Seconds 3
}

function Test-Postgres {
    try {
        Test-NetConnection -ComputerName 127.0.0.1 -Port 5433 -WarningAction SilentlyContinue -InformationLevel Quiet
    } catch { $false }
}

function Ensure-Postgres {
    Write-Host "🐘 Verificando PostgreSQL (5433)..." -ForegroundColor Cyan
    if (Test-Postgres) {
        Write-Host "   ✅ Ya estaba arriba" -ForegroundColor Green
        return $true
    }

    Write-Host "   ⚠️  Caído. Iniciando Docker Desktop si hace falta..." -ForegroundColor Yellow
    $dockerProc = Get-Process "Docker Desktop" -ErrorAction SilentlyContinue
    if (-not $dockerProc) {
        $dockerPath = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
        if (Test-Path $dockerPath) {
            Start-Process -FilePath $dockerPath -WindowStyle Hidden
            Write-Host "   ⏳ Esperando 20s a que arranque Docker Desktop..."
            Start-Sleep -Seconds 20
        }
    }

    Write-Host "   🚀 docker compose up -d ..."
    Push-Location $root
    try {
        docker compose -f docker-compose.db2.yml up -d 2>&1 | Select-Object -Last 4 | ForEach-Object { Write-Host "      $_" }
    } finally { Pop-Location }

    Write-Host "   ⏳ Esperando hasta ${DbWaitSec}s a que PostgreSQL acepte conexiones..."
    $start = Get-Date
    while ((Get-Date) - $start -lt [TimeSpan]::FromSeconds($DbWaitSec)) {
        if (Test-Postgres) {
            Write-Host "   ✅ PostgreSQL aceptando conexiones en 5433" -ForegroundColor Green
            return $true
        }
        Start-Sleep -Seconds 2
    }
    Write-Host "   ❌ Timeout esperando a PostgreSQL" -ForegroundColor Red
    return $false
}

function Start-Dashboard {
    Write-Host "📊 Lanzando dashboard_server..." -ForegroundColor Cyan
    $env:PYTHONUNBUFFERED = "1"
    $proc = Start-Process -FilePath $venvPython `
        -ArgumentList "-u","-m","analytics.dashboard_server" `
        -WorkingDirectory $root `
        -RedirectStandardOutput (Join-Path $logs "dashboard.log") `
        -RedirectStandardError (Join-Path $logs "dashboard.err") `
        -PassThru -WindowStyle Hidden
    Write-Host "   PID=$($proc.Id)" -ForegroundColor Green
    Start-Sleep -Seconds 6
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:8765/api/smc/XAUUSD.txt" -UseBasicParsing -TimeoutSec 5
        Write-Host "   ✅ Dashboard responde ($($r.StatusCode), $($r.RawContentLength) bytes)" -ForegroundColor Green
    } catch {
        Write-Host "   ⚠️  Dashboard aún no responde — revisar logs/dashboard.err" -ForegroundColor Yellow
    }
}

function Start-Bot {
    Write-Host "🤖 Lanzando bot principal..." -ForegroundColor Cyan
    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $current = Join-Path $logs "bot_live_v2.log"
    if (Test-Path $current) {
        Move-Item $current (Join-Path $logs "bot_live_$ts.log") -Force
    }
    $env:PYTHONUNBUFFERED = "1"
    $proc = Start-Process -FilePath $venvPython `
        -ArgumentList "-u","-m","core.main" `
        -WorkingDirectory $root `
        -RedirectStandardOutput $current `
        -RedirectStandardError (Join-Path $logs "bot_live_v2.err") `
        -PassThru -WindowStyle Hidden
    Write-Host "   PID=$($proc.Id)" -ForegroundColor Green
    return $proc.Id
}

if ($StopAll) {
    Stop-AurionPython
    Write-Host "✅ Detenido" -ForegroundColor Green
    exit 0
}

if ($RestartBot) {
    Get-Process python -ErrorAction SilentlyContinue | ForEach-Object {
        $cmd = (Get-CimInstance Win32_Process -Filter "ProcessId=$($_.Id)" -ErrorAction SilentlyContinue).CommandLine
        if ($cmd -match "core\.main") { Stop-Process -Id $_.Id -Force; Write-Host "Killed bot PID $($_.Id)" }
    }
    Start-Sleep -Seconds 3
    if (-not (Ensure-Postgres)) { exit 1 }
    Start-Bot | Out-Null
    Write-Host "✅ Bot reiniciado" -ForegroundColor Green
    exit 0
}

Write-Host ""
Write-Host "================================================================" -ForegroundColor Magenta
Write-Host " AURION X 4.2.5 — Launcher" -ForegroundColor Magenta
Write-Host "================================================================" -ForegroundColor Magenta

Stop-AurionPython
if (-not (Ensure-Postgres)) {
    Write-Host "❌ No se pudo levantar PostgreSQL. Abortando." -ForegroundColor Red
    exit 1
}
Start-Dashboard
$botPid = Start-Bot
Start-Sleep -Seconds 25

Write-Host ""
Write-Host "================================================================" -ForegroundColor Magenta
Write-Host " Verificación final" -ForegroundColor Magenta
Write-Host "================================================================" -ForegroundColor Magenta

$log = Join-Path $logs "bot_live_v2.log"
if (Test-Path $log) {
    $lines = Get-Content $log
    Write-Host "   Bot log: $($lines.Count) líneas"
    $err = $lines | Where-Object { $_ -match "Traceback|ERROR \|" }
    if ($err) {
        Write-Host "   ❌ ERRORES detectados:" -ForegroundColor Red
        $err | Select-Object -First 5 | ForEach-Object { Write-Host "      $_" }
    } else {
        Write-Host "   ✅ Sin errores" -ForegroundColor Green
    }
    $cg = $lines | Where-Object { $_ -match "CHOP_GUARD\] activo" } | Select-Object -First 1
    if ($cg) { Write-Host "   ✅ ChopGuard activo" -ForegroundColor Green }
    $ml = $lines | Where-Object { $_ -match "model_XAUUSD.*n_features=25" } | Select-Object -First 1
    if ($ml) { Write-Host "   ✅ Modelo v2 cargado" -ForegroundColor Green }
}

Write-Host ""
Write-Host "✅ AURION X arrancado correctamente. Bot PID=$botPid" -ForegroundColor Green
Write-Host "   Tail del log:   Get-Content $log -Tail 5 -Wait" -ForegroundColor Gray
