# AURION X 4.0 HYBRID — backup pre-migración de PostgreSQL.
# Genera dumps SQL de las dos BD (principal + db2) antes de tocar nada del v4.
#
# Uso (PowerShell elevado):
#   .\scripts\backup_pre_v4.ps1
#
# Requiere docker-compose corriendo. Credenciales se leen de .env del compose.

$ErrorActionPreference = "Stop"

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$BackupDir = "D:/AURION_DATA/backups"
New-Item -Path $BackupDir -ItemType Directory -Force | Out-Null

Write-Host "[v4-backup] Iniciando backup pre-migración ($Stamp)..." -ForegroundColor Cyan

# Carga credenciales del .env (busca PG_USER, PG_PASSWORD, PG_DB, PG_DB2)
$envPath = Join-Path $PSScriptRoot ".." ".env"
if (Test-Path $envPath) {
    Get-Content $envPath | Where-Object { $_ -match "^[A-Z_]+=" } | ForEach-Object {
        $parts = $_ -split "=", 2
        Set-Item -Path "Env:$($parts[0].Trim())" -Value $parts[1].Trim()
    }
}

$pgUser = if ($env:PG_USER) { $env:PG_USER } else { "aurion" }
$pgDb1 = if ($env:PG_DB) { $env:PG_DB } else { "aurion_x" }
$pgDb2 = if ($env:PG_DB2) { $env:PG_DB2 } else { "aurion_x_db2" }
$container1 = if ($env:PG_CONTAINER_1) { $env:PG_CONTAINER_1 } else { "aurion_pg" }
$container2 = if ($env:PG_CONTAINER_2) { $env:PG_CONTAINER_2 } else { "aurion_pg_db2" }

function Dump-Container([string]$container, [string]$db, [string]$alias) {
    $outFile = "$BackupDir/pre_v4_${alias}_${Stamp}.sql"
    Write-Host "[v4-backup] Dumping $container ($db) -> $outFile"
    docker exec $container pg_dump -U $pgUser -d $db --no-owner --no-acl | Out-File -Encoding UTF8 -FilePath $outFile
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "[v4-backup] Fallo al dumpear $container (db=$db). Continuando."
        return
    }
    $size = (Get-Item $outFile).Length / 1MB
    Write-Host ("[v4-backup] OK: {0:N2} MB" -f $size) -ForegroundColor Green
}

# Verifica que los contenedores estén arriba
$running = docker ps --format "{{.Names}}"

if ($running -contains $container1) {
    Dump-Container $container1 $pgDb1 "main"
} else {
    Write-Warning "[v4-backup] Contenedor $container1 NO está corriendo. Saltando BD principal."
}

if ($running -contains $container2) {
    Dump-Container $container2 $pgDb2 "db2"
} else {
    Write-Warning "[v4-backup] Contenedor $container2 NO está corriendo. Saltando BD db2."
}

Write-Host "[v4-backup] Listo. Backups en $BackupDir" -ForegroundColor Cyan
