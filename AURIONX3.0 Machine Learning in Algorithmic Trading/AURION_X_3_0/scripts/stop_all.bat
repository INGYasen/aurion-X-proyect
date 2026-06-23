@echo off
REM ============================================================
REM AURION X 4.1 - Detiene TODOS los procesos AURION limpiamente.
REM ============================================================
echo.
echo ============================================================
echo  AURION X 4.1 - STOP ALL
echo ============================================================
echo.
echo Matando todos los procesos Python...
powershell -NoProfile -Command "Get-Process python -ErrorAction SilentlyContinue | ForEach-Object { Write-Host ('  PID=' + $_.Id + ' RAM=' + [math]::Round($_.WorkingSet/1MB,1) + 'MB'); Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue }"
timeout /t 2 /nobreak >nul
powershell -NoProfile -Command "$alive = Get-Process python -ErrorAction SilentlyContinue; if ($alive) { Write-Host ('AVISO: aun quedan ' + $alive.Count + ' procesos vivos') } else { Write-Host 'OK: sistema limpio' }"
echo.
pause
