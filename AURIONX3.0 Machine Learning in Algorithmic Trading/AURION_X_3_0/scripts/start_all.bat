@echo off
REM ============================================================
REM AURION X 4.1 - Launcher unificado (CMD / doble click).
REM Mata cualquier python previo, arranca Dashboard + Bot con venv_v4
REM en DOS ventanas separadas con titulo claro.
REM ============================================================
setlocal

set "PROJECT_ROOT=%~dp0.."
set "VENV=D:\AURION_DATA\venv_v4"
set "PY=%VENV%\Scripts\python.exe"

if not exist "%PY%" (
    echo ERROR: venv v4 no encontrado en %VENV%
    echo Crea el entorno primero con: pip install -r requirements-v4.txt
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  AURION X 4.1 - Launcher unificado
echo ============================================================
echo  venv : %VENV%
echo  raiz : %PROJECT_ROOT%
echo.

REM Paso 1 - Limpieza: matar cualquier python previo para evitar duplicados
echo [1/3] Limpiando procesos Python previos...
powershell -NoProfile -Command "Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force"
timeout /t 2 /nobreak >nul

REM Variables de cache hacia D: para no llenar C:
set "PIP_CACHE_DIR=D:\AURION_DATA\pip_cache"
set "TMP=D:\AURION_DATA\tmp"
set "TEMP=D:\AURION_DATA\tmp"
set "TORCH_HOME=D:\AURION_DATA\torch_hub"
set "HF_HOME=D:\AURION_DATA\torch_hub\hf"
set "XDG_CACHE_HOME=D:\AURION_DATA\cache"
set "AURION_DATA_DIR=D:\AURION_DATA"
set "AURION_MODELS=D:\AURION_DATA\models"
set "AURION_REPORTS=D:\AURION_DATA\reports"

REM Paso 2 - Dashboard en ventana titulada
echo [2/3] Arrancando DASHBOARD en nueva ventana...
start "AURION DASHBOARD :8765" /D "%PROJECT_ROOT%" cmd /k ""%PY%" -m analytics.dashboard_server"
timeout /t 5 /nobreak >nul

REM Paso 3 - Bot en ventana titulada
echo [3/3] Arrancando BOT en nueva ventana...
start "AURION BOT (venv_v4 + SMC)" /D "%PROJECT_ROOT%" cmd /k ""%PY%" core\main.py --log-level INFO"

echo.
echo ============================================================
echo  OK - Sistema arrancado:
echo    - Ventana DASHBOARD escuchando en http://127.0.0.1:8765
echo    - Ventana BOT operando XAUUSD con SMC institucional
echo.
echo  Endpoints utiles:
echo    http://127.0.0.1:8765/                 - dashboard
echo    http://127.0.0.1:8765/smc/XAUUSD       - visual SMC
echo    http://127.0.0.1:8765/api/smc/XAUUSD.txt - datos para EA MT5
echo.
echo  Para parar todo: cierra las dos ventanas que se abrieron
echo                   o ejecuta scripts\stop_all.bat
echo ============================================================
echo.
pause
endlocal
