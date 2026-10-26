#!/usr/bin/env bash
# Evidencias Octave del examen modelo U2 (un script = un ejercicio).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OCT="$ROOT/octave"
EVID="$ROOT/evidencias"
cd "$OCT"
scripts=(
  ejercicio_01_newton_puntofijo.m
  ejercicio_02_jacobi_seidel.m
  ejercicio_03_newton_interp.m
  ejercicio_04_minimos.m
  ejercicio_05_lagrange.m
)
for script in "${scripts[@]}"; do
  base="${script%.m}"
  export MN_EVID_DIR="$EVID/$base"
  mkdir -p "$MN_EVID_DIR"
  echo ">> Octave: $base"
  octave --no-gui --quiet "$script"
done
echo "Listo. Evidencias en $EVID"
