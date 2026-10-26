#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OCT="$ROOT/octave"
EVID="$ROOT/evidencias"
mkdir -p "$EVID"
cd "$OCT"
for script in ejercicio_*.m; do
  base="${script%.m}"
  export MN_EVID_DIR="$EVID/$base"
  mkdir -p "$MN_EVID_DIR"
  echo ">> Octave: $base"
  octave --no-gui --quiet "$script"
done
echo "Listo. Evidencias en $EVID"
