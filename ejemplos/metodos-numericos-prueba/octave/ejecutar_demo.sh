#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
octave --no-gui --quiet octave/biseccion_ejemplo.m
