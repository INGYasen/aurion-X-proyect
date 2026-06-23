"""Audita parameters.yaml vs uso real en codigo.

AURION X 4.3 (7-jun-2026): 710 lineas de config en YAML => alto riesgo de
curve-fitting y parametros muertos. Este script:

    1. Parsea recursivamente parameters.yaml y extrae TODAS las keys (path completo).
    2. Para cada key, busca su nombre (terminal) en codigo .py.
    3. Reporta:
        - DEAD: keys cuyo nombre nunca aparece en codigo.
        - SUSPECT: keys que aparecen <2 veces (revisar manualmente).

Notas:
    - Algunas keys son consumidas con .get("foo", default) - el grep funciona OK.
    - Algunas keys son construidas dinamicamente; falsos positivos posibles
      (revisar antes de eliminar).

Uso:
    python -m tools.audit_yaml_dead_keys
    python -m tools.audit_yaml_dead_keys --remove-dead       # despues de revisar
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml  # noqa: E402


def walk_yaml(node, path: list[str]) -> list[tuple[str, str]]:
    """Devuelve [(full_path, leaf_key), ...] para cada nodo no-dict."""
    out = []
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(v, dict):
                out.extend(walk_yaml(v, path + [str(k)]))
            else:
                out.append((".".join(path + [str(k)]), str(k)))
    return out


def count_occurrences_in_code(key: str, root: Path) -> int:
    """Cuenta veces que aparece la key en archivos .py."""
    count = 0
    # Buscar formas tipicas: "foo": ..., 'foo': ..., .get("foo"), get('foo'), ["foo"]
    patterns = [
        rf'["\']\b{re.escape(key)}\b["\']',
    ]
    pattern = re.compile("|".join(patterns))
    for py_file in root.rglob("*.py"):
        # Skip self
        if py_file.name == "audit_yaml_dead_keys.py":
            continue
        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        count += len(pattern.findall(content))
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml-path", default="config/parameters.yaml")
    parser.add_argument("--code-root", default=".")
    parser.add_argument("--dead-threshold", type=int, default=0,
                          help="Keys con <=N apariciones se marcan dead")
    parser.add_argument("--suspect-threshold", type=int, default=1,
                          help="Keys con <=N apariciones se marcan suspect")
    args = parser.parse_args()

    yaml_path = Path(args.yaml_path)
    code_root = Path(args.code_root)

    print(f"Cargando YAML: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    print(f"Escaneando codigo: {code_root}")
    keys = walk_yaml(cfg, [])
    print(f"Total keys YAML: {len(keys)}\n")

    dead = []
    suspect = []
    seen_keys: set[str] = set()
    for full_path, leaf in keys:
        if leaf in seen_keys:
            continue  # ya analizado este nombre terminal (mismo nombre en multiples paths)
        seen_keys.add(leaf)
        n = count_occurrences_in_code(leaf, code_root)
        if n <= args.dead_threshold:
            dead.append((full_path, leaf, n))
        elif n <= args.suspect_threshold:
            suspect.append((full_path, leaf, n))

    print(f"DEAD KEYS (n <= {args.dead_threshold}): {len(dead)}")
    print("-" * 80)
    for fp, leaf, n in sorted(dead, key=lambda t: t[0]):
        print(f"  {n:>3}x  {fp}")

    print(f"\nSUSPECT KEYS (n <= {args.suspect_threshold}): {len(suspect)}")
    print("-" * 80)
    for fp, leaf, n in sorted(suspect, key=lambda t: t[0]):
        print(f"  {n:>3}x  {fp}")

    total = len(dead) + len(suspect)
    print(f"\nResumen: {len(dead)} dead + {len(suspect)} suspect = {total} candidatos de revision")
    print(f"de {len(seen_keys)} keys unicas en YAML")


if __name__ == "__main__":
    main()
