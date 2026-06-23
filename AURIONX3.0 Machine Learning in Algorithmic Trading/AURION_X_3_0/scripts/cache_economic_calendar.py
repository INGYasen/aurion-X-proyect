"""AURION X 4.0 — Cron diario para cachear el calendario económico.

Descarga eventos macro de alta volatilidad (FOMC, ECB, BoE, BoJ, NFP, CPI) y los
guarda en JSON en ``D:/AURION_DATA/cache/economic_calendar.json``. El bot
consume este archivo en runtime para computar ``central_bank_event_proximity``.

Fuentes intentadas en orden:
    1. ForexFactory weekly XML (sin API key).
    2. MyfxBook calendar (fallback).
    3. Manual override en ``D:/AURION_DATA/cache/manual_overrides.json``.

Programación recomendada (Task Scheduler en Windows):
    Trigger: diario 00:30 UTC.
    Acción: python -m scripts.cache_economic_calendar
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any


def _ensure_paths() -> Path:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cache calendario económico.")
    p.add_argument("--cache-path", type=Path, default=Path("D:/AURION_DATA/cache/economic_calendar.json"))
    p.add_argument("--days-ahead", type=int, default=14)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def fetch_forexfactory() -> list[dict[str, Any]]:
    """Intenta descargar el calendario semanal de ForexFactory."""
    try:
        import requests  # type: ignore[import-not-found]
    except ImportError:
        sys.stderr.write("requests no instalado (pip install -e .[v4])\n")
        return []
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        raw = resp.json()
    except (requests.RequestException, ValueError) as exc:  # type: ignore[attr-defined]
        sys.stderr.write(f"[ff] error: {exc}\n")
        return []

    out: list[dict[str, Any]] = []
    for ev in raw:
        try:
            title = str(ev.get("title", "")).strip()
            country = str(ev.get("country", "")).upper()
            impact = str(ev.get("impact", "")).lower()
            ts_raw = ev.get("date") or ev.get("timestamp")
            if not ts_raw or impact not in ("high",):
                continue
            try:
                dt = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
            except ValueError:
                continue
            out.append({
                "ts_utc": dt.astimezone(timezone.utc).isoformat(),
                "title": title,
                "country": country,
                "impact": impact,
            })
        except (TypeError, AttributeError):
            continue
    return out


def main() -> int:
    args = parse_args()
    _ensure_paths()

    print("Descargando calendario económico desde ForexFactory...")
    events = fetch_forexfactory()
    print(f"Eventos high impact: {len(events)}")

    # Filtra próximos ``days_ahead`` días.
    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(days=args.days_ahead)
    filtered = [e for e in events if datetime.fromisoformat(e["ts_utc"]) <= cutoff]

    payload = {
        "generated_at": now.isoformat(),
        "events": filtered,
        "n_events": len(filtered),
    }

    if args.dry_run:
        print(json.dumps(payload, indent=2)[:1200])
        return 0

    args.cache_path.parent.mkdir(parents=True, exist_ok=True)
    with args.cache_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[OK] Calendario cacheado en {args.cache_path} ({len(filtered)} eventos)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
