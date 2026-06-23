"""Watcher del bot: notifica cuando se abre/cierra un trade.

Sigue el log live del bot y emite una linea sentinel ASCII para cada
TRADE_ALLOWED / TRADE_CLOSED. Idempotente: solo notifica eventos nuevos
desde el momento del arranque (no historia).

Uso:
    python tools/_watch_trades.py
    python tools/_watch_trades.py --since-start   # incluye eventos del log actual
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

LOG_PATH = Path(r"D:\AURION_DATA\logs\bot_live.log")

OPEN_RE = re.compile(r"\[TRADE_ALLOWED\] symbol=(\S+) side=(\S+) lot=([\d.]+) "
                     r"ml_conf_raw=([\d.]+) effective_conf=([\d.]+).*?strategy=(\S+)")
CLOSE_RE = re.compile(r"\[TRADE_CLOSED\] symbol=(\S+) ticket=(\S+) dir=(\S+) "
                      r"lot=([\d.]+) exit=([\d.]+) pnl=([\-\d.]+) R=([\-\d.]+) "
                      r"reason=(\S+)")


def tail_follow(path: Path, from_start: bool = False):
    while not path.exists():
        print(f"[WATCH] esperando log: {path}", flush=True)
        time.sleep(2)
    with path.open("r", encoding="utf-8", errors="replace") as f:
        if not from_start:
            f.seek(0, 2)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.6)
                continue
            yield line.rstrip("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--since-start", action="store_true",
                        help="Tambien procesar eventos ya presentes en el log")
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print(f"  AURION TRADE WATCHER — monitorenado {LOG_PATH}", flush=True)
    print(f"  modo: {'desde inicio del log' if args.since_start else 'solo eventos nuevos'}", flush=True)
    print("=" * 70, flush=True)
    n_open = 0
    n_close = 0
    for line in tail_follow(LOG_PATH, from_start=args.since_start):
        m = OPEN_RE.search(line)
        if m:
            n_open += 1
            sym, side, lot, raw, eff, strat = m.groups()
            print(f"\n>>> TRADE_OPEN #{n_open} | {sym} {side} | lot={lot} | "
                  f"ml_raw={raw} eff={eff} | strategy={strat}", flush=True)
            continue
        m = CLOSE_RE.search(line)
        if m:
            n_close += 1
            sym, tic, side, lot, exit_p, pnl, r, reason = m.groups()
            arrow = "WIN" if float(pnl) > 0 else ("LOSS" if float(pnl) < 0 else "BE")
            print(f"\n<<< TRADE_CLOSE #{n_close} | {sym} {side} | "
                  f"pnl={pnl} R={r} | {arrow} | reason={reason} | ticket={tic}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[WATCH] detenido por usuario", flush=True)
        sys.exit(0)
