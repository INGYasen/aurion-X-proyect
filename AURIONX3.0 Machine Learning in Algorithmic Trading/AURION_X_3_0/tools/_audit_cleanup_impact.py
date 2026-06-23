"""Audita espacio NDX100/BTCUSD - version rapida."""
import sys
from pathlib import Path

from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent.parent))
from analytics.db_connection import init_db, new_session


def _fmt(n: float) -> str:
    return f"{n/1024:.2f} MB" if n >= 1024 else f"{n:.1f} KB"


print("=" * 70)
print(" CLEANUP IMPACT: NDX100 + BTCUSD")
print("=" * 70)

print("\n[1] MODELOS .pkl")
md = Path(r"D:\AURION_DATA\models")
sizes = {"XAUUSD": 0.0, "NDX100": 0.0, "BTCUSD": 0.0, "global/otros": 0.0}
for p in md.glob("*.pkl"):
    sz = p.stat().st_size / 1024
    if "XAUUSD" in p.name: sizes["XAUUSD"] += sz
    elif "NDX100" in p.name: sizes["NDX100"] += sz
    elif "BTCUSD" in p.name: sizes["BTCUSD"] += sz
    else: sizes["global/otros"] += sz
for sym, sz in sizes.items():
    print(f"  {sym:<15} {_fmt(sz)}")

print("\n[2] BACKUPS")
bd = md / "backups"
bk_sizes = {"XAUUSD": 0.0, "NDX100": 0.0, "BTCUSD": 0.0, "global/otros": 0.0}
bk_count = {"XAUUSD": 0, "NDX100": 0, "BTCUSD": 0, "global/otros": 0}
if bd.exists():
    for p in bd.glob("*"):
        if not p.is_file(): continue
        sz = p.stat().st_size / 1024
        if "XAUUSD" in p.name: bk_sizes["XAUUSD"] += sz; bk_count["XAUUSD"] += 1
        elif "NDX100" in p.name: bk_sizes["NDX100"] += sz; bk_count["NDX100"] += 1
        elif "BTCUSD" in p.name: bk_sizes["BTCUSD"] += sz; bk_count["BTCUSD"] += 1
        else: bk_sizes["global/otros"] += sz; bk_count["global/otros"] += 1
for sym in bk_sizes:
    print(f"  {sym:<15} {bk_count[sym]:>3} files  {_fmt(bk_sizes[sym])}")

print("\n[3] TRADES EN DB (conteo - rapido)")
init_db()
s = new_session()
counts = {}
try:
    for sym in ("XAUUSD", "NDX100", "BTCUSD"):
        n = s.execute(text("SELECT COUNT(*) FROM trades WHERE symbol = :s"),
                      {"s": sym}).scalar() or 0
        counts[sym] = n
        print(f"  trades {sym:<10} {n:>5} filas")
finally:
    s.close()

print("\n[4] DECISION YA TOMADA en parameters.yaml:")
print(f'  symbols: ["XAUUSD"]  -> NDX100/BTCUSD ya BLOQUEADOS para procesamiento.')

print("\n" + "=" * 70)
print(" CONCLUSION HONESTA")
print("=" * 70)
liberable = (sizes["NDX100"] + sizes["BTCUSD"]
             + bk_sizes["NDX100"] + bk_sizes["BTCUSD"])
print(f"\n  Espacio liberable en disco D: {_fmt(liberable)}")
print(f"  (.pkl modelos + backups de NDX100+BTCUSD)")
print()
print(f"  Trades en DB a eliminar (si decides borrar): {counts.get('NDX100',0) + counts.get('BTCUSD',0)} filas")
print()
print(f"  Velocidad de aprendizaje XAUUSD: NO mejora.")
print(f"    El trainer ya hace WHERE symbol='XAUUSD' en cada query;")
print(f"    los trades NDX/BTC en la misma tabla NO ralentizan nada.")
print()
print(f"  Velocidad del bot live: NO mejora.")
print(f"    El bot solo procesa symbols=['XAUUSD']; NDX/BTC no consumen CPU.")
print()
print(f"  Lo unico que se gana: liberar {_fmt(liberable)} de disco D")
print(f"  y un poquito de RAM por no precargar los 2 modelos extra.")
