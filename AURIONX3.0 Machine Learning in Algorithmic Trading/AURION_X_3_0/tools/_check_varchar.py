import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sqlalchemy import text
from analytics.db_connection import init_db, new_session
init_db()
s = new_session()
r = s.execute(text(
    "SELECT column_name, data_type, character_maximum_length "
    "FROM information_schema.columns WHERE table_name='trades' "
    "AND data_type='character varying'"
)).fetchall()
for x in r:
    print(x)
