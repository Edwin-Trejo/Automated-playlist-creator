import sqlite3
from pathlib import Path

COLUMNS = ("spotify_id", "uri", "name", "artists", "preview_found", "genre", "genre_conf",
           "danceability", "energy", "valence", "acousticness", "tempo", "model_version", "analyzed_at")

SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks (
    spotify_id TEXT PRIMARY KEY,
    uri TEXT NOT NULL,
    name TEXT NOT NULL,
    artists TEXT NOT NULL,
    preview_found INTEGER NOT NULL DEFAULT 0,
    genre TEXT, genre_conf REAL,
    danceability REAL, energy REAL, valence REAL, acousticness REAL, tempo REAL,
    model_version TEXT, analyzed_at TEXT
)
"""


class TrackStore:
    """SQLite cache of analyzed liked songs. Opens a connection per call so it is safe across threads."""

    def __init__(self, path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(SCHEMA)

    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def upsert(self, row):
        values = [row.get(c) for c in COLUMNS]
        updates = ", ".join(f"{c}=excluded.{c}" for c in COLUMNS[1:])
        sql = (f"INSERT INTO tracks ({', '.join(COLUMNS)}) VALUES ({', '.join('?' * len(COLUMNS))}) "
               f"ON CONFLICT(spotify_id) DO UPDATE SET {updates}")
        with self._connect() as conn:
            conn.execute(sql, values)

    def get(self, spotify_id):
        with self._connect() as conn:
            r = conn.execute("SELECT * FROM tracks WHERE spotify_id = ?", (spotify_id,)).fetchone()
        return dict(r) if r else None

    def get_many(self, ids):
        if not ids:
            return []
        with self._connect() as conn:
            rows = conn.execute(f"SELECT * FROM tracks WHERE spotify_id IN ({', '.join('?' * len(ids))})", ids).fetchall()
        by_id = {r["spotify_id"]: dict(r) for r in rows}
        return [by_id[i] for i in ids if i in by_id]

    def ids_with_version(self, version):
        with self._connect() as conn:
            rows = conn.execute("SELECT spotify_id FROM tracks WHERE model_version = ?", (version,)).fetchall()
        return {r["spotify_id"] for r in rows}

    def analyzed_tracks(self):
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM tracks WHERE preview_found = 1 ORDER BY spotify_id").fetchall()
        return [dict(r) for r in rows]

    def counts(self):
        with self._connect() as conn:
            total, analyzed = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(preview_found), 0) FROM tracks").fetchone()
        return {"total": total, "analyzed": analyzed, "no_preview": total - analyzed}
