import sqlite3
from pathlib import Path

COLUMNS = ("spotify_id", "uri", "name", "artists", "preview_found", "genre", "genre_conf",
           "danceability", "energy", "valence", "acousticness", "tempo", "model_version", "analyzed_at",
           "analysis_source")
# analysis_source: "audio" (preview analyzed by our models), "dataset" (Spotify's scores from dataset.csv)
# or "artist" (estimated from the artist's other songs). preview_found = 1 means the song has usable values.

LYRIC_COLUMNS = ("lyric_sad", "lyric_happy", "lyric_romantic", "lyric_angry", "lyric_party", "lyric_calm")
# Lyric mood scores (0..1, from src/ml/lyrics_mood.py) are written separately by set_lyrics, so re-analyzing a
# song's audio keeps them. lyrics_status: "found", "none" (not on LRCLIB) or "instrumental".
LYRICS_SCHEMA = {**{c: "REAL" for c in LYRIC_COLUMNS}, "lyrics_status": "TEXT", "lyrics_version": "TEXT"}

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
);
CREATE TABLE IF NOT EXISTS playlists (
    key TEXT PRIMARY KEY,
    playlist_id TEXT NOT NULL,
    name TEXT NOT NULL,
    url TEXT,
    updated_at TEXT
);
CREATE TABLE IF NOT EXISTS track_sources (
    source_id TEXT NOT NULL,
    spotify_id TEXT NOT NULL,
    PRIMARY KEY (source_id, spotify_id)
);
-- Libraries synced before sources existed only held Liked Songs.
INSERT INTO track_sources (source_id, spotify_id)
    SELECT 'liked', spotify_id FROM tracks WHERE NOT EXISTS (SELECT 1 FROM track_sources);
"""


class TrackStore:
    """SQLite cache of analyzed liked songs. Opens a connection per call so it is safe across threads."""

    def __init__(self, path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            existing = {r["name"] for r in conn.execute("PRAGMA table_info(tracks)")}
            # Columns added after the first release, for databases created before them.
            for column, kind in {"analysis_source": "TEXT", **LYRICS_SCHEMA}.items():
                if column not in existing:
                    conn.execute(f"ALTER TABLE tracks ADD COLUMN {column} {kind}")

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

    def ids_done(self, version):
        """Songs that need no more work: analyzed from audio or dataset.csv with this model version.

        Songs without a preview and artist-based estimates are left out, so every sync tries them again.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT spotify_id FROM tracks WHERE model_version = ? AND preview_found = 1 "
                "AND COALESCE(analysis_source, 'audio') != 'artist'", (version,)).fetchall()
        return {r["spotify_id"] for r in rows}

    def set_lyrics(self, spotify_id, status, scores, version):
        """Store a song's lyric mood scores ({"sad": 0.8, ...}, or None when it has no lyrics)."""
        scores = scores or {}
        values = [scores.get(c.removeprefix("lyric_")) for c in LYRIC_COLUMNS]
        sets = ", ".join(f"{c} = ?" for c in LYRIC_COLUMNS)
        with self._connect() as conn:
            conn.execute(f"UPDATE tracks SET {sets}, lyrics_status = ?, lyrics_version = ? WHERE spotify_id = ?",
                         [*values, status, version, spotify_id])

    def ids_with_lyrics(self, version):
        """Songs whose lyrics were already checked with this lyrics model version."""
        with self._connect() as conn:
            rows = conn.execute("SELECT spotify_id FROM tracks WHERE lyrics_version = ?", (version,)).fetchall()
        return {r["spotify_id"] for r in rows}

    @staticmethod
    def _source_filter(sources):
        """SQL condition + params limiting tracks to the given sources (None = all tracks)."""
        if sources is None:
            return "1", []
        sources = list(sources)
        if not sources:
            return "0", []
        return (f"spotify_id IN (SELECT spotify_id FROM track_sources WHERE source_id IN ({', '.join('?' * len(sources))}))",
                sources)

    def analyzed_tracks(self, sources=None):
        cond, params = self._source_filter(sources)
        with self._connect() as conn:
            rows = conn.execute(f"SELECT * FROM tracks WHERE preview_found = 1 AND {cond} ORDER BY spotify_id",
                                params).fetchall()
        return [dict(r) for r in rows]

    def set_source_tracks(self, source_id, spotify_ids):
        """Replace the songs a source (Liked Songs or a playlist) contains."""
        with self._connect() as conn:
            conn.execute("DELETE FROM track_sources WHERE source_id = ?", (source_id,))
            conn.executemany("INSERT OR IGNORE INTO track_sources (source_id, spotify_id) VALUES (?, ?)",
                             [(source_id, sid) for sid in spotify_ids])

    def source_counts(self):
        """{source_id: {"total": songs synced, "analyzed": songs with a preview}}."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT s.source_id, COUNT(*) AS total, COALESCE(SUM(t.preview_found), 0) AS analyzed "
                "FROM track_sources s LEFT JOIN tracks t ON t.spotify_id = s.spotify_id GROUP BY s.source_id").fetchall()
        return {r["source_id"]: {"total": r["total"], "analyzed": r["analyzed"]} for r in rows}

    def get_playlist(self, key):
        """The Spotify playlist the app saves to for this mood + genre key."""
        with self._connect() as conn:
            r = conn.execute("SELECT * FROM playlists WHERE key = ?", (key,)).fetchone()
        return dict(r) if r else None

    def set_playlist(self, key, playlist_id, name, url):
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO playlists (key, playlist_id, name, url, updated_at) VALUES (?, ?, ?, ?, datetime('now')) "
                "ON CONFLICT(key) DO UPDATE SET playlist_id=excluded.playlist_id, name=excluded.name, "
                "url=excluded.url, updated_at=excluded.updated_at",
                (key, playlist_id, name, url))

    def counts(self, sources=None):
        cond, params = self._source_filter(sources)
        with self._connect() as conn:
            total, analyzed = conn.execute(
                f"SELECT COUNT(*), COALESCE(SUM(preview_found), 0) FROM tracks WHERE {cond}", params).fetchone()
        return {"total": total, "analyzed": analyzed, "no_preview": total - analyzed}
