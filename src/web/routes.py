from collections import Counter
from functools import wraps

from flask import Blueprint, current_app, jsonify, redirect, render_template, request, session, url_for
from spotipy.cache_handler import FlaskSessionCacheHandler
from spotipy.exceptions import SpotifyException

from src.library_sync import LibrarySync, SyncProgress, iter_liked_tracks, start_background_sync
from src.ml.analyzer import ModelsMissingError
from src.moods import FULL_RANGE, MOODS, build_targets
from src.selector import select_tracks
from src.spotify_client import create_playlist

bp = Blueprint("main", __name__)
TRACK_FIELDS = ("spotify_id", "name", "artists", "genre", "genre_conf", "danceability",
                "energy", "valence", "acousticness", "tempo", "score")


def _state():
    return current_app.extensions["playlist"]


def _auth():
    return _state().oauth_factory(FlaskSessionCacheHandler(session))


def _spotify():
    auth = _auth()
    if not auth.validate_token(auth.cache_handler.get_cached_token()):
        return None
    return _state().spotify_factory(auth)


def spotify_required(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        sp = _spotify()
        if sp is None:
            return jsonify(error="Please log in with Spotify first."), 401
        return view(sp, *args, **kwargs)
    return wrapper


@bp.get("/")
def index():
    if _spotify() is None:
        return render_template("index.html")
    store = _state().store
    genres = sorted({t["genre"] for t in store.analyzed_tracks() if t["genre"]})
    app_data = {"moods": {m: {f: list(r) for f, r in v.items()} for m, v in MOODS.items()},
                "full": {f: list(r) for f, r in FULL_RANGE.items()}}
    return render_template("dashboard.html", counts=store.counts(), moods=list(MOODS), genres=genres, app_data=app_data)


@bp.get("/login")
def login():
    return redirect(_auth().get_authorize_url())


@bp.get("/callback")
def callback():
    code = request.args.get("code")
    if not code:
        return render_template("index.html", error=request.args.get("error", "Login was cancelled.")), 400
    _auth().get_access_token(code)
    return redirect(url_for("main.index"))


@bp.get("/logout")
def logout():
    session.clear()
    return redirect(url_for("main.index"))


@bp.post("/sync")
@spotify_required
def sync(sp):
    state = _state()
    if state.progress.running:
        return jsonify({**state.progress.to_dict(), "error": "A sync is already running."}), 409
    try:
        analyzer = state.get_analyzer()
    except ModelsMissingError as exc:
        return jsonify(error=str(exc)), 503
    # Fetch inside the request: the Spotify token lives in the Flask session, which a background thread cannot read.
    try:
        tracks = list(iter_liked_tracks(sp))
    except SpotifyException as exc:
        return jsonify(error=f"Spotify error: {exc.msg}"), 502
    state.progress = SyncProgress()
    syncer = LibrarySync(state.store, analyzer, find_preview=state.find_preview, download=state.download)
    state.sync_thread = start_background_sync(tracks, syncer, state.progress)
    return jsonify(state.progress.to_dict()), 202


@bp.get("/sync/status")
def sync_status():
    state = _state()
    return jsonify(**state.progress.to_dict(), counts=state.store.counts())


@bp.get("/library")
def library():
    return jsonify(tracks=_state().store.analyzed_tracks())


@bp.post("/generate")
def generate():
    body = request.get_json(silent=True) or {}
    try:
        targets = build_targets(body.get("mood"), body.get("overrides"))
        size = max(1, min(int(body.get("size", 25)), 100))
    except (ValueError, TypeError) as exc:
        return jsonify(error=str(exc)), 400
    tracks = select_tracks(_state().store.analyzed_tracks(), targets, body.get("genres") or None, size)
    return jsonify(tracks=[{k: t.get(k) for k in TRACK_FIELDS} for t in tracks],
                   matched=len(tracks), requested=size)


@bp.post("/playlist/save")
@spotify_required
def save_playlist(sp):
    body = request.get_json(silent=True) or {}
    rows = _state().store.get_many(list(body.get("track_ids") or []))
    if not rows:
        return jsonify(error="No songs to save."), 400
    genres = [g for g, _ in Counter(r["genre"] for r in rows if r["genre"]).most_common(3)]
    artists = [a for a, _ in Counter(r["artists"].split(", ")[0] for r in rows).most_common(5)]
    text = _state().llm(body.get("mood"), genres, artists,
                        base_url=current_app.config["OLLAMA_URL"], model=current_app.config["OLLAMA_MODEL"])
    try:
        result = create_playlist(sp, text["name"], text["description"], [r["uri"] for r in rows])
    except SpotifyException as exc:
        return jsonify(error=f"Spotify error: {exc.msg}"), 502
    return jsonify(**result, **text)
