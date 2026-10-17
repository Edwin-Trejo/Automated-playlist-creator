from functools import wraps

from flask import Blueprint, current_app, jsonify, redirect, render_template, request, session, url_for
from spotipy.cache_handler import FlaskSessionCacheHandler
from spotipy.exceptions import SpotifyException

from src.library_sync import LibrarySync, SyncProgress, iter_liked_tracks, start_background_sync
from src.ml.analyzer import ModelsMissingError
from src.moods import FULL_RANGE, MOODS, build_targets
from src.playlists import playlist_description, playlist_key, playlist_name
from src.selector import select_tracks
from src.spotify_client import (add_items, create_playlist, current_user_id, find_playlist, playlist_track_ids,
                                unique_name, user_playlists)

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
    app_data = {"features": list(FULL_RANGE)}
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


def _target_playlist(sp, mood, genres, playlists=None):
    """The existing Spotify playlist this mood + genre filter saves to, or None if there isn't one yet."""
    saved = _state().store.get_playlist(playlist_key(mood, genres)) or {}
    playlists = user_playlists(sp) if playlists is None else playlists
    return find_playlist(playlists, current_user_id(sp), saved.get("playlist_id"), playlist_name(mood, genres))


@bp.post("/generate")
def generate():
    body = request.get_json(silent=True) or {}
    mood, genres = body.get("mood"), body.get("genres") or []
    try:
        targets = build_targets(mood, body.get("adjust"))
        size = max(1, min(int(body.get("size", 25)), 100))
    except (ValueError, TypeError) as exc:
        return jsonify(error=str(exc)), 400
    tracks = _state().store.analyzed_tracks()
    target = {"name": playlist_name(mood, genres), "url": None, "exists": False, "already_in": 0}
    sp = None if body.get("new_playlist") else _spotify()
    if sp is not None:
        # Adding to the existing playlist: only suggest songs that aren't in it yet.
        try:
            existing = _target_playlist(sp, mood, genres)
            if existing:
                in_playlist = set(playlist_track_ids(sp, existing["id"]))
                tracks = [t for t in tracks if t["spotify_id"] not in in_playlist]
                target.update(name=existing["name"], url=existing["external_urls"]["spotify"], exists=True,
                              already_in=len(in_playlist))
        except SpotifyException as exc:
            return jsonify(error=f"Spotify error: {exc.msg}"), 502
    tracks = select_tracks(tracks, targets, genres or None, size)
    return jsonify(tracks=[{k: t.get(k) for k in TRACK_FIELDS} for t in tracks],
                   matched=len(tracks), requested=size, target=target)


@bp.post("/playlist/save")
@spotify_required
def save_playlist(sp):
    body = request.get_json(silent=True) or {}
    rows = _state().store.get_many(list(body.get("track_ids") or []))
    if not rows:
        return jsonify(error="No songs to save."), 400
    mood, genres = body.get("mood"), body.get("genres") or []
    name, description = playlist_name(mood, genres), playlist_description(mood, genres)
    uris = [r["uri"] for r in rows]
    try:
        playlists = user_playlists(sp)
        existing = None if body.get("new_playlist") else _target_playlist(sp, mood, genres, playlists)
        if existing:
            in_playlist = set(playlist_track_ids(sp, existing["id"]))
            new_uris = [r["uri"] for r in rows if r["spotify_id"] not in in_playlist]
            add_items(sp, existing["id"], new_uris)
            result = {"id": existing["id"], "url": existing["external_urls"]["spotify"], "name": existing["name"],
                      "created": False, "added": len(new_uris), "total": len(in_playlist) + len(new_uris)}
        else:
            name = unique_name(name, playlists) if body.get("new_playlist") else name
            result = {**create_playlist(sp, name, description, uris), "name": name,
                      "created": True, "added": len(uris), "total": len(uris)}
    except SpotifyException as exc:
        return jsonify(error=f"Spotify error: {exc.msg}"), 502
    # The newest playlist for this mood + genres is where the next "add" goes.
    _state().store.set_playlist(playlist_key(mood, genres), result["id"], result["name"], result["url"])
    return jsonify(**result, description=description)
