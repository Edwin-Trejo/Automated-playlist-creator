import json

from src.fallback import DatasetLookup, artist_estimate
from src.moods import FULL_RANGE
from src.selector import select_tracks

HEADER = ",track_id,artists,album_name,track_name,popularity,duration_ms,explicit,danceability,energy,key,loudness," \
         "mode,speechiness,acousticness,instrumentalness,liveness,valence,tempo,time_signature,track_genre\n"


def csv_row(i, tid, artists, name, genre, dance=0.5):
    return f"{i},{tid},{artists},Album,{name},50,200000,False,{dance},0.6,1,-5,1,0.05,0.2,0,0.1,0.7,120.5,4,{genre}\n"


def lookup(tmp_path):
    (tmp_path / "d.csv").write_text(HEADER + csv_row(0, "id1", "J Balvin;Bad Bunny", "Azul", "reggaeton", 0.9)
                                    + csv_row(1, "id1", "J Balvin;Bad Bunny", "Azul", "latin", 0.9)
                                    + csv_row(2, "id2", "Kesha", "Blow", "happy"), encoding="utf-8")
    (tmp_path / "g.json").write_text(json.dumps({"Reggaeton / Urbano": ["reggaeton"], "Latin": ["latin"]}))
    return DatasetLookup(tmp_path / "d.csv", tmp_path / "g.json")


def test_dataset_lookup_by_id_and_by_cleaned_title_and_artist(tmp_path):
    d = lookup(tmp_path)
    by_id = d.find({"spotify_id": "id1", "name": "x", "artists": "x"})
    assert by_id["danceability"] == 0.9 and by_id["tempo"] == 120.5 and by_id["genre"] in ("Reggaeton / Urbano", "Latin")
    by_name = d.find({"spotify_id": "other", "name": "Azul - Remastered", "artists": "J Balvin, Someone"})
    assert by_name["danceability"] == 0.9
    assert d.find({"spotify_id": "nope", "name": "Unknown", "artists": "Nobody"}) is None


def test_dataset_genre_outside_the_groups_is_none(tmp_path):
    assert lookup(tmp_path).find({"spotify_id": "id2", "name": "Blow", "artists": "Kesha"})["genre"] is None


def test_artist_estimate_ignores_other_artists_and_estimates():
    songs = [{"artists": "A, X", "genre": "Pop", "danceability": 0.2, "energy": 0.4, "valence": 0.6,
              "acousticness": 0.1, "tempo": 100.0},
             {"artists": "A", "genre": "Rock", "danceability": 0.4, "energy": 0.6, "valence": 0.2,
              "acousticness": 0.3, "tempo": 140.0},
             {"artists": "A", "genre": "Pop", "danceability": 0.6, "energy": 0.8, "valence": 0.4,
              "acousticness": 0.5, "tempo": 120.0},
             {"artists": "B", "genre": "Metal", "danceability": 1, "energy": 1, "valence": 1, "acousticness": 1, "tempo": 1},
             {"artists": "A", "genre": "Metal", "analysis_source": "artist", "danceability": 1, "energy": 1,
              "valence": 1, "acousticness": 1, "tempo": 1}]
    est = artist_estimate({"artists": "a, Guest"}, songs)
    assert est["genre"] == "Pop" and abs(est["danceability"] - 0.4) < 1e-9 and est["tempo"] == 120.0
    assert abs(est["genre_conf"] - 1 / 3) < 1e-3
    assert artist_estimate({"artists": "Nobody"}, songs) is None


def test_estimated_songs_rank_below_equally_good_analyzed_ones():
    base = {"genre": "Pop", "genre_conf": 0.5, "danceability": 0.85, "energy": 0.5, "valence": 0.5, "tempo": 120.0}
    tracks = [{**base, "spotify_id": "est", "name": "a", "analysis_source": "artist"},
              {**base, "spotify_id": "real", "name": "b", "analysis_source": "audio"}]
    out = select_tracks(tracks, {**FULL_RANGE, "danceability": (0.7, 1.0)})
    assert [t["spotify_id"] for t in out] == ["real", "est"]
    assert out[1]["score"] == 0.8
