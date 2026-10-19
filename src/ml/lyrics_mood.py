"""Lyric moods without any paid API: a multilingual embedding of the lyrics, then small linear models trained
to reproduce Claude's mood labels (see scripts/train_lyrics_model.py and data/lyrics_labels.csv)."""
import json
import logging

import numpy as np
import onnxruntime as ort
import requests
from tokenizers import Tokenizer

from src.audio.embeddings import available_cpus
from src.config import LYRICS_EMBEDDER_DIR, LYRICS_MODEL_JSON

log = logging.getLogger(__name__)

LYRIC_MOODS = ("sad", "happy", "romantic", "angry", "party", "calm")
EMBEDDER_REPO = "Xenova/multilingual-e5-small"
EMBEDDER_FILES = {"model_quantized.onnx": "onnx/model_quantized.onnx", "tokenizer.json": "tokenizer.json"}
CHUNK_TOKENS = 512  # the model's limit; longer lyrics are split and averaged
MAX_CHUNKS = 3
PREFIX = "query: "  # e5 models expect this prefix


def download_embedder(folder=LYRICS_EMBEDDER_DIR, session=None):
    """Fetch the embedding model from Hugging Face once; later calls find it on disk."""
    folder.mkdir(parents=True, exist_ok=True)
    http = session or requests
    for name, remote in EMBEDDER_FILES.items():
        path = folder / name
        if path.exists():
            continue
        log.info("Downloading %s/%s (first use only)", EMBEDDER_REPO, remote)
        part = path.with_suffix(path.suffix + ".part")
        with http.get(f"https://huggingface.co/{EMBEDDER_REPO}/resolve/main/{remote}", stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(part, "wb") as f:
                for block in r.iter_content(1 << 20):
                    f.write(block)
        part.replace(path)
    return folder


class Embedder:
    """Lyrics -> one 384-number vector (mean of token vectors, averaged over chunks, unit length)."""

    def __init__(self, folder=LYRICS_EMBEDDER_DIR):
        folder = download_embedder(folder)
        self.tok = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self.tok.no_truncation()
        self.tok.no_padding()
        self.bos, self.eos = self.tok.token_to_id("<s>"), self.tok.token_to_id("</s>")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = available_cpus()
        opts.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(folder / "model_quantized.onnx"), sess_options=opts,
                                         providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}

    def __call__(self, text):
        ids = self.tok.encode(PREFIX + text, add_special_tokens=False).ids
        step = CHUNK_TOKENS - 2
        chunks = [ids[i:i + step] for i in range(0, max(len(ids), 1), step)][:MAX_CHUNKS]
        width = max(len(c) for c in chunks) + 2
        input_ids = np.ones((len(chunks), width), dtype=np.int64)  # 1 = <pad>
        mask = np.zeros((len(chunks), width), dtype=np.int64)
        for i, c in enumerate(chunks):
            row = [self.bos, *c, self.eos]
            input_ids[i, :len(row)] = row
            mask[i, :len(row)] = 1
        feed = {"input_ids": input_ids, "attention_mask": mask, "token_type_ids": np.zeros_like(input_ids)}
        hidden = self.sess.run(None, {k: v for k, v in feed.items() if k in self.inputs})[0]
        pooled = (hidden * mask[..., None]).sum(1) / mask.sum(1, keepdims=True)
        vec = pooled.mean(0)
        return vec / np.linalg.norm(vec)


class LyricsMood:
    """Scores lyrics 0..1 for each of LYRIC_MOODS."""

    def __init__(self, embedder, coef, intercept, version):
        self.embedder = embedder
        self.coef = np.asarray(coef, dtype=np.float32)  # (moods, 384)
        self.intercept = np.asarray(intercept, dtype=np.float32)
        self.version = version

    @classmethod
    def load(cls, model_json=LYRICS_MODEL_JSON, embedder_dir=LYRICS_EMBEDDER_DIR):
        meta = json.loads(model_json.read_text(encoding="utf-8"))
        if tuple(meta["moods"]) != LYRIC_MOODS:
            raise ValueError(f"{model_json} has moods {meta['moods']}, expected {LYRIC_MOODS}")
        return cls(Embedder(embedder_dir), meta["coef"], meta["intercept"], meta["version"])

    def score(self, lyrics):
        scores = np.clip(self.coef @ self.embedder(lyrics) + self.intercept, 0.0, 1.0)
        return {m: round(float(s), 4) for m, s in zip(LYRIC_MOODS, scores)}
