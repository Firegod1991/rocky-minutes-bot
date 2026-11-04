"""faster-whisper transcription of per speaker tracks.

Each track is first cut into speech chunks with Silero VAD, then every chunk is
transcribed on its own. That keeps timestamps exact (Whisper is free to merge
two utterances into one segment otherwise) which matters because the final
transcript interleaves speakers by time.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import numpy as np

from .config import Config
from .merge import Segment

log = logging.getLogger(__name__)

SR = 16_000
_model = None
_model_lock = threading.Lock()


def get_model(cfg: Config):
    global _model
    with _model_lock:
        if _model is None:
            from faster_whisper import WhisperModel

            kwargs = {}
            if cfg.whisper_threads > 0:
                kwargs["cpu_threads"] = cfg.whisper_threads
            log.info(
                "loading whisper model=%s device=%s compute=%s",
                cfg.whisper_model, cfg.whisper_device, cfg.whisper_compute_type,
            )
            _model = WhisperModel(
                cfg.whisper_model,
                device=cfg.whisper_device,
                compute_type=cfg.whisper_compute_type,
                **kwargs,
            )
        return _model


def load_wav_16k(path: Path) -> np.ndarray:
    """Read our own 16 kHz mono int16 WAVs as float32 without going through PyAV."""
    import wave

    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2, path
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def speech_chunks(audio: np.ndarray, max_len_s: float = 30.0, merge_gap_s: float = 0.6) -> list[tuple[int, int]]:
    """(start_sample, end_sample) spans of speech, merged across short pauses."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    opts = VadOptions(min_silence_duration_ms=500, speech_pad_ms=300, min_speech_duration_ms=250,
                      max_speech_duration_s=max_len_s)
    spans = [(int(s["start"]), int(s["end"])) for s in get_speech_timestamps(audio, opts, sampling_rate=SR)]
    merged: list[tuple[int, int]] = []
    for s, e in spans:
        if merged and s - merged[-1][1] <= merge_gap_s * SR and (e - merged[-1][0]) <= max_len_s * SR:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


def transcribe_track(cfg: Config, path: Path, speaker: str) -> list[Segment]:
    model = get_model(cfg)
    audio = load_wav_16k(path)
    out: list[Segment] = []
    chunks = speech_chunks(audio)
    for s, e in chunks:
        offset = s / SR
        segments, _info = model.transcribe(
            audio[s:e],
            language=cfg.whisper_language,
            beam_size=5,
            condition_on_previous_text=False,
            initial_prompt=cfg.whisper_prompt,
            vad_filter=False,
        )
        for seg in segments:
            text = seg.text.strip()
            if not text or seg.no_speech_prob > 0.8:
                continue
            out.append(Segment(offset + float(seg.start), offset + float(seg.end), speaker, text))
    log.info("%s: %d chunks -> %d segments", speaker, len(chunks), len(out))
    return out


def transcribe_session_dir(cfg: Config, session_dir: Path) -> list[Segment]:
    """Transcribe every track listed in tracks.json. Blocking; run in a thread."""
    manifest = json.loads((session_dir / "tracks.json").read_text())
    all_segments: list[Segment] = []
    for tr in manifest.get("tracks", []):
        if tr.get("speech_seconds", 0) < 0.5:
            continue
        path = session_dir / tr["file"]
        if not path.exists():
            continue
        try:
            all_segments.extend(transcribe_track(cfg, path, tr["name"]))
        except Exception:
            log.exception("transcription failed for %s", path)
    return all_segments
