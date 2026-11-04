"""Live listening: cut each speaker's audio into utterances and transcribe them fast.

The recording sink calls feed() from the voice receive thread with 16 kHz mono
audio. A monitor thread closes an utterance after a short pause, and a worker
thread transcribes it with a small Whisper model. Results go back to the bot's
event loop through a callback, where wake word handling happens.
"""
from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

import numpy as np

log = logging.getLogger(__name__)

SR = 16_000
END_SILENCE_S = 0.8  # pause that closes an utterance
MAX_UTTERANCE_S = 15.0
MIN_UTTERANCE_S = 0.35
HALLUCINATIONS = {
    "thank you.", "thanks for watching!", "thank you for watching.", "you", ".", "bye.", "",
    "thanks for watching.", "thank you so much.",
}

_live_model = None
_live_lock = threading.Lock()


def get_live_model(name: str, threads: int = 4):
    global _live_model
    with _live_lock:
        if _live_model is None:
            from faster_whisper import WhisperModel

            log.info("loading live whisper model=%s", name)
            _live_model = WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=threads)
        return _live_model


@dataclass
class _Buf:
    name: str
    start: float  # seconds since session start
    last: float  # monotonic time of last frame
    parts: list = field(default_factory=list)
    samples: int = 0


@dataclass
class Utterance:
    user_id: int
    name: str
    start: float
    audio: np.ndarray


OnText = Callable[[int, str, float, str], Awaitable[None]]


class LiveListener:
    def __init__(self, loop: asyncio.AbstractEventLoop, on_text: OnText, model_name: str,
                 prompt: str = "Glossary: Rocky.", transcriber: Optional[Callable[[np.ndarray], str]] = None):
        self.loop = loop
        self.on_text = on_text
        self.model_name = model_name
        self.prompt = prompt
        self._transcriber = transcriber
        self.t0 = time.monotonic()
        self.bufs: dict[int, _Buf] = {}
        self.lock = threading.Lock()
        self.q: "queue.Queue[Optional[Utterance]]" = queue.Queue()
        self.stop_ev = threading.Event()
        self.lines: list[tuple[float, str, str]] = []  # (t, speaker, text) for "meeting so far"
        self.stats = {"frames": 0, "utterances": 0, "transcribed": 0, "errors": 0}
        self._threads = [
            threading.Thread(target=self._monitor, name="live-monitor", daemon=True),
            threading.Thread(target=self._worker, name="live-worker", daemon=True),
        ]
        for t in self._threads:
            t.start()

    # called from the voice thread -----------------------------------------
    def feed(self, user_id: int, name: str, mono16k: np.ndarray) -> None:
        if self.stop_ev.is_set() or mono16k.size == 0:
            return
        now = time.monotonic()
        self.stats["frames"] += 1
        with self.lock:
            b = self.bufs.get(user_id)
            if b is None:
                b = self.bufs[user_id] = _Buf(name=name, start=now - self.t0, last=now)
            b.parts.append(mono16k)
            b.samples += mono16k.size
            b.last = now
            if b.samples >= MAX_UTTERANCE_S * SR:
                self._close(user_id)

    def _close(self, user_id: int) -> None:
        b = self.bufs.pop(user_id, None)
        if b is None or b.samples < MIN_UTTERANCE_S * SR:
            return
        audio = np.concatenate(b.parts).astype(np.float32) / 32768.0
        self.stats["utterances"] += 1
        if self.q.qsize() > 30:  # badly behind: drop oldest so commands stay responsive
            try:
                self.q.get_nowait()
            except queue.Empty:
                pass
        self.q.put(Utterance(user_id, b.name, b.start, audio))

    # threads --------------------------------------------------------------
    def _monitor(self) -> None:
        while not self.stop_ev.wait(0.15):
            now = time.monotonic()
            with self.lock:
                for uid in [u for u, b in self.bufs.items() if now - b.last >= END_SILENCE_S]:
                    self._close(uid)

    def _transcribe(self, audio: np.ndarray) -> str:
        if self._transcriber:
            return self._transcriber(audio)
        model = get_live_model(self.model_name)
        segs, _ = model.transcribe(
            audio, language="en", beam_size=1, best_of=1, temperature=0.0,
            condition_on_previous_text=False, without_timestamps=True,
            initial_prompt=self.prompt, vad_filter=False,
        )
        return " ".join(s.text.strip() for s in segs if s.no_speech_prob < 0.7).strip()

    def _worker(self) -> None:
        while True:
            u = self.q.get()
            if u is None:
                return
            try:
                t = time.monotonic()
                text = self._transcribe(u.audio)
                self.stats["transcribed"] += 1
                if text.lower().strip() in HALLUCINATIONS:
                    continue
                self.lines.append((u.start, u.name, text))
                log.info("heard %s (%.1fs audio, %.1fs): %s", u.name, u.audio.size / SR, time.monotonic() - t, text[:120])
                asyncio.run_coroutine_threadsafe(self.on_text(u.user_id, u.name, u.start, text), self.loop)
            except Exception:
                self.stats["errors"] += 1
                log.exception("live transcription failed")

    def close(self) -> None:
        self.stop_ev.set()
        with self.lock:
            for uid in list(self.bufs):
                self._close(uid)
        self.q.put(None)

    def transcript_so_far(self, max_lines: int = 200) -> str:
        from .merge import fmt_ts

        return "\n".join(f"[{fmt_ts(t)}] {who}: {txt}" for t, who, txt in sorted(self.lines)[-max_lines:])
