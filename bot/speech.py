"""Rocky's speaking voice: local text to speech streamed into the voice channel.

Engines: Kokoro (natural, default) or Piper (lighter). Answers are split into
sentences; the first sentence starts playing while the rest are still being
generated, so Rocky starts talking about a second after it has the answer.
"""
from __future__ import annotations

import asyncio
import io
import logging
import re
import threading
import wave
from pathlib import Path
from typing import Optional

import discord
import numpy as np

log = logging.getLogger(__name__)

FRAME = 3840  # 20 ms of 48 kHz stereo s16, what Discord playback reads per tick
KOKORO_FILES = {
    "kokoro-v1.0.int8.onnx": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx",
    "voices-v1.0.bin": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
}

_engines: dict = {}
_engine_lock = threading.Lock()


def speakable(text: str) -> str:
    """Strip markdown, links, citations and mentions so TTS reads it naturally."""
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text or "")  # [label](url) first, keep the label
    t = re.sub(r"<?https?://\S+>?", "", t)
    t = re.sub(r"\[(\d+)\]", "", t)  # [1] citations
    t = re.sub(r"<@[!&]?\d+>", "someone", t)
    t = re.sub(r"[*_`>#~|]", "", t)
    t = t.replace("•", ". ")
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\s+([,.!?;:])", r"\1", t)
    return t[:600]


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    out: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # glue very short fragments to the previous sentence so speech doesn't sound choppy
        if out and len(p) < 18:
            out[-1] = f"{out[-1]} {p}"
        else:
            out.append(p)
    return out


def _to_pcm48_stereo(samples: np.ndarray, sr: int) -> bytes:
    x = samples.astype(np.float32)
    if np.abs(x).max(initial=0) <= 1.5:
        x = x * 32767.0  # float [-1, 1] -> int16 scale
    n = int(len(x) * 48000 / sr)
    if n <= 0:
        return b""
    y = np.interp(np.linspace(0, len(x), n, endpoint=False), np.arange(len(x)), x)
    fade = min(480, n // 4)
    if fade:
        y[:fade] *= np.linspace(0, 1, fade)
        y[-fade:] *= np.linspace(1, 0, fade)
    return np.repeat(np.clip(y, -32768, 32767).astype(np.int16), 2).tobytes()


class _Kokoro:
    def __init__(self, models_dir: Path, voice: str, speed: float):
        from kokoro_onnx import Kokoro

        models_dir.mkdir(parents=True, exist_ok=True)
        for name, url in KOKORO_FILES.items():
            p = models_dir / name
            if not p.exists():
                import urllib.request

                log.info("downloading %s", name)
                urllib.request.urlretrieve(url, p)
        self.k = Kokoro(str(models_dir / "kokoro-v1.0.int8.onnx"), str(models_dir / "voices-v1.0.bin"))
        self.voice, self.speed = voice, speed

    def pcm(self, text: str) -> bytes:
        samples, sr = self.k.create(text, voice=self.voice, speed=self.speed, lang="en-us")
        return _to_pcm48_stereo(np.asarray(samples), sr)


class _Piper:
    def __init__(self, models_dir: Path, voice: str, speed: float):
        from piper import PiperVoice

        path = models_dir / f"{voice}.onnx"
        if not path.exists():
            from piper.download_voices import download_voice

            models_dir.mkdir(parents=True, exist_ok=True)
            download_voice(voice, models_dir)
        self.v = PiperVoice.load(str(path))
        self.length_scale = 1.0 / max(speed, 0.5)

    def pcm(self, text: str) -> bytes:
        from piper.config import SynthesisConfig

        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            self.v.synthesize_wav(text, w, syn_config=SynthesisConfig(length_scale=self.length_scale))
        buf.seek(0)
        with wave.open(buf, "rb") as w:
            sr = w.getframerate()
            a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
        return _to_pcm48_stereo(a, sr)


def get_engine(engine: str, models_dir: Path, voice: str, speed: float = 1.05):
    key = (engine, str(models_dir), voice, speed)
    with _engine_lock:
        if key not in _engines:
            log.info("loading %s voice %s", engine, voice)
            cls = _Kokoro if engine == "kokoro" else _Piper
            _engines[key] = cls(models_dir, voice, speed)
        return _engines[key]


def synthesize_pcm48(text: str, models_dir: Path, voice: str, engine: str = "kokoro", speed: float = 1.05) -> bytes:
    """Whole text -> 48 kHz stereo PCM (used for tests and warmup)."""
    return get_engine(engine, models_dir, voice, speed).pcm(text)


class StreamingPCM(discord.AudioSource):
    """discord AudioSource fed sentence by sentence from a worker thread."""

    def __init__(self, lead_ms: int = 120):
        self.buf = bytearray(b"\x00" * (FRAME * (lead_ms // 20)))
        self.cond = threading.Condition()
        self.done = False

    def feed(self, pcm: bytes) -> None:
        with self.cond:
            self.buf.extend(pcm)
            self.cond.notify_all()

    def finish(self) -> None:
        with self.cond:
            self.done = True
            self.cond.notify_all()

    # discord.AudioSource interface
    def read(self) -> bytes:
        with self.cond:
            while len(self.buf) < FRAME and not self.done:
                self.cond.wait(timeout=0.5)
            if len(self.buf) >= FRAME:
                out = bytes(self.buf[:FRAME])
                del self.buf[:FRAME]
                return out
            if self.buf:  # last partial frame, pad with silence
                out = bytes(self.buf) + b"\x00" * (FRAME - len(self.buf))
                self.buf.clear()
                return out
            return b""

    def is_opus(self) -> bool:
        return False

    def cleanup(self) -> None:
        pass


class Speaker:
    """One speaking queue per voice connection so answers never talk over each other."""

    def __init__(self, models_dir: Path, voice: str, engine: str = "kokoro", speed: float = 1.05):
        self.models_dir, self.voice, self.engine, self.speed = models_dir, voice, engine, speed
        self._lock = asyncio.Lock()

    def warm(self) -> None:
        """Load the voice and run it once so the first real reply has no cold start."""
        try:
            synthesize_pcm48("Ready.", self.models_dir, self.voice, self.engine, self.speed)
        except Exception:
            log.exception("voice warmup failed")

    async def say(self, vc, text: str) -> Optional[float]:
        parts = sentences(speakable(text))
        if not parts or vc is None or not vc.is_connected():
            return None
        loop = asyncio.get_running_loop()
        async with self._lock:
            if not vc.is_connected():
                return None
            src = StreamingPCM()
            eng = await loop.run_in_executor(None, get_engine, self.engine, self.models_dir, self.voice, self.speed)

            rest = parts[1:]

            def produce() -> None:
                try:
                    for s in rest:
                        src.feed(eng.pcm(s))
                except Exception:
                    log.exception("speech synthesis failed")
                finally:
                    src.finish()

            # first sentence before we start the player, the rest while it plays
            src.feed(await loop.run_in_executor(None, eng.pcm, parts[0]))
            threading.Thread(target=produce, name="tts", daemon=True).start()
            done = asyncio.Event()
            vc.play(src, after=lambda e: loop.call_soon_threadsafe(done.set))
            try:
                await asyncio.wait_for(done.wait(), timeout=120)
            except asyncio.TimeoutError:
                vc.stop()
        return None
