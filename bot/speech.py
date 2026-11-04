"""Rocky's speaking voice: local Piper text to speech, played into the voice channel."""
from __future__ import annotations

import asyncio
import io
import logging
import re
import threading
import wave
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

_voice = None
_voice_lock = threading.Lock()


def voice_path(voices_dir: Path, name: str) -> Path:
    return voices_dir / f"{name}.onnx"


def get_voice(voices_dir: Path, name: str):
    global _voice
    with _voice_lock:
        if _voice is None:
            from piper import PiperVoice

            path = voice_path(voices_dir, name)
            if not path.exists():
                from piper.download_voices import download_voice

                voices_dir.mkdir(parents=True, exist_ok=True)
                download_voice(name, voices_dir)
            log.info("loading piper voice %s", name)
            _voice = PiperVoice.load(str(path))
        return _voice


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


def synthesize_pcm48(text: str, voices_dir: Path, name: str, length_scale: float = 1.0) -> bytes:
    """Text -> 48 kHz stereo 16 bit PCM, the format Discord playback wants."""
    from piper.config import SynthesisConfig

    voice = get_voice(voices_dir, name)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        voice.synthesize_wav(text, w, syn_config=SynthesisConfig(length_scale=length_scale))
    buf.seek(0)
    with wave.open(buf, "rb") as w:
        sr, ch = w.getframerate(), w.getnchannels()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if ch > 1:
        pcm = pcm.reshape(-1, ch).mean(axis=1).astype(np.int16)
    n = int(len(pcm) * 48000 / sr)
    x = np.interp(np.linspace(0, len(pcm), n, endpoint=False), np.arange(len(pcm)), pcm.astype(np.float32))
    # short fade in/out avoids clicks; 150 ms of lead silence lets Discord's jitter buffer settle
    fade = min(480, n // 4)
    if fade:
        x[:fade] *= np.linspace(0, 1, fade)
        x[-fade:] *= np.linspace(1, 0, fade)
    x = np.concatenate([np.zeros(7200, dtype=np.float32), x])
    stereo = np.repeat(np.clip(x, -32768, 32767).astype(np.int16), 2)
    return stereo.tobytes()


class Speaker:
    """One speaking queue per voice connection so answers never talk over each other."""

    def __init__(self, voices_dir: Path, voice_name: str, executor=None):
        self.voices_dir = voices_dir
        self.voice_name = voice_name
        self.executor = executor
        self._lock = asyncio.Lock()

    async def say(self, vc, text: str) -> Optional[float]:
        import discord

        text = speakable(text)
        if not text or vc is None or not vc.is_connected():
            return None
        loop = asyncio.get_running_loop()
        pcm = await loop.run_in_executor(self.executor, synthesize_pcm48, text, self.voices_dir, self.voice_name)
        async with self._lock:
            if not vc.is_connected():
                return None
            done = asyncio.Event()
            vc.play(discord.PCMAudio(io.BytesIO(pcm)), after=lambda e: loop.call_soon_threadsafe(done.set))
            await asyncio.wait_for(done.wait(), timeout=len(pcm) / 192000 + 15)
        return len(pcm) / 192000
