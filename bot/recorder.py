"""Per speaker, timeline aligned recording.

Discord delivers one Opus stream per speaker. The library decodes each to
48 kHz stereo PCM and hands it to the sink one 20 ms frame at a time, but only
while that person is actually talking. To rebuild "who said what, when" across
speakers, every user's track is padded with silence so that sample position
== wall clock position since the recording started. Tracks are downmixed to
16 kHz mono on the fly, which is what Whisper wants and keeps disk use sane.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import wave
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
from discord.ext import voice_recv

log = logging.getLogger(__name__)

SRC_RATE = 48_000
SRC_CHANNELS = 2
OUT_RATE = 16_000
DECIM = SRC_RATE // OUT_RATE  # 3
RTP_WRAP = 2**32
# If the RTP clock jumps by more than this we stop trusting it and realign to
# the wall clock (new SSRC after a rejoin, decoder reset, etc).
MAX_TRUSTED_GAP_SRC = 30 * 60 * SRC_RATE


def downmix(pcm: bytes) -> np.ndarray:
    """48 kHz stereo int16 bytes -> 16 kHz mono int16 array."""
    if not pcm:
        return np.zeros(0, dtype=np.int16)
    a = np.frombuffer(pcm, dtype=np.int16)
    if a.size % SRC_CHANNELS:
        a = a[: a.size - a.size % SRC_CHANNELS]
    mono = a.reshape(-1, SRC_CHANNELS).mean(axis=1)
    n = mono.size - mono.size % DECIM
    if n == 0:
        return np.zeros(0, dtype=np.int16)
    out = mono[:n].reshape(-1, DECIM).mean(axis=1)
    return np.clip(out, -32768, 32767).astype(np.int16)


@dataclass
class Track:
    user_id: int
    name: str
    path: Path
    wav: wave.Wave_write
    written: int = 0  # samples at OUT_RATE
    last_ts: Optional[int] = None  # last RTP timestamp (48 kHz units)
    last_frame: int = 0  # samples (48 kHz) in the last frame
    speech_samples: int = 0
    first_seen: float = 0.0

    def pad_to(self, target: int) -> None:
        gap = target - self.written
        if gap <= 0:
            return
        # write in chunks to bound memory
        chunk = OUT_RATE * 10
        while gap > 0:
            n = min(gap, chunk)
            self.wav.writeframes(bytes(n * 2))
            self.written += n
            gap -= n

    def append(self, mono: np.ndarray) -> None:
        if mono.size == 0:
            return
        self.wav.writeframes(mono.tobytes())
        self.written += int(mono.size)
        self.speech_samples += int(mono.size)


class TimelineSink(voice_recv.AudioSink):
    """Writes one timeline aligned 16 kHz mono WAV per speaker."""

    def __init__(self, out_dir: Path):
        super().__init__()
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.tracks: dict[int, Track] = {}
        self.names: dict[int, str] = {}
        self._lock = threading.Lock()
        self._t0 = time.monotonic()
        self.started_at = datetime.now(timezone.utc)
        self.finished = False
        self.frames = 0

    # voice_recv API -----------------------------------------------------
    def wants_opus(self) -> bool:
        return False

    def write(self, user: Any, data: voice_recv.VoiceData) -> None:  # type: ignore[override]
        if self.finished:
            return
        src = user or getattr(data, "source", None)
        if src is None:
            return  # SSRC not mapped to a user yet; nothing sensible to file it under
        uid = int(src.id)
        pcm = data.pcm
        if not pcm:
            return
        now = time.monotonic() - self._t0
        packet = getattr(data, "packet", None)
        ts = getattr(packet, "timestamp", None)

        with self._lock:
            tr = self.tracks.get(uid)
            if tr is None:
                name = self._display_name(src)
                tr = self._open_track(uid, name)
                tr.first_seen = now
                tr.pad_to(int(now * OUT_RATE))
            else:
                self._fill_gap(tr, ts, now)

            mono = downmix(pcm)
            tr.append(mono)
            if ts is not None:
                tr.last_ts = int(ts)
                tr.last_frame = len(pcm) // (2 * SRC_CHANNELS)
            self.frames += 1

    def cleanup(self) -> None:
        with self._lock:
            if self.finished:
                return
            self.finished = True
            end = int((time.monotonic() - self._t0) * OUT_RATE)
            if self.tracks:
                end = max(end, max(tr.written for tr in self.tracks.values()))
            for tr in self.tracks.values():
                try:
                    tr.pad_to(end)
                    tr.wav.close()
                except Exception:
                    log.exception("closing track for %s", tr.user_id)
        self.write_manifest()

    # helpers -------------------------------------------------------------
    @staticmethod
    def _display_name(src: Any) -> str:
        for attr in ("display_name", "global_name", "name"):
            v = getattr(src, attr, None)
            if v:
                return str(v)
        return f"user-{getattr(src, 'id', '?')}"

    def _open_track(self, uid: int, name: str) -> Track:
        path = self.out_dir / f"{uid}.wav"
        w = wave.open(str(path), "wb")
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(OUT_RATE)
        tr = Track(user_id=uid, name=name, path=path, wav=w)
        self.tracks[uid] = tr
        self.names[uid] = name
        log.info("new speaker track: %s (%s)", name, uid)
        return tr

    def _fill_gap(self, tr: Track, ts: Optional[int], now: float) -> None:
        wall_target = int(now * OUT_RATE)
        if ts is None or tr.last_ts is None:
            # no RTP clock, trust the wall clock with a little jitter slack
            if wall_target - tr.written > OUT_RATE // 2:
                tr.pad_to(wall_target)
            return
        delta = (int(ts) - tr.last_ts) % RTP_WRAP
        gap_src = delta - tr.last_frame
        if 0 < gap_src <= MAX_TRUSTED_GAP_SRC:
            tr.pad_to(tr.written + gap_src // DECIM)
        elif gap_src > MAX_TRUSTED_GAP_SRC or delta == 0:
            # clock jumped or reset; realign to wall clock
            if wall_target > tr.written:
                tr.pad_to(wall_target)
        # drift guard: never let a track fall more than 2 s behind the wall clock
        if wall_target - tr.written > 2 * OUT_RATE:
            tr.pad_to(wall_target)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._t0

    def write_manifest(self) -> None:
        manifest = {
            "started_at": self.started_at.isoformat(),
            "duration_seconds": round(self.elapsed, 1),
            "frames": self.frames,
            "tracks": [
                {
                    "user_id": tr.user_id,
                    "name": tr.name,
                    "file": tr.path.name,
                    "speech_seconds": round(tr.speech_samples / OUT_RATE, 1),
                }
                for tr in self.tracks.values()
            ],
        }
        (self.out_dir / "tracks.json").write_text(json.dumps(manifest, indent=2))


@dataclass
class Session:
    guild_id: int
    channel_id: int
    channel_name: str
    guild_name: str
    dir: Path
    started_at: datetime
    sink: Optional[TimelineSink] = None
    ended_at: Optional[datetime] = None
    participants: dict[int, str] = field(default_factory=dict)
    leave_task: Any = None
    manual: bool = False

    @property
    def duration(self) -> float:
        end = self.ended_at or datetime.now(timezone.utc)
        return (end - self.started_at).total_seconds()

    def meta(self) -> dict:
        return {
            "guild_id": self.guild_id,
            "guild_name": self.guild_name,
            "channel_id": self.channel_id,
            "channel_name": self.channel_name,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration_seconds": round(self.duration, 1),
            "participants": {str(k): v for k, v in self.participants.items()},
            "manual": self.manual,
        }

    def save_meta(self, status: str) -> None:
        m = self.meta()
        m["status"] = status
        (self.dir / "meta.json").write_text(json.dumps(m, indent=2))
