"""Feed synthetic frames through TimelineSink and check timeline alignment."""
import json
import time
import wave
from types import SimpleNamespace

import numpy as np

from bot.recorder import OUT_RATE, SRC_RATE, TimelineSink, downmix

FRAME = 960  # 20 ms at 48 kHz


def tone_frame(freq=440.0, amp=8000):
    t = np.arange(FRAME) / SRC_RATE
    s = (np.sin(2 * np.pi * freq * t) * amp).astype(np.int16)
    return np.repeat(s, 2).tobytes()  # stereo interleaved


def pkt(ts):
    return SimpleNamespace(timestamp=ts)


def data(ts, pcm):
    return SimpleNamespace(packet=pkt(ts), pcm=pcm, source=None)


def test_downmix_shapes():
    out = downmix(tone_frame())
    assert out.dtype == np.int16 and out.size == FRAME // 3


def test_rtp_gap_is_padded(tmp_path):
    sink = TimelineSink(tmp_path)
    user = SimpleNamespace(id=1, display_name="fire")
    ts = 1000
    # 10 frames of speech, 1 s of silence (50 frames worth of RTP clock), 10 more frames
    for _ in range(10):
        sink.write(user, data(ts, tone_frame()))
        ts += FRAME
    ts += 50 * FRAME
    for _ in range(10):
        sink.write(user, data(ts, tone_frame()))
        ts += FRAME
    sink.cleanup()
    tr = sink.tracks[1]
    with wave.open(str(tr.path)) as w:
        assert w.getframerate() == OUT_RATE and w.getnchannels() == 1
        n = w.getnframes()
    # 20 frames of speech + 50 frames of padding = 70 frames * 320 samples, plus the tail pad
    assert n >= 70 * 320
    assert tr.speech_samples == 20 * 320
    manifest = json.loads((tmp_path / "tracks.json").read_text())
    assert manifest["tracks"][0]["name"] == "fire"


def test_second_speaker_offset_by_wall_clock(tmp_path):
    sink = TimelineSink(tmp_path)
    a = SimpleNamespace(id=1, display_name="a")
    b = SimpleNamespace(id=2, display_name="b")
    sink.write(a, data(0, tone_frame()))
    time.sleep(0.3)
    sink.write(b, data(0, tone_frame()))
    sink.cleanup()
    ta, tb = sink.tracks[1], sink.tracks[2]
    # b's speech starts ~0.3 s in; a's at ~0
    assert tb.first_seen >= 0.25
    assert ta.written == tb.written  # both padded to the same end


def test_unknown_user_ignored(tmp_path):
    sink = TimelineSink(tmp_path)
    sink.write(None, data(0, tone_frame()))
    sink.cleanup()
    assert sink.tracks == {}
