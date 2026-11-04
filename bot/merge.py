"""Pure functions: merge per speaker segments into a readable transcript."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(order=True)
class Segment:
    start: float
    end: float
    speaker: str
    text: str


def fmt_ts(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def coalesce(segments: list[Segment], max_gap: float = 1.5) -> list[Segment]:
    """Sort by time and join back to back segments from the same speaker."""
    out: list[Segment] = []
    for seg in sorted(segments, key=lambda s: (s.start, s.end)):
        text = seg.text.strip()
        if not text:
            continue
        if out and out[-1].speaker == seg.speaker and seg.start - out[-1].end <= max_gap:
            out[-1].text = f"{out[-1].text} {text}"
            out[-1].end = max(out[-1].end, seg.end)
        else:
            out.append(Segment(seg.start, seg.end, seg.speaker, text))
    return out


def render_transcript(segments: list[Segment]) -> str:
    lines = [f"[{fmt_ts(s.start)}] {s.speaker}: {s.text}" for s in coalesce(segments)]
    return "\n".join(lines)


def talk_time(segments: list[Segment]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for s in segments:
        totals[s.speaker] = totals.get(s.speaker, 0.0) + max(0.0, s.end - s.start)
    return dict(sorted(totals.items(), key=lambda kv: -kv[1]))


def chunk_text(text: str, limit: int = 1900) -> list[str]:
    """Split on line boundaries so each chunk fits a Discord message."""
    chunks: list[str] = []
    cur = ""
    for line in text.splitlines(keepends=True):
        while len(line) > limit:
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(line[:limit])
            line = line[limit:]
        if len(cur) + len(line) > limit:
            chunks.append(cur)
            cur = line
        else:
            cur += line
    if cur.strip():
        chunks.append(cur)
    return chunks or [""]
