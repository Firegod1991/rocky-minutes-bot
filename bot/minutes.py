"""Turn a transcript into meeting minutes with Claude (falls back to a plain summary header)."""
from __future__ import annotations

import logging
from datetime import datetime

from .config import Config
from .merge import Segment, fmt_ts, talk_time

log = logging.getLogger(__name__)

SYSTEM = (
    "You write concise, accurate meeting minutes from a voice chat transcript. "
    "Speakers are labeled by their Discord username. Never invent facts, names, or "
    "decisions that are not in the transcript. If something is unclear, say so. "
    "Do not use hyphens or em dashes anywhere; use commas or separate sentences instead."
)

PROMPT = """Meeting: {title}
Date: {date}
Duration: {duration}
Participants: {participants}

Transcript (timestamps are HH:MM:SS from the start):
<transcript>
{transcript}
</transcript>

Write the minutes in Discord flavored Markdown with exactly these sections, in this order:

**Summary**
Three to six sentences on what the meeting was about and what was accomplished.

**Key Points**
Bulleted list of the main topics discussed. Attribute positions to speakers by username where it matters.

**Decisions**
Bulleted list of decisions actually made. Write "None recorded" if there were none.

**Action Items**
Bulleted list in the form `@username: task (deadline if stated)`. Write "None recorded" if there were none.

**Open Questions**
Bulleted list of unresolved questions or things to follow up on. Write "None" if there were none.

Keep the whole thing under 500 words. Plain Markdown only, no tables, no headers with #."""

MAX_TRANSCRIPT_CHARS = 150_000


def header(title: str, started: datetime, duration_s: float, participants: list[str]) -> str:
    return (
        f"**{title}**\n"
        f"Date: {started.astimezone().strftime('%Y-%m-%d %H:%M %Z')}\n"
        f"Duration: {fmt_ts(duration_s)}\n"
        f"Participants: {', '.join(participants) if participants else 'none detected'}\n"
    )


def talk_time_block(segments: list[Segment]) -> str:
    tt = talk_time(segments)
    if not tt:
        return ""
    lines = [f"{name}: {fmt_ts(sec)}" for name, sec in tt.items()]
    return "**Talk time**\n" + "\n".join(lines) + "\n"


async def generate_minutes(
    cfg: Config,
    title: str,
    started: datetime,
    duration_s: float,
    participants: list[str],
    transcript: str,
    segments: list[Segment],
    logged_items: list[dict] | None = None,
) -> tuple[str, str]:
    """Returns (minutes_markdown, mode) where mode is 'claude' or 'fallback'."""
    head = header(title, started, duration_s, participants)
    logged = logged_items_block(logged_items or [])
    if not transcript.strip():
        return head + "\nNo speech was detected in this meeting.\n" + (f"\n{logged}" if logged else ""), "fallback"

    body = ""
    mode = "fallback"
    if cfg.anthropic_api_key:
        try:
            extra = f"\n\nAction items logged by voice during the meeting (include them):\n{logged}" if logged else ""
            body = await _ask_claude(cfg, title, started, duration_s, participants, transcript + extra)
            mode = "claude"
        except Exception:
            log.exception("Claude summarization failed; posting transcript only")
    if not body:
        body = "Automatic summary unavailable. Full transcript attached.\n"
        if logged:
            body += f"\n{logged}"
    return f"{head}\n{body.strip()}\n\n{talk_time_block(segments)}", mode


def logged_items_block(items: list[dict]) -> str:
    if not items:
        return ""
    lines = [f"• {it.get('text', '').strip()} (logged by {it.get('by', 'someone')} at {fmt_ts(it.get('t', 0))})"
             for it in items if it.get("text")]
    return "**Action Items (logged by voice)**\n" + "\n".join(lines) + "\n"


async def _ask_claude(cfg, title, started, duration_s, participants, transcript) -> str:
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(api_key=cfg.anthropic_api_key, max_retries=3, timeout=180)
    if len(transcript) > MAX_TRANSCRIPT_CHARS:
        transcript = transcript[:MAX_TRANSCRIPT_CHARS] + "\n[transcript truncated]"
    prompt = PROMPT.format(
        title=title,
        date=started.astimezone().strftime("%Y-%m-%d %H:%M %Z"),
        duration=fmt_ts(duration_s),
        participants=", ".join(participants) or "unknown",
        transcript=transcript,
    )
    resp = await client.messages.create(
        model=cfg.claude_model,
        max_tokens=1500,
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(getattr(b, "text", "") for b in resp.content).strip()
