"""Rocky's brain: wake word detection, command parsing, search and answers."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from .config import Config
from .knowledge import Hit, KnowledgeBase

log = logging.getLogger(__name__)

# Whisper's usual spellings of "Rocky"
WAKE_VARIANTS = {"rocky": r"r[oa]ck(?:y|ie|i|ey)|rocky's"}


def wake_pattern(wake_words: list[str]) -> re.Pattern:
    alts = [WAKE_VARIANTS.get(w, re.escape(w)) for w in wake_words]
    return re.compile(
        rf"^\W*(?:(?:hey|hi|ok|okay|yo|so|um|uh|and)\W+)?(?:{'|'.join(alts)})\b[\s,.:;!?-]*(?P<rest>.*)$",
        re.IGNORECASE | re.DOTALL,
    )


def parse_wake(text: str, pattern: re.Pattern) -> Optional[str]:
    """Return the command after the wake word, '' for a bare wake word, None if not addressed."""
    m = pattern.match(text or "")
    if not m:
        return None
    return m.group("rest").strip().rstrip(".!")


@dataclass
class Command:
    kind: str  # find | ask | action | stop | notes | help
    arg: str = ""


_FIND = re.compile(
    r"^(?:can you |could you |please |go ahead and )*"
    r"(?:pull(?:\s+up)?|bring\s+up|find|show(?:\s+me)?|open(?:\s+up)?|get(?:\s+me)?|grab|look\s+up|search(?:\s+for)?|"
    r"send(?:\s+me)?|post|share|link(?:\s+me)?|where(?:'s|\s+is)|give\s+me)\s+(?P<q>.+)$",
    re.IGNORECASE,
)
_ACTION = re.compile(
    r"^(?:please\s+)?(?:add\s+(?:an?\s+)?|log\s+(?:an?\s+)?|new\s+|note\s+(?:an?\s+)?)?(?:action\s+items?|to\s*do|task)\s*[:,.-]?\s*(?P<item>.+)$",
    re.IGNORECASE,
)
_STOP = re.compile(r"^(?:please\s+)?(?:stop|end|pause|quit)\s+(?:the\s+)?(?:recording|meeting|notes|listening|taking notes)\b", re.I)
_NOTES = re.compile(
    r"^(?:please\s+)?(?:take|post|send|publish|make|write|drop|give\s+us)\s+(?:the\s+|our\s+)?(?:notes|minutes|summary)\b", re.I
)
_HELP = re.compile(r"^(?:help|what can you do|commands|what are your commands)\b", re.I)


def parse_command(text: str) -> Command:
    t = (text or "").strip().strip(".!")
    if not t:
        return Command("help")
    if _HELP.match(t):
        return Command("help")
    if _STOP.match(t):
        return Command("stop")
    if _NOTES.match(t):
        return Command("notes")
    m = _ACTION.match(t)
    if m:
        return Command("action", m.group("item").strip())
    m = _FIND.match(t)
    if m:
        return Command("find", m.group("q").strip())
    return Command("ask", t)


HELP_TEXT = (
    "**Rocky commands** (say \"Rocky, ...\" in voice or @Rocky in text)\n"
    "`pull up <topic>`: finds the doc, sheet, slides or message and posts the link\n"
    "`<any question>`: answers from channel history, shared Google files and past meetings\n"
    "`action item: <who> <does what> <by when>`: logs it into this meeting's notes\n"
    "`take notes now`: posts notes for the meeting so far and keeps recording\n"
    "`stop recording`: stops and posts notes; Rocky stays out until the channel empties"
)


def _when(ts: float) -> str:
    if not ts:
        return ""
    return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone().strftime("%b %d")


def format_find(query: str, hits: list[Hit], limit: int = 3) -> str:
    if not hits:
        return (f"I couldn't find anything about **{query}**. If it's a Google file, post its link in any channel "
                "and share it as \"anyone with the link can view\" so I can read it.")
    lines = [f"Here's what I found for **{query}**:"]
    for h in hits[:limit]:
        where = f"#{h.channel}" if h.channel else ""
        meta = " · ".join(x for x in (where, h.author, _when(h.ts)) if x)
        title = h.title or h.snippet[:80] or "message"
        if h.kind in ("msg", "minutes", "meeting"):
            snippet = h.snippet.replace("\n", " ")[:180]
            lines.append(f"• {snippet}\n  {meta} · <{h.url}>" if h.url else f"• {snippet}\n  {meta}")
        else:
            lines.append(f"• **{title}**\n  <{h.url}>\n  {meta}")
    return "\n".join(lines)


def sources_block(hits: list[Hit], max_chars: int = 14_000) -> str:
    out, used = [], 0
    for i, h in enumerate(hits, 1):
        body = h.body.strip()
        piece = (f"[{i}] {h.kind} | #{h.channel} | {h.author} | {_when(h.ts)} | {h.title}\n"
                 f"link: {h.url}\n{body[:2500]}\n")
        if used + len(piece) > max_chars:
            break
        out.append(piece)
        used += len(piece)
    return "\n".join(out)


ASK_SYSTEM = (
    "You are Rocky, the assistant inside a team's Discord server. Answer the question using only the sources "
    "and the live meeting transcript provided. Be brief: under 120 words unless the question needs more. "
    "Cite sources inline as [1], [2] and end with the links you cited, one per line, wrapped in <> so Discord "
    "does not embed them. If the sources do not contain the answer, say so in one sentence and suggest where "
    "it might be. Use plain Discord markdown. Do not use hyphens or em dashes; use commas or new sentences."
)


async def answer_question(cfg: Config, kb: KnowledgeBase, guild_id: int, question: str, asker: str,
                          meeting_so_far: str = "") -> str:
    hits = kb.search(question, guild_id=guild_id, limit=8)
    if not cfg.anthropic_api_key:
        head = "I can't write answers yet (no ANTHROPIC_API_KEY set), but these look relevant:\n"
        return head + format_find(question, hits).split("\n", 1)[-1] if hits else format_find(question, hits)
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(api_key=cfg.anthropic_api_key, max_retries=2, timeout=60)
    content = f"Question from {asker}: {question}\n\n<sources>\n{sources_block(hits) or '(none found)'}\n</sources>"
    if meeting_so_far:
        content += f"\n\n<live_meeting_transcript>\n{meeting_so_far[-12000:]}\n</live_meeting_transcript>"
    try:
        resp = await client.messages.create(
            model=cfg.claude_model, max_tokens=600, system=ASK_SYSTEM,
            messages=[{"role": "user", "content": content}],
        )
        text = "".join(getattr(b, "text", "") for b in resp.content).strip()
        return text or format_find(question, hits)
    except Exception:
        log.exception("Claude answer failed")
        return "I hit an error writing that answer. Closest matches:\n" + format_find(question, hits).split("\n", 1)[-1]
