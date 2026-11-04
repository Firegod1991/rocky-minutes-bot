"""Rocky's brain: wake word detection, command parsing, search and answers."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from . import llm
from .config import Config
from .knowledge import Embedder, Hit, KnowledgeBase, hybrid_search

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
_HELP = re.compile(r"^(?:help|what can you do|commands|what are your commands|voice commands)\b", re.I)
_DOCS = re.compile(
    r"^(?:(?:list|show|what are|pull up|give me|send)\s+)?(?:all\s+|the\s+|our\s+|me\s+)*(?:docs|documents|files|google docs|links)"
    r"(?:\s+(?:you have|we have|you know|you know about|you can see|list))?\??$"
    r"|^what\s+(?:docs|documents|files)\s+(?:do\s+(?:you|we)\s+have|can\s+you\s+see)\??$", re.I)


def parse_command(text: str) -> Command:
    t = (text or "").strip().strip(".!")
    t = re.sub(r"^/(?:ask|minutes)\s*", "", t)  # "@Rocky /ask docs" typed as plain text
    if not t:
        return Command("help")
    if _HELP.match(t):
        return Command("help")
    if _DOCS.match(t):
        return Command("docs")
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
    "`docs`: lists every Google Doc, Sheet and Slides file I can see\n"
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
            snippet = _clean(h.snippet)[:180]
            link = f" · [jump]({h.url})" if h.url else ""
            lines.append(f"• {snippet}\n  {meta}{link}")
        else:
            lines.append(f"• [**{_clean(title)}**]({h.url})\n  {meta}")
    return "\n".join(lines)


def _clean(s: str) -> str:
    """Neutralize mentions and squash whitespace so quoted text can't ping anyone."""
    s = re.sub(r"<@[!&]?\d+>", "@someone", s or "")
    return re.sub(r"\s+", " ", s).strip()


def format_docs(files: list[tuple[str, str, str]], folders: list[dict] | None = None,
                drive_files: list[tuple[str, str, str]] | None = None, share_with: str = "") -> str:
    folders, drive_files = folders or [], drive_files or []
    if not files and not folders and not drive_files:
        return ("I don't see any Google files yet. Post a Doc, Sheet, Slides or Drive folder link in any channel "
                "and I'll pick it up.")
    lines = []
    if folders:
        lines.append(f"**Drive folders ({len(folders)})**")
        for f in folders[:10]:
            state = {"ok": f"{f.get('files', 0)} files", "no_access": f"can't open it, share it with `{share_with}` as Viewer"
                     if share_with else "can't open it", "no_credentials": "Drive access isn't set up yet",
                     "new": "reading now"}.get(f.get("status"), f.get("status", ""))
            lines.append(f"• [{_clean(f.get('name') or 'Drive folder')}]({f['url']}): {state}")
    total = len(files) + len(drive_files)
    if total:
        lines.append(f"**Files I can search ({total})**")
        for title, url, status in (drive_files + files)[:30]:
            note = "" if status == "ok" else " (link only, not shared publicly)"
            lines.append(f"• [{_clean(title)}]({url}){note}")
        if total > 30:
            lines.append(f"…and {total - 30} more")
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


DOC_RULES = (
    "Sources may be spreadsheet rows written as 'Column: value' pairs, tagged with the tab name in brackets. "
    "Read the column names carefully (mass each vs total mass, price each vs total). When the question needs a "
    "total, difference or margin, compute it from the rows and show the numbers you used. Say how current a value "
    "is when the source gives a date or a status such as Picked, Leading, Open or TBD. "
)

ASK_SYSTEM = (
    "You are Rocky, the assistant inside a team's Discord server. Answer the question using only the sources "
    "and the live meeting transcript provided. " + DOC_RULES + "Be brief: under 120 words unless the question needs more. "
    "Cite sources inline as [1], [2] and end with the links you cited, one per line, wrapped in <> so Discord "
    "does not embed them. If the sources do not contain the answer, say so in one sentence and suggest where "
    "it might be. Use plain Discord markdown. Do not use hyphens or em dashes; use commas or new sentences."
)


VOICE_SYSTEM = (
    "You are Rocky, the team's assistant, answering OUT LOUD in a Discord voice call. Use only the sources and the live "
    "meeting transcript provided. " + DOC_RULES + "Reply in one to three short spoken sentences: plain conversational English, no lists, "
    "no links, no citations, no markdown, numbers written the way you'd say them. Personality: calm, precise, dry and "
    "deadpan, like a seasoned mission robot; a brief wry aside is fine now and then, but the answer always comes first. "
    "If the sources don't have the answer, say so in one sentence and say where it might be found."
)


def embedder_for(cfg: Config) -> Embedder | None:
    return Embedder(cfg.ollama_url, cfg.embed_model) if (cfg.ollama_url and cfg.embed_model) else None


def source_links(hits: list[Hit], limit: int = 3) -> str:
    out, seen = [], set()
    for h in hits:
        if len(out) >= limit or not h.url or h.url in seen:
            continue
        seen.add(h.url)
        label = _clean(h.title if h.is_file else (h.snippet or h.title))[:70] or "source"
        out.append(f"[{label}]({h.url})")
    return ("Sources: " + " · ".join(out)) if out else ""


async def answer_question(cfg: Config, kb: KnowledgeBase, guild_id: int, question: str, asker: str,
                          meeting_so_far: str = "", spoken: bool = False) -> str:
    text, _ = await answer_with_hits(cfg, kb, guild_id, question, asker, meeting_so_far, spoken)
    return text


async def answer_with_hits(cfg: Config, kb: KnowledgeBase, guild_id: int, question: str, asker: str,
                           meeting_so_far: str = "", spoken: bool = False) -> tuple[str, list[Hit]]:
    hits = await hybrid_search(kb, embedder_for(cfg), question, guild_id, limit=10)
    if not llm.enabled(cfg):
        if spoken:
            return ("I can't form answers without a language model, but I posted the closest matches in the chat.", hits)
        head = "I can't write answers yet (no language model configured), but these look relevant:\n"
        return (head + format_find(question, hits).split("\n", 1)[-1] if hits else format_find(question, hits)), hits
    local = llm.provider(cfg) == "ollama"
    content = (f"Question from {asker}: {question}\n\n<sources>\n"
               f"{sources_block(hits, max_chars=7_000 if local else 14_000) or '(none found)'}\n</sources>")
    if meeting_so_far:
        content += f"\n\n<live_meeting_transcript>\n{meeting_so_far[-(6000 if local else 12000):]}\n</live_meeting_transcript>"
    try:
        text = await llm.complete(cfg, VOICE_SYSTEM if spoken else ASK_SYSTEM, content, 300 if spoken else 600, fast=spoken)
        return (text or format_find(question, hits)), hits
    except Exception:
        log.exception("answer failed")
        if spoken:
            return "I hit an error working that out. The closest matches are in the chat.", hits
        return "I hit an error writing that answer. Closest matches:\n" + format_find(question, hits).split("\n", 1)[-1], hits
