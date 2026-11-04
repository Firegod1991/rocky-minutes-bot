"""Feeds the knowledge base from Discord: channel history, new messages, Google files."""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Optional

import discord

from .drive import folder_links
from .knowledge import KIND_LABEL, KnowledgeBase, fetch_google_file, google_links, title_from_text

if TYPE_CHECKING:
    from .config import Config

log = logging.getLogger(__name__)


def message_key(m: discord.Message) -> str:
    return f"msg:{m.id}"


def _channel_label(ch) -> str:
    parent = getattr(ch, "parent", None)
    name = getattr(ch, "name", "dm")
    return f"{parent.name} > {name}" if parent is not None and isinstance(ch, discord.Thread) else name


class Indexer:
    def __init__(self, cfg: "Config", kb: KnowledgeBase):
        self.cfg = cfg
        self.kb = kb
        self.queue: asyncio.Queue = asyncio.Queue()
        self._http = None
        self._tasks: list[asyncio.Task] = []
        self.backfill_done = asyncio.Event()
        self.drive = None  # DriveIndexer, set by the bot when Drive is configured
        self.on_new_folder = None  # async callback(folder dict, message) for live folder posts

    # lifecycle ------------------------------------------------------------
    def start(self) -> None:
        import aiohttp

        self._http = aiohttp.ClientSession(headers={"User-Agent": "Mozilla/5.0 (RockyMinutesBot)"})
        self._tasks.append(asyncio.create_task(self._file_worker()))
        self._tasks.append(asyncio.create_task(self._refresh_loop()))

    async def close(self) -> None:
        for t in self._tasks:
            t.cancel()
        if self._http:
            await self._http.close()

    # messages -------------------------------------------------------------
    def index_message(self, m: discord.Message, minutes_channel_id: Optional[int] = None) -> None:
        if m.guild is None:
            return
        is_minutes = minutes_channel_id is not None and m.channel.id == minutes_channel_id
        if m.author.bot and not is_minutes:
            return
        # questions addressed to a bot ("@Rocky ...") are not knowledge
        lead = (m.content or "").lstrip()
        if lead.startswith("<@") and (any(u.bot for u in m.mentions) or any(r.managed for r in m.role_mentions)):
            self.kb.delete(message_key(m))
            return
        parts = [m.content or ""]
        for a in m.attachments:
            parts.append(f"[attachment: {a.filename}] {a.url}")
        for e in m.embeds:
            parts.extend(x for x in (e.title, e.description, e.url) if x)
        body = "\n".join(p for p in parts if p).strip()
        if not body:
            return
        kind = "minutes" if (is_minutes and m.author.bot) else "msg"
        label = _channel_label(m.channel)
        author = getattr(m.author, "display_name", None) or m.author.name
        self.kb.upsert(message_key(m), kind, m.guild.id, label, author, m.jump_url,
                       (m.content or "")[:80], body, m.created_at.timestamp())
        if self.drive is not None:
            for folder_id, url in folder_links(body):
                if self.drive.add_folder(folder_id, m.guild.id, url, author):
                    folder = next(f for f in self.drive.folders(m.guild.id) if f["folder_id"] == folder_id)
                    cb = self.on_new_folder
                    asyncio.get_running_loop().create_task(cb(folder, m) if cb else self.drive.crawl(folder))
        for gkind, fid, url in google_links(body):
            self.kb.note_file(gkind, fid, url, m.guild.id, label, author, m.content or "", m.created_at.timestamp())
            self.queue.put_nowait(fid)
        for a in m.attachments:
            if a.filename.lower().endswith((".txt", ".md", ".csv")) and a.size < 2_000_000:
                self.queue.put_nowait(("attachment", a.url, a.filename, m))

    async def backfill(self, guild: discord.Guild, minutes_channel_id: Optional[int]) -> None:
        limit = self.cfg.index_history_limit
        channels = [c for c in guild.text_channels] + [c for c in guild.voice_channels]
        total = 0
        for ch in channels:
            perms = ch.permissions_for(guild.me)
            if not (perms.view_channel and perms.read_message_history):
                continue
            n = 0
            try:
                # some channels make Discord's history call hang; never let one block the rest
                async with asyncio.timeout(60):
                    async for m in ch.history(limit=limit, oldest_first=False):
                        self.index_message(m, minutes_channel_id)
                        n += 1
                    # active threads under text channels
                    for th in getattr(ch, "threads", []):
                        async for m in th.history(limit=limit):
                            self.index_message(m, minutes_channel_id)
                            n += 1
            except discord.Forbidden:
                continue
            except TimeoutError:
                log.warning("backfill timed out for #%s after %d messages; skipping", ch.name, n)
            except Exception:
                log.warning("backfill failed for #%s", ch.name, exc_info=True)
            total += n
            await asyncio.sleep(0)
        log.info("indexed %d messages from %s (%s)", total, guild.name, self.kb.count())
        self.backfill_done.set()

    # meetings -------------------------------------------------------------
    def index_meeting(self, guild_id: int, sdir_name: str, channel_name: str, title: str,
                      transcript: str, minutes_md: str, url: str, ts: float) -> None:
        if transcript.strip():
            self.kb.upsert(f"meeting:{sdir_name}", "meeting", guild_id, channel_name, "meeting transcript",
                           url, title, transcript, ts)
        if minutes_md.strip():
            self.kb.upsert(f"minutes:{sdir_name}", "minutes", guild_id, channel_name, "meeting minutes",
                           url, title, minutes_md, ts)

    # google files ---------------------------------------------------------
    async def _file_worker(self) -> None:
        while True:
            item = await self.queue.get()
            try:
                if isinstance(item, tuple) and item[0] == "attachment":
                    await self._fetch_attachment(*item[1:])
                else:
                    await self._fetch_one(item)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("file worker error")
            await asyncio.sleep(0.5)  # be polite to Google

    async def _fetch_one(self, file_id: str, force: bool = False) -> None:
        rows = [r for r in self.kb.files_due(0 if force else self.cfg.doc_refresh_hours * 3600) if r["file_id"] == file_id]
        if not rows:
            return
        f = rows[0]
        status, text = await fetch_google_file(self._http, f["kind"], file_id)
        label = KIND_LABEL.get(f["kind"], "Google file")
        if status == "ok":
            title = title_from_text(f["kind"], text, f["context"])
            self.kb.upsert(f"gdoc:{file_id}", "gdoc", f["guild_id"], f["channel"], f["author"], f["url"],
                           f"{label}: {title}", text, f["ts"])
            self.kb.mark_file(file_id, "ok", title)
            log.info("indexed %s %r (%d chars)", label, title, len(text))
        else:
            # still searchable by the words around the link
            title = title_from_text(f["kind"], "", f["context"])
            body = f"{f['context']}\n(contents not readable by the bot: file is {status}; share it as 'anyone with the link can view')"
            self.kb.upsert(f"gdoc:{file_id}", "gdoc", f["guild_id"], f["channel"], f["author"], f["url"],
                           f"{label}: {title}", body, f["ts"])
            self.kb.mark_file(file_id, status, title)
            log.info("%s %s is %s", label, file_id, status)

    async def _fetch_attachment(self, url: str, filename: str, m: discord.Message) -> None:
        async with self._http.get(url) as r:
            if r.status != 200:
                return
            text = (await r.read())[:2_000_000].decode("utf-8", errors="replace")
        label = _channel_label(m.channel)
        self.kb.upsert(f"file:{m.id}:{filename}", "file", m.guild.id, label,
                       getattr(m.author, "display_name", m.author.name), m.jump_url, filename, text,
                       m.created_at.timestamp())

    async def _refresh_loop(self) -> None:
        while True:
            await asyncio.sleep(60)
            for f in self.kb.files_due(self.cfg.doc_refresh_hours * 3600):
                self.queue.put_nowait(f["file_id"])
            await asyncio.sleep(self.cfg.doc_refresh_hours * 3600)
