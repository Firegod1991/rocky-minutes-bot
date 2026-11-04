"""Searchable knowledge base: Discord messages, linked Google files, past meetings.

Everything lives in one SQLite FTS5 table. Long documents are split into
chunks so a search returns the relevant passage, not just the whole file.
Google Docs, Sheets and Slides are pulled through their public export URLs,
which works for any file shared as "anyone with the link can view".
"""
from __future__ import annotations

import asyncio
import logging
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

log = logging.getLogger(__name__)

GOOGLE_RE = re.compile(
    r"https?://docs\.google\.com/(?P<kind>document|spreadsheets|presentation)/(?:u/\d+/)?d/(?P<id>[A-Za-z0-9_-]{20,})[^\s<>)\]]*"
)
DRIVE_RE = re.compile(r"https?://drive\.google\.com/(?:file/d/|open\?id=)(?P<id>[A-Za-z0-9_-]{20,})[^\s<>)\]]*")
EXPORT_URL = {
    "document": "https://docs.google.com/document/d/{id}/export?format=txt",
    "spreadsheets": "https://docs.google.com/spreadsheets/d/{id}/export?format=csv",
    "presentation": "https://docs.google.com/presentation/d/{id}/export/txt",
}
KIND_LABEL = {"document": "Google Doc", "spreadsheets": "Google Sheet", "presentation": "Google Slides"}

CHUNK_CHARS = 1500
STOPWORDS = set(
    """a an the and or of to in on for at by with from about as is are was were be been it its this that these those
    i me my we our you your he she they them their what which who whom when where why how do does did can could
    would should will shall may might must please pull up bring show find open get grab look search lookup me
    doc docs document documents google file files sheet sheets slide slides deck link links talking regarding
    related re any some there here up""".split()
)


@dataclass
class Hit:
    key: str
    kind: str
    channel: str
    author: str
    url: str
    title: str
    snippet: str
    body: str
    ts: float
    score: float

    @property
    def is_file(self) -> bool:
        return self.kind in ("gdoc", "file")


def google_links(text: str) -> list[tuple[str, str, str]]:
    """(kind, file_id, url) for every Google Docs/Sheets/Slides link in text."""
    out, seen = [], set()
    for m in GOOGLE_RE.finditer(text or ""):
        if m.group("id") not in seen:
            seen.add(m.group("id"))
            out.append((m.group("kind"), m.group("id"), m.group(0)))
    return out


def chunks(text: str, size: int = CHUNK_CHARS) -> list[str]:
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    out, cur = [], ""
    for para in re.split(r"\n\s*\n|\n", text):
        para = para.strip()
        if not para:
            continue
        if len(cur) + len(para) + 1 > size and cur:
            out.append(cur)
            cur = ""
        while len(para) > size:
            out.append(para[:size])
            para = para[size:]
        cur = f"{cur}\n{para}" if cur else para
    if cur:
        out.append(cur)
    return out


def fts_query(text: str) -> str:
    """Turn free text into a forgiving FTS5 OR query of meaningful words."""
    words = re.findall(r"[A-Za-z0-9]+", (text or "").lower())
    terms = [w for w in words if w not in STOPWORDS and len(w) > 1]
    if not terms:
        terms = [w for w in words if len(w) > 2]
    terms = list(dict.fromkeys(terms))[:12]
    return " OR ".join(f'"{t}"*' for t in terms)


class KnowledgeBase:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS kb USING fts5(
                key UNINDEXED, kind UNINDEXED, guild_id UNINDEXED, channel, author,
                url UNINDEXED, title, body, ts UNINDEXED, tokenize='porter unicode61');
            CREATE TABLE IF NOT EXISTS keys(key TEXT PRIMARY KEY, rid INTEGER);
            CREATE TABLE IF NOT EXISTS files(
                file_id TEXT PRIMARY KEY, kind TEXT, url TEXT, guild_id INTEGER, channel TEXT,
                author TEXT, context TEXT, ts REAL, fetched REAL DEFAULT 0, status TEXT DEFAULT 'new', title TEXT);
            """
        )
        self.db.commit()

    # writes ---------------------------------------------------------------
    def _delete_prefix(self, prefix: str) -> None:
        rows = self.db.execute("SELECT key, rid FROM keys WHERE key = ? OR key LIKE ?", (prefix, prefix + "#%")).fetchall()
        for key, rid in rows:
            self.db.execute("DELETE FROM kb WHERE rowid = ?", (rid,))
            self.db.execute("DELETE FROM keys WHERE key = ?", (key,))

    def upsert(self, key: str, kind: str, guild_id: int, channel: str, author: str, url: str,
               title: str, body: str, ts: float) -> None:
        """Replace everything stored under key with fresh (chunked) content."""
        parts = chunks(body) or [""]
        with self._lock:
            self._delete_prefix(key)
            for i, part in enumerate(parts):
                k = key if len(parts) == 1 else f"{key}#{i}"
                cur = self.db.execute(
                    "INSERT INTO kb(key, kind, guild_id, channel, author, url, title, body, ts) VALUES (?,?,?,?,?,?,?,?,?)",
                    (k, kind, guild_id, channel, author, url, title, part, ts),
                )
                self.db.execute("INSERT OR REPLACE INTO keys(key, rid) VALUES (?, ?)", (k, cur.lastrowid))
            self.db.commit()

    def delete(self, key: str) -> None:
        with self._lock:
            self._delete_prefix(key)
            self.db.commit()

    def has(self, key: str) -> bool:
        with self._lock:
            return self.db.execute("SELECT 1 FROM keys WHERE key = ? OR key LIKE ? LIMIT 1", (key, key + "#%")).fetchone() is not None

    def count(self) -> dict[str, int]:
        with self._lock:
            return dict(self.db.execute("SELECT kind, count(*) FROM kb GROUP BY kind").fetchall())

    # google files ---------------------------------------------------------
    def note_file(self, kind: str, file_id: str, url: str, guild_id: int, channel: str, author: str,
                  context: str, ts: float) -> None:
        with self._lock:
            self.db.execute(
                """INSERT INTO files(file_id, kind, url, guild_id, channel, author, context, ts)
                   VALUES (?,?,?,?,?,?,?,?)
                   ON CONFLICT(file_id) DO UPDATE SET context = excluded.context WHERE length(excluded.context) > length(files.context)""",
                (file_id, kind, url, guild_id, channel, author, context[:500], ts),
            )
            self.db.commit()

    def files_due(self, refresh_seconds: float) -> list[dict]:
        cutoff = time.time() - refresh_seconds
        with self._lock:
            rows = self.db.execute(
                "SELECT file_id, kind, url, guild_id, channel, author, context, ts, status FROM files WHERE fetched < ?",
                (cutoff,),
            ).fetchall()
        cols = ["file_id", "kind", "url", "guild_id", "channel", "author", "context", "ts", "status"]
        return [dict(zip(cols, r)) for r in rows]

    def mark_file(self, file_id: str, status: str, title: str = "") -> None:
        with self._lock:
            self.db.execute("UPDATE files SET fetched = ?, status = ?, title = ? WHERE file_id = ?",
                            (time.time(), status, title, file_id))
            self.db.commit()

    # reads ----------------------------------------------------------------
    def search(self, text: str, guild_id: Optional[int] = None, kinds: Optional[Iterable[str]] = None,
               limit: int = 8) -> list[Hit]:
        q = fts_query(text)
        if not q:
            return []
        sql = (
            "SELECT key, kind, channel, author, url, title, snippet(kb, 7, '**', '**', ' … ', 28), body, ts, "
            "bm25(kb, 0, 0, 0, 1.0, 0.5, 0, 6.0, 1.0, 0) AS score FROM kb WHERE kb MATCH ?"
        )
        args: list = [q]
        if guild_id is not None:
            sql += " AND guild_id = ?"
            args.append(guild_id)
        kinds = list(kinds or [])
        if kinds:
            sql += f" AND kind IN ({','.join('?' * len(kinds))})"
            args += kinds
        sql += " ORDER BY score LIMIT ?"
        args.append(limit * 4)
        with self._lock:
            try:
                rows = self.db.execute(sql, args).fetchall()
            except sqlite3.OperationalError:
                log.exception("bad search query %r", q)
                return []
        hits, seen = [], set()
        for r in rows:
            h = Hit(key=r[0], kind=r[1], channel=r[2] or "", author=r[3] or "", url=r[4] or "", title=r[5] or "",
                    snippet=r[6] or "", body=r[7] or "", ts=float(r[8] or 0), score=float(r[9]))
            # files rank above chat chatter about them
            if h.is_file:
                h.score *= 1.6
            dedupe = h.url if h.is_file else h.key
            if dedupe in seen:
                continue
            seen.add(dedupe)
            hits.append(h)
        hits.sort(key=lambda h: h.score)  # bm25: more negative is better
        return hits[:limit]


async def fetch_google_file(session, kind: str, file_id: str, timeout: float = 30) -> tuple[str, str]:
    """Returns (status, text). status is ok | private | missing | error."""
    import aiohttp

    url = EXPORT_URL[kind].format(id=file_id)
    try:
        async with session.get(url, allow_redirects=True, timeout=aiohttp.ClientTimeout(total=timeout)) as r:
            final = str(r.url)
            if "accounts.google.com" in final or "ServiceLogin" in final or r.status in (401, 403):
                return "private", ""
            if r.status == 404:
                return "missing", ""
            if r.status != 200:
                return "error", ""
            ctype = r.headers.get("content-type", "")
            if "text/html" in ctype:
                return "private", ""
            raw = await r.read()
            return "ok", raw[:2_000_000].decode("utf-8", errors="replace").lstrip("﻿")
    except asyncio.TimeoutError:
        return "error", ""
    except Exception:
        log.warning("fetch failed for %s %s", kind, file_id, exc_info=True)
        return "error", ""


def title_from_text(kind: str, text: str, context: str) -> str:
    if kind == "spreadsheets":
        first = text.splitlines()[0] if text else ""
        if first:
            return f"Sheet ({first[:60]})"
    for line in (text or "").splitlines():
        line = line.strip(" \t​#*")
        if 3 <= len(line) <= 120:
            return line
    ctx = re.sub(r"https?://\S+", "", context or "").strip()
    return ctx[:80] or KIND_LABEL.get(kind, "Google file")
