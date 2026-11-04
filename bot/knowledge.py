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
    "spreadsheets": "https://docs.google.com/spreadsheets/d/{id}/export?format=xlsx",
    "presentation": "https://docs.google.com/presentation/d/{id}/export/txt",
    "file": "https://drive.google.com/uc?export=download&id={id}",
}
KIND_LABEL = {"document": "Google Doc", "spreadsheets": "Google Sheet", "presentation": "Google Slides",
              "file": "Drive file"}

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
    for m in DRIVE_RE.finditer(text or ""):
        if m.group("id") not in seen:
            seen.add(m.group("id"))
            out.append(("file", m.group("id"), m.group(0)))
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
            CREATE TABLE IF NOT EXISTS vecs(rid INTEGER PRIMARY KEY, vec BLOB);
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
            self.db.execute("DELETE FROM vecs WHERE rid = ?", (rid,))

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

    def list_files(self, guild_id: int) -> list[tuple[str, str, str]]:
        """(title, url, status) for every Google file seen in this guild, newest first."""
        with self._lock:
            rows = self.db.execute(
                "SELECT kind, coalesce(nullif(title, ''), context), url, status FROM files WHERE guild_id = ? ORDER BY ts DESC",
                (guild_id,),
            ).fetchall()
        return [(f"{KIND_LABEL.get(k, 'File')}: {t or 'untitled'}", u, s) for k, t, u, s in rows]

    def list_drive_files(self, guild_id: int) -> list[tuple[str, str, str]]:
        with self._lock:
            rows = self.db.execute(
                "SELECT title, url, min(channel) FROM kb WHERE kind = 'gdoc' AND guild_id = ? AND url LIKE 'https://%' "
                "AND rowid IN (SELECT rid FROM keys WHERE key LIKE 'gdrive:%') GROUP BY url ORDER BY 3, 1",
                (guild_id,),
            ).fetchall()
        return [(t, u, "ok") for t, u, _ in rows]

    # reads ----------------------------------------------------------------
    def search(self, text: str, guild_id: Optional[int] = None, kinds: Optional[Iterable[str]] = None,
               limit: int = 8, one_per_file: bool = True) -> list[Hit]:
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
            dedupe = h.url if (h.is_file and one_per_file) else h.key
            if dedupe in seen:
                continue
            seen.add(dedupe)
            hits.append(h)
        hits.sort(key=lambda h: h.score)  # bm25: more negative is better
        return hits[:limit]


def _disposition_name(header: str) -> str:
    """File name from a Content-Disposition header (Google puts the real title there)."""
    from urllib.parse import unquote

    m = re.search(r"filename\*=UTF-8''([^;]+)", header or "", re.I)
    if m:
        return unquote(m.group(1)).strip('"')
    m = re.search(r'filename="?([^";]+)"?', header or "", re.I)
    return m.group(1) if m else ""


def file_to_text(name: str, ctype: str, data: bytes) -> str:
    """Text from a downloaded file: Google export, pdf, docx, xlsx, pptx or plain text."""
    from .drive import docx_text, pdf_text, xlsx_text

    low, ctype = (name or "").lower(), (ctype or "").lower()
    if data[:4] == b"%PDF" or low.endswith(".pdf"):
        return pdf_text(data)
    if data[:2] == b"PK":
        import io as _io
        import zipfile

        names = zipfile.ZipFile(_io.BytesIO(data)).namelist()
        if "word/document.xml" in names:
            return docx_text(data)
        if any(n.startswith("xl/") for n in names):
            return xlsx_text(data)
        if any(n.startswith("ppt/slides/") for n in names):
            return pptx_text(data)
        return ""
    if any(t in ctype for t in ("image/", "video/", "audio/", "octet-stream")) and not low.endswith((".txt", ".md", ".csv", ".json")):
        return ""
    return data.decode("utf-8", errors="replace").lstrip("\ufeff")


def pptx_text(data: bytes) -> str:
    import io as _io
    import zipfile

    out = []
    with zipfile.ZipFile(_io.BytesIO(data)) as z:
        slides = sorted((n for n in z.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", n)),
                        key=lambda n: int(re.findall(r"\d+", n)[-1]))
        for i, n in enumerate(slides, 1):
            words = re.findall(r"<a:t>([^<]*)</a:t>", z.read(n).decode("utf-8", "replace"))
            if words:
                out.append(f"Slide {i}: " + " ".join(words))
    return "\n".join(out)


async def fetch_google_file(session, kind: str, file_id: str, timeout: float = 60) -> tuple[str, str, str]:
    """Returns (status, text, real file title). status is ok | private | missing | error."""
    import aiohttp

    from .drive import _read_all

    url = EXPORT_URL.get(kind, EXPORT_URL["file"]).format(id=file_id)
    try:
        async with session.get(url, allow_redirects=True, headers={"User-Agent": "Mozilla/5.0 (RockyBot)"},
                               timeout=aiohttp.ClientTimeout(total=timeout)) as r:
            final = str(r.url)
            if "accounts.google.com" in final or "ServiceLogin" in final or r.status in (401, 403):
                return "private", "", ""
            if r.status == 404:
                return "missing", "", ""
            if r.status != 200:
                return "error", "", ""
            ctype = r.headers.get("content-type", "")
            name = _disposition_name(r.headers.get("content-disposition", ""))
            data = await _read_all(r)
        if "text/html" in ctype:
            # big Drive files show a "can't scan for viruses" page first; follow its confirm link once
            page = data.decode("utf-8", "replace")
            if kind == "file" and ("uc-download-link" in page or "confirm" in page) and "accounts.google.com" not in page:
                async with session.get(f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t",
                                       timeout=aiohttp.ClientTimeout(total=timeout)) as r2:
                    if r2.status != 200 or "text/html" in r2.headers.get("content-type", ""):
                        return "private", "", ""
                    ctype = r2.headers.get("content-type", "")
                    name = _disposition_name(r2.headers.get("content-disposition", "")) or name
                    data = await _read_all(r2)
            else:
                return "private", "", ""
        title = re.sub(r"\.(txt|xlsx|csv|docx|pdf|pptx|md)$", "", name, flags=re.I).strip()
        text = await asyncio.get_running_loop().run_in_executor(None, file_to_text, name, ctype, data)
        return "ok", text, title
    except asyncio.TimeoutError:
        return "error", "", ""
    except Exception:
        log.warning("fetch failed for %s %s", kind, file_id, exc_info=True)
        return "error", "", ""


def title_from_text(kind: str, text: str, context: str) -> str:
    for line in (text or "").splitlines():
        line = line.strip(" \t​#*")
        if 3 <= len(line) <= 120:
            return line
    ctx = re.sub(r"https?://\S+", "", context or "").strip()
    return ctx[:80] or KIND_LABEL.get(kind, "Google file")


# ---------------------------------------------------------------------------------------------
# Meaning based search: embeddings from a local Ollama model (nomic-embed-text), stored per chunk.

class Embedder:
    def __init__(self, url: str, model: str):
        self.url = (url or "").rstrip("/")
        self.model = model
        self.ok = bool(self.url and self.model)

    async def embed(self, texts: list[str], query: bool = False):
        import aiohttp
        import numpy as np

        if not self.ok or not texts:
            return None
        prefix = "search_query: " if query else "search_document: "
        body = {"model": self.model, "input": [prefix + t[:6000] for t in texts], "keep_alive": "30m"}
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as s:
                async with s.post(self.url + "/api/embed", json=body) as r:
                    if r.status != 200:
                        log.warning("embedding failed: HTTP %s %s", r.status, (await r.text())[:200])
                        return None
                    data = await r.json()
        except Exception as e:
            log.warning("embedding unavailable: %s", e)
            return None
        v = np.asarray(data.get("embeddings") or [], dtype=np.float32)
        if v.ndim != 2 or len(v) != len(texts):
            return None
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)


def _kb_pending(kb: KnowledgeBase, limit: int) -> list[tuple[int, str]]:
    with kb._lock:
        rows = kb.db.execute(
            "SELECT rowid, title, channel, body FROM kb WHERE rowid NOT IN (SELECT rid FROM vecs) LIMIT ?", (limit,)
        ).fetchall()
    return [(rid, f"{title or ''} | {channel or ''}\n{(body or '')[:4000]}") for rid, title, channel, body in rows]


async def embed_pending(kb: KnowledgeBase, embedder: Embedder, batch: int = 32) -> int:
    """Embed chunks that don't have a vector yet. Returns how many were done."""
    import numpy as np

    done = 0
    while True:
        rows = _kb_pending(kb, batch)
        if not rows:
            return done
        vecs = await embedder.embed([t for _, t in rows])
        if vecs is None:
            return done
        with kb._lock:
            kb.db.executemany("INSERT OR REPLACE INTO vecs(rid, vec) VALUES (?, ?)",
                              [(rid, np.asarray(v, dtype=np.float32).tobytes()) for (rid, _), v in zip(rows, vecs)])
            kb.db.commit()
        done += len(rows)


def vector_search(kb: KnowledgeBase, qvec, guild_id: Optional[int], kinds: Optional[Iterable[str]] = None,
                  limit: int = 20) -> list[Hit]:
    import numpy as np

    sql = ("SELECT v.rid, v.vec FROM vecs v JOIN kb ON kb.rowid = v.rid WHERE 1=1"
           + (" AND kb.guild_id = ?" if guild_id is not None else ""))
    args: list = [guild_id] if guild_id is not None else []
    kinds = list(kinds or [])
    if kinds:
        sql += f" AND kb.kind IN ({','.join('?' * len(kinds))})"
        args += kinds
    with kb._lock:
        rows = kb.db.execute(sql, args).fetchall()
    if not rows:
        return []
    mat = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float32).reshape(len(rows), -1)
    if mat.shape[1] != qvec.shape[-1]:
        return []
    sims = mat @ qvec.reshape(-1)
    order = np.argsort(-sims)[:limit]
    rids = [rows[i][0] for i in order]
    with kb._lock:
        got = {r[0]: r for r in kb.db.execute(
            f"SELECT rowid, key, kind, channel, author, url, title, body, ts FROM kb WHERE rowid IN ({','.join('?' * len(rids))})",
            rids).fetchall()}
    hits = []
    for i in order:
        r = got.get(rows[i][0])
        if r:
            body = r[7] or ""
            hits.append(Hit(key=r[1], kind=r[2], channel=r[3] or "", author=r[4] or "", url=r[5] or "", title=r[6] or "",
                            snippet=body[:200], body=body, ts=float(r[8] or 0), score=-float(sims[i])))
    return hits


async def hybrid_search(kb: KnowledgeBase, embedder: Optional[Embedder], text: str, guild_id: Optional[int],
                        kinds: Optional[Iterable[str]] = None, limit: int = 10) -> list[Hit]:
    """Keyword (FTS) and meaning (embedding) results fused by reciprocal rank; several chunks per file allowed."""
    fts = kb.search(text, guild_id=guild_id, kinds=kinds, limit=25, one_per_file=False)
    vec: list[Hit] = []
    if embedder is not None and embedder.ok:
        q = await embedder.embed([text], query=True)
        if q is not None:
            vec = vector_search(kb, q[0], guild_id, kinds, limit=25)
    scores: dict[str, float] = {}
    by_key: dict[str, Hit] = {}
    for ranked in (fts, vec):
        for rank, h in enumerate(ranked):
            scores[h.key] = scores.get(h.key, 0.0) + 1.0 / (60 + rank) * (1.3 if h.is_file else 1.0)
            by_key.setdefault(h.key, h)
    best = sorted(scores, key=lambda k: -scores[k])[:limit]
    return [by_key[k] for k in best]
