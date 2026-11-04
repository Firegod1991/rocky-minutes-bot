"""Read-only Google Drive folder crawler.

Share a Drive folder with Rocky's service account (Viewer is enough) and post the
folder link anywhere in Discord. Rocky walks the folder and every subfolder and
indexes Google Docs, Sheets, Slides, PDFs, Word files and plain text files.

Access is read only twice over: the token is minted with the drive.readonly
scope, and the account only has whatever sharing you gave it.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import re
import sqlite3
import threading
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

FOLDER_RE = re.compile(r"https?://drive\.google\.com/drive/(?:u/\d+/)?folders/(?P<id>[A-Za-z0-9_-]{10,})[^\s<>)\]]*")
API = "https://www.googleapis.com/drive/v3"
SCOPE = "https://www.googleapis.com/auth/drive.readonly"
FOLDER_MIME = "application/vnd.google-apps.folder"
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
EXPORTS = {
    "application/vnd.google-apps.document": ("text/plain", "Google Doc"),
    "application/vnd.google-apps.spreadsheet": (XLSX, "Google Sheet"),
    "application/vnd.google-apps.presentation": ("text/plain", "Google Slides"),
}
DOWNLOADS = {
    "application/pdf": "PDF",
    "text/plain": "Text file",
    "text/markdown": "Markdown file",
    "text/csv": "CSV file",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "Word doc",
    XLSX: "Excel workbook",
}
MAX_BYTES = 25_000_000
MAX_FILES = 3000
MAX_DEPTH = 12


async def _read_all(r, limit: int = MAX_BYTES) -> bytes:
    """Read a whole response body; StreamReader.read(n) can return just one chunk."""
    buf = bytearray()
    async for chunk in r.content.iter_chunked(1 << 16):
        buf += chunk
        if len(buf) > limit:
            raise RuntimeError("file too large")
    return bytes(buf)


def folder_links(text: str) -> list[tuple[str, str]]:
    out, seen = [], set()
    for m in FOLDER_RE.finditer(text or ""):
        if m.group("id") not in seen:
            seen.add(m.group("id"))
            out.append((m.group("id"), m.group(0)))
    return out


def pdf_text(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    return "\n".join((p.extract_text() or "") for p in reader.pages[:400])


def _is_num(c: str) -> bool:
    return bool(re.fullmatch(r"[$\u20ac]?-?[\d,]*\.?\d+%?", c.replace(" ", "")))


def _is_header(r: list[str], nxt: list[str]) -> bool:
    """Header row: 3+ short text (non numeric) cells, and the next row fills 2+ of those columns."""
    filled = [c for c in r if c]
    if len(filled) < 3 or any(len(c) > 60 or _is_num(c) for c in filled):
        return False
    return sum(1 for j, c in enumerate(r) if c and j < len(nxt) and nxt[j]) >= 2


def xlsx_text(data: bytes, max_rows: int = 5000) -> str:
    """Every tab of a workbook as text. Table rows are written as 'Column: value' records so each
    line keeps its meaning on its own (which number is the mass, which is the price...).
    A blank row ends a table, so several tables stacked on one tab are each read correctly."""
    from openpyxl import load_workbook

    wb = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    parts = []
    for ws in wb.worksheets:
        rows: list[list[str]] = []
        for i, row in enumerate(ws.iter_rows(values_only=True)):
            if i >= max_rows:
                break
            cells = ["" if v is None else re.sub(r"\s+", " ", str(v)).strip() for v in row]
            while cells and not cells[-1]:
                cells.pop()
            rows.append(cells)
        out, seen, header = [], set(), None
        for i, r in enumerate(rows):
            if not any(r):
                header = None
                continue
            nxt = next((x for x in rows[i + 1:i + 2]), [])
            if header is None and _is_header(r, nxt):
                header = r
                line = f"[{ws.title}] columns: " + " | ".join(c for c in r if c)
            elif header is not None:
                pairs = [f"{header[j] if j < len(header) and header[j] else f'Col{j + 1}'}: {c}"
                         for j, c in enumerate(r) if c]
                line = f"[{ws.title}] " + "; ".join(pairs)
            else:
                # free text or label/value lines; merged cells repeat, keep one copy
                line = f"[{ws.title}] " + " | ".join(dict.fromkeys(c for c in r if c))
            if line not in seen:
                seen.add(line)
                out.append(line)
        if out:
            parts.append(f"## Sheet: {ws.title}\n" + "\n".join(out))
    return "\n\n".join(parts)


def docx_text(data: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        xml = z.read("word/document.xml").decode("utf-8", errors="replace")
    xml = re.sub(r"</w:p>", "\n", xml)
    return re.sub(r"<[^>]+>", "", xml)


@dataclass
class DriveFile:
    id: str
    name: str
    mime: str
    modified: str
    url: str
    path: str  # folder path inside the shared root, e.g. "Team Docs / Engineering"


class DriveAuth:
    """Bearer tokens from a service account key (preferred) or an API key (public folders only)."""

    def __init__(self, sa_file: Path, api_key: str = ""):
        self.sa_file = sa_file
        self.api_key = api_key
        self._creds = None
        self._lock = threading.Lock()

    @property
    def mode(self) -> str:
        if self.sa_file and self.sa_file.exists():
            return "service_account"
        if self.api_key:
            return "api_key"
        return ""

    def account_email(self) -> str:
        try:
            return json.loads(self.sa_file.read_text()).get("client_email", "")
        except Exception:
            return ""

    def _token_sync(self) -> str:
        from google.auth.transport.requests import Request
        from google.oauth2 import service_account

        with self._lock:
            if self._creds is None:
                self._creds = service_account.Credentials.from_service_account_file(str(self.sa_file), scopes=[SCOPE])
            if not self._creds.valid:
                self._creds.refresh(Request())
            return self._creds.token

    async def params_headers(self) -> tuple[dict, dict]:
        if self.mode == "service_account":
            token = await asyncio.get_running_loop().run_in_executor(None, self._token_sync)
            return {}, {"Authorization": f"Bearer {token}"}
        if self.mode == "api_key":
            return {"key": self.api_key}, {}
        raise RuntimeError("no Google Drive credentials configured")


class DriveIndexer:
    def __init__(self, kb, auth: DriveAuth, refresh_hours: int = 6):
        self.kb = kb
        self.auth = auth
        self.refresh_hours = refresh_hours
        self._busy: set[str] = set()
        self._http = None
        self.db: sqlite3.Connection = kb.db
        with kb._lock:
            self.db.executescript(
                """
                CREATE TABLE IF NOT EXISTS drive_folders(
                    folder_id TEXT PRIMARY KEY, guild_id INTEGER, url TEXT, name TEXT, added_by TEXT,
                    last_crawl REAL DEFAULT 0, status TEXT DEFAULT 'new', files INTEGER DEFAULT 0);
                CREATE TABLE IF NOT EXISTS drive_files(
                    file_id TEXT PRIMARY KEY, folder_id TEXT, modified TEXT, status TEXT);
                """
            )
            self.db.commit()

    # registry ---------------------------------------------------------------
    def add_folder(self, folder_id: str, guild_id: int, url: str, added_by: str) -> bool:
        with self.kb._lock:
            cur = self.db.execute(
                "INSERT OR IGNORE INTO drive_folders(folder_id, guild_id, url, added_by) VALUES (?,?,?,?)",
                (folder_id, guild_id, url, added_by),
            )
            self.db.commit()
            return cur.rowcount > 0

    def folders(self, guild_id: Optional[int] = None) -> list[dict]:
        cols = ["folder_id", "guild_id", "url", "name", "added_by", "last_crawl", "status", "files"]
        sql = f"SELECT {', '.join(cols)} FROM drive_folders"
        args: tuple = ()
        if guild_id is not None:
            sql += " WHERE guild_id = ?"
            args = (guild_id,)
        with self.kb._lock:
            return [dict(zip(cols, r)) for r in self.db.execute(sql, args).fetchall()]

    def _set_folder(self, folder_id: str, **kv) -> None:
        sets = ", ".join(f"{k} = ?" for k in kv)
        with self.kb._lock:
            self.db.execute(f"UPDATE drive_folders SET {sets} WHERE folder_id = ?", (*kv.values(), folder_id))
            self.db.commit()

    # http -------------------------------------------------------------------
    async def _session(self):
        import aiohttp

        if self._http is None or self._http.closed:
            self._http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120))
        return self._http

    async def _get(self, path: str, params: dict, raw: bool = False):
        p, h = await self.auth.params_headers()
        s = await self._session()
        async with s.get(API + path, params={**params, **p}, headers=h) as r:
            if r.status != 200:
                body = (await r.text())[:300]
                raise PermissionError(f"{r.status}: {body}") if r.status in (401, 403, 404) else RuntimeError(f"{r.status}: {body}")
            if raw:
                return await _read_all(r)
            return await r.json()

    async def _list(self, folder_id: str) -> list[dict]:
        items, token = [], None
        while True:
            params = {
                "q": f"'{folder_id}' in parents and trashed = false",
                "fields": "nextPageToken, files(id, name, mimeType, modifiedTime, webViewLink, shortcutDetails)",
                "pageSize": "1000", "supportsAllDrives": "true", "includeItemsFromAllDrives": "true",
            }
            if token:
                params["pageToken"] = token
            data = await self._get("/files", params)
            items += data.get("files", [])
            token = data.get("nextPageToken")
            if not token:
                return items

    async def walk(self, root_id: str) -> tuple[str, list[DriveFile]]:
        meta = await self._get(f"/files/{root_id}", {"fields": "id, name, mimeType", "supportsAllDrives": "true"})
        root_name = meta.get("name", "Shared folder")
        out: list[DriveFile] = []
        queue = [(root_id, root_name, 0)]
        seen = {root_id}
        while queue and len(out) < MAX_FILES:
            fid, path, depth = queue.pop(0)
            for f in await self._list(fid):
                mime = f.get("mimeType", "")
                target_id = f["id"]
                if mime == "application/vnd.google-apps.shortcut":
                    sd = f.get("shortcutDetails") or {}
                    target_id, mime = sd.get("targetId", f["id"]), sd.get("targetMimeType", "")
                if mime == FOLDER_MIME:
                    if depth < MAX_DEPTH and target_id not in seen:
                        seen.add(target_id)
                        queue.append((target_id, f"{path} / {f['name']}", depth + 1))
                    continue
                if mime in EXPORTS or mime in DOWNLOADS or f["name"].lower().endswith((".md", ".txt", ".csv")):
                    out.append(DriveFile(target_id, f["name"], mime, f.get("modifiedTime", ""),
                                         f.get("webViewLink", f"https://drive.google.com/file/d/{target_id}/view"), path))
        return root_name, out

    async def read(self, f: DriveFile) -> str:
        if f.mime in EXPORTS:
            target = EXPORTS[f.mime][0]
            data = await self._get(f"/files/{f.id}/export", {"mimeType": target}, raw=True)
            if target == XLSX:
                return await asyncio.get_running_loop().run_in_executor(None, xlsx_text, data)
            return data.decode("utf-8", errors="replace").lstrip("\ufeff")
        data = await self._get(f"/files/{f.id}", {"alt": "media", "supportsAllDrives": "true"}, raw=True)
        if f.mime == "application/pdf":
            return await asyncio.get_running_loop().run_in_executor(None, pdf_text, data)
        if f.mime.endswith("wordprocessingml.document"):
            return docx_text(data)
        if f.mime == XLSX:
            return await asyncio.get_running_loop().run_in_executor(None, xlsx_text, data)
        return data.decode("utf-8", errors="replace")

    # public link mode (no credentials) ------------------------------------------
    # Folders and files shared as "anyone with the link" can be read with no key at all:
    # the folder's embedded view lists its items, and Docs/Sheets/Slides have public export URLs.
    async def _public_get(self, url: str, raw: bool = True):
        s = await self._session()
        async with s.get(url, headers={"User-Agent": "Mozilla/5.0 (RockyBot)"}, allow_redirects=True) as r:
            if r.status in (401, 403, 404):
                raise PermissionError(f"{r.status}")
            if r.status != 200:
                raise RuntimeError(f"{r.status}")
            data = await _read_all(r)
            ctype = r.headers.get("Content-Type", "")
            if raw and "text/html" in ctype and b"accounts.google.com" in data[:20000]:
                raise PermissionError("not shared publicly")
            return data

    @staticmethod
    def _public_kind(href: str, name: str) -> tuple[str, str]:
        """(item id, mime) from an embedded folder view link."""
        for pat, mime in ((r"/drive/(?:u/\d+/)?folders/([\w-]+)", FOLDER_MIME),
                          (r"docs\.google\.com/document/d/([\w-]+)", "application/vnd.google-apps.document"),
                          (r"docs\.google\.com/spreadsheets/d/([\w-]+)", "application/vnd.google-apps.spreadsheet"),
                          (r"docs\.google\.com/presentation/d/([\w-]+)", "application/vnd.google-apps.presentation"),
                          (r"/file/d/([\w-]+)", "")):
            m = re.search(pat, href)
            if m:
                if not mime:
                    low = name.lower()
                    mime = ("application/pdf" if low.endswith(".pdf") else
                            "application/vnd.openxmlformats-officedocument.wordprocessingml.document" if low.endswith(".docx") else
                            XLSX if low.endswith(".xlsx") else
                            "text/csv" if low.endswith(".csv") else
                            "text/plain" if low.endswith((".txt", ".md")) else "")
                return m.group(1), mime
        return "", ""

    async def _public_list(self, folder_id: str) -> tuple[str, list[tuple[str, str, str, str]]]:
        """(folder name, [(id, name, mime, url)]) for a public folder."""
        import html as _html

        data = await self._public_get(f"https://drive.google.com/embeddedfolderview?id={folder_id}", raw=False)
        page = data.decode("utf-8", "replace")
        t = re.search(r"<title>(.*?)</title>", page, re.S)
        name = _html.unescape(t.group(1).strip()) if t else "Shared folder"
        items = []
        for href, title in re.findall(
                r'<div class="flip-entry" id="entry-[\w-]+".*?<a href="([^"]+)".*?<div class="flip-entry-title">(.*?)</div>',
                page, re.S):
            href, title = _html.unescape(href), _html.unescape(re.sub(r"<[^>]+>", "", title)).strip()
            iid, mime = self._public_kind(href, title)
            if iid:
                items.append((iid, title, mime, href))
        if not items and "flip-entry" not in page and "accounts.google.com" in page:
            raise PermissionError("folder is not shared as anyone with the link")
        return name, items

    async def walk_public(self, root_id: str) -> tuple[str, list[DriveFile]]:
        root_name, _ = await self._public_list(root_id)
        # public pages carry no modified time: re-read each refresh period instead
        stamp = f"public:{int(time.time() // max(self.refresh_hours * 3600, 3600))}"
        out: list[DriveFile] = []
        queue, seen = [(root_id, root_name, 0)], {root_id}
        while queue and len(out) < MAX_FILES:
            fid, path, depth = queue.pop(0)
            try:
                _, items = await self._public_list(fid)
            except PermissionError:
                log.info("drive: subfolder %s (%s) is not public, skipped", path, fid)
                continue
            for iid, name, mime, href in items:
                if mime == FOLDER_MIME:
                    if depth < MAX_DEPTH and iid not in seen:
                        seen.add(iid)
                        queue.append((iid, f"{path} / {name}", depth + 1))
                elif mime:
                    out.append(DriveFile(iid, name, mime, stamp, href, path))
            await asyncio.sleep(0.2)
        return root_name, out

    async def read_public(self, f: DriveFile) -> str:
        if f.mime == "application/vnd.google-apps.document":
            data = await self._public_get(f"https://docs.google.com/document/d/{f.id}/export?format=txt")
            return data.decode("utf-8", "replace").lstrip("﻿")
        if f.mime == "application/vnd.google-apps.spreadsheet":
            data = await self._public_get(f"https://docs.google.com/spreadsheets/d/{f.id}/export?format=xlsx")
            return await asyncio.get_running_loop().run_in_executor(None, xlsx_text, data)
        if f.mime == "application/vnd.google-apps.presentation":
            data = await self._public_get(f"https://docs.google.com/presentation/d/{f.id}/export/txt")
            return data.decode("utf-8", "replace").lstrip("﻿")
        data = await self._public_get(f"https://drive.google.com/uc?export=download&id={f.id}")
        if f.mime == "application/pdf":
            return await asyncio.get_running_loop().run_in_executor(None, pdf_text, data)
        if f.mime.endswith("wordprocessingml.document"):
            return docx_text(data)
        if f.mime == XLSX:
            return await asyncio.get_running_loop().run_in_executor(None, xlsx_text, data)
        return data.decode("utf-8", errors="replace")

    # crawl --------------------------------------------------------------------
    async def crawl(self, folder: dict, force: bool = False) -> str:
        fid, guild_id = folder["folder_id"], folder["guild_id"]
        if fid in self._busy:
            return "busy"
        self._busy.add(fid)
        try:
            files, root_name, reader = None, None, self.read
            if self.auth.mode:
                try:
                    root_name, files = await self.walk(fid)
                except PermissionError as e:
                    log.info("drive folder %s not readable with credentials (%s); trying public link", fid, str(e)[:80])
            if files is None:
                try:
                    root_name, files = await self.walk_public(fid)
                    reader = self.read_public
                except PermissionError as e:
                    log.warning("drive folder %s not readable: %s", fid, e)
                    self._set_folder(fid, status="no_access", last_crawl=time.time())
                    return "no_access"
            ok, skipped, failed = await self._index_files(fid, guild_id, files, force, reader)
            self._set_folder(fid, status="ok", name=root_name, files=len(files), last_crawl=time.time())
            log.info("drive folder %r: %d files (%d updated, %d unchanged, %d unreadable)",
                     root_name, len(files), ok, skipped, failed)
            return "ok"
        finally:
            self._busy.discard(fid)

    async def _index_files(self, fid: str, guild_id: int, files: list[DriveFile], force: bool = False,
                           reader=None) -> tuple[int, int, int]:
        """Index changed files under a folder key, drop files that disappeared. Returns (updated, unchanged, failed)."""
        with self.kb._lock:
            rows = self.db.execute("SELECT file_id, modified, status FROM drive_files WHERE folder_id = ?", (fid,)).fetchall()
        known = {r[0]: r[1] for r in rows}
        retry = {r[0] for r in rows if r[2] == "error"}  # unreadable last time: try again
        ok = skipped = failed = 0
        current = set()
        for f in files:
            current.add(f.id)
            if not force and f.id not in retry and known.get(f.id) == f.modified:
                skipped += 1
                continue
            label = EXPORTS.get(f.mime, (None, DOWNLOADS.get(f.mime, "File")))[1]
            try:
                text = await (reader or self.read)(f)
                status = "ok" if text.strip() else "empty"
            except Exception as e:
                log.info("drive file %s (%s) unreadable: %s", f.name, f.id, str(e)[:120])
                text, status = "", "error"
                failed += 1
            body = text if text.strip() else f"{f.name} (contents could not be read)"
            self.kb.upsert(f"gdrive:{f.id}", "gdoc", guild_id, f"Drive: {f.path}", "Google Drive", f.url,
                           f"{label}: {f.name}", body, time.time())
            with self.kb._lock:
                self.db.execute("INSERT OR REPLACE INTO drive_files(file_id, folder_id, modified, status) VALUES (?,?,?,?)",
                                (f.id, fid, f.modified, status))
                self.db.commit()
            ok += status == "ok"
            await asyncio.sleep(0.1)
        # forget files that were removed from the folder
        for gone in set(known) - current:
            self.kb.delete(f"gdrive:{gone}")
            with self.kb._lock:
                self.db.execute("DELETE FROM drive_files WHERE file_id = ?", (gone,))
                self.db.commit()
        return ok, skipped, failed

    async def discover_shared(self, guild_id: Optional[int]) -> tuple[int, int]:
        """Pick up everything people shared with Rocky's account: folders get crawled, files get indexed.

        Returns (new_folders, files_indexed). No Discord link needed.
        """
        if guild_id is None or self.auth.mode != "service_account":
            return 0, 0
        items, token = [], None
        while True:
            params = {
                "q": "sharedWithMe = true and trashed = false",
                "fields": "nextPageToken, files(id, name, mimeType, modifiedTime, webViewLink, shortcutDetails)",
                "pageSize": "1000", "supportsAllDrives": "true", "includeItemsFromAllDrives": "true",
            }
            if token:
                params["pageToken"] = token
            data = await self._get("/files", params)
            items += data.get("files", [])
            token = data.get("nextPageToken")
            if not token:
                break
        new_folders, files = 0, []
        for f in items:
            mime = f.get("mimeType", "")
            if mime == FOLDER_MIME:
                url = f.get("webViewLink") or f"https://drive.google.com/drive/folders/{f['id']}"
                if self.add_folder(f["id"], guild_id, url, "shared with Rocky"):
                    new_folders += 1
                continue
            if mime in EXPORTS or mime in DOWNLOADS or f["name"].lower().endswith((".md", ".txt", ".csv")):
                files.append(DriveFile(f["id"], f["name"], mime, f.get("modifiedTime", ""),
                                       f.get("webViewLink", f"https://drive.google.com/file/d/{f['id']}/view"),
                                       "Shared with Rocky"))
        ok, _, _ = await self._index_files("__shared__", guild_id, files)
        if new_folders or ok:
            log.info("drive: discovered %d new shared folders, indexed %d shared files", new_folders, ok)
        return new_folders, ok

    async def refresh_loop(self, guild_id_fn=None) -> None:
        while True:
            try:
                await self.discover_shared(guild_id_fn() if guild_id_fn else None)
            except Exception:
                log.exception("drive discovery failed")
            for f in self.folders():
                if time.time() - (f["last_crawl"] or 0) >= self.refresh_hours * 3600 or f["status"] in ("new",):
                    try:
                        await self.crawl(f)
                    except Exception:
                        log.exception("drive crawl failed for %s", f["folder_id"])
            await asyncio.sleep(120)

    async def close(self) -> None:
        if self._http:
            await self._http.close()
