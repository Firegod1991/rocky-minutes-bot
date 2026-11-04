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
EXPORTS = {
    "application/vnd.google-apps.document": ("text/plain", "Google Doc"),
    "application/vnd.google-apps.spreadsheet": ("text/csv", "Google Sheet"),
    "application/vnd.google-apps.presentation": ("text/plain", "Google Slides"),
}
DOWNLOADS = {
    "application/pdf": "PDF",
    "text/plain": "Text file",
    "text/markdown": "Markdown file",
    "text/csv": "CSV file",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "Word doc",
}
MAX_BYTES = 25_000_000
MAX_FILES = 3000
MAX_DEPTH = 12


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
                data = await r.content.read(MAX_BYTES + 1)
                if len(data) > MAX_BYTES:
                    raise RuntimeError("file too large")
                return data
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
            data = await self._get(f"/files/{f.id}/export", {"mimeType": EXPORTS[f.mime][0]}, raw=True)
            return data.decode("utf-8", errors="replace").lstrip("﻿")
        data = await self._get(f"/files/{f.id}", {"alt": "media", "supportsAllDrives": "true"}, raw=True)
        if f.mime == "application/pdf":
            return await asyncio.get_running_loop().run_in_executor(None, pdf_text, data)
        if f.mime.endswith("wordprocessingml.document"):
            return docx_text(data)
        return data.decode("utf-8", errors="replace")

    # crawl --------------------------------------------------------------------
    async def crawl(self, folder: dict, force: bool = False) -> str:
        fid, guild_id = folder["folder_id"], folder["guild_id"]
        if fid in self._busy:
            return "busy"
        if not self.auth.mode:
            self._set_folder(fid, status="no_credentials")
            return "no_credentials"
        self._busy.add(fid)
        try:
            try:
                root_name, files = await self.walk(fid)
            except PermissionError as e:
                log.warning("drive folder %s not readable: %s", fid, e)
                self._set_folder(fid, status="no_access", last_crawl=time.time())
                return "no_access"
            with self.kb._lock:
                known = dict(self.db.execute("SELECT file_id, modified FROM drive_files WHERE folder_id = ?", (fid,)).fetchall())
            ok = skipped = failed = 0
            current = set()
            for f in files:
                current.add(f.id)
                if not force and known.get(f.id) == f.modified:
                    skipped += 1
                    continue
                label = EXPORTS.get(f.mime, (None, DOWNLOADS.get(f.mime, "File")))[1]
                try:
                    text = await self.read(f)
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
            self._set_folder(fid, status="ok", name=root_name, files=len(files), last_crawl=time.time())
            log.info("drive folder %r: %d files (%d updated, %d unchanged, %d unreadable)",
                     root_name, len(files), ok, skipped, failed)
            return "ok"
        finally:
            self._busy.discard(fid)

    async def refresh_loop(self) -> None:
        while True:
            for f in self.folders():
                if time.time() - (f["last_crawl"] or 0) >= self.refresh_hours * 3600 or f["status"] in ("new",):
                    try:
                        await self.crawl(f)
                    except Exception:
                        log.exception("drive crawl failed for %s", f["folder_id"])
            await asyncio.sleep(300)

    async def close(self) -> None:
        if self._http:
            await self._http.close()
