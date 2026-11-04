import asyncio
import io
import zipfile
from pathlib import Path


from bot.drive import FOLDER_MIME, DriveAuth, DriveIndexer, docx_text, folder_links
from bot.knowledge import KnowledgeBase

TREE = {
    "root": [
        {"id": "doc1", "name": "Computer Systems", "mimeType": "application/vnd.google-apps.document", "modifiedTime": "1"},
        {"id": "sub", "name": "Engineering", "mimeType": FOLDER_MIME},
        {"id": "img", "name": "photo.png", "mimeType": "image/png"},
    ],
    "sub": [
        {"id": "sheet1", "name": "Mass Budget", "mimeType": "application/vnd.google-apps.spreadsheet", "modifiedTime": "1"},
        {"id": "deep", "name": "Thermal", "mimeType": FOLDER_MIME},
    ],
    "deep": [{"id": "notes", "name": "notes.txt", "mimeType": "text/plain", "modifiedTime": "1"}],
}
def _xlsx(sheets):
    from openpyxl import Workbook
    wb = Workbook()
    wb.remove(wb.active)
    for title, rows in sheets.items():
        ws = wb.create_sheet(title)
        for r in rows:
            ws.append(r)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


CONTENT = {
    "doc1": b"Flight computer: dual redundant ARM boards.",
    "sheet1": _xlsx({"Mass": [["subsystem", "mass_kg"], ["chassis", 62], ["total", 155]],
                     "Margins": [["item", "kg"], ["margin", 12]]}),
    "notes": b"radiator panels sized for 40 W",
    "shared1": b"Mission assurance risk: battery thermal runaway, likelihood 2, impact 5",
}
SHARED = [
    {"id": "shared1", "name": "MA Workbook notes", "mimeType": "text/plain", "modifiedTime": "1"},
    {"id": "sharedfolder", "name": "Team Folder", "mimeType": FOLDER_MIME},
]


class FakeDrive(DriveIndexer):
    def __init__(self, kb, deny=False):
        super().__init__(kb, DriveAuth(Path("/nonexistent"), api_key="k"))
        self.deny, self.calls = deny, []

    async def _get(self, path, params, raw=False):
        self.calls.append(path)
        if self.deny:
            raise PermissionError("404: not found")
        if path == "/files":
            if params["q"].startswith("sharedWithMe"):
                return {"files": SHARED}
            fid = params["q"].split("'")[1]
            return {"files": TREE.get(fid, [])}
        if path.startswith("/files/") and not raw:
            return {"id": "root", "name": "Team Docs"}
        fid = path.split("/")[2]
        return CONTENT[fid]


def test_folder_links():
    t = "here https://drive.google.com/drive/folders/1AbCdEfGhIjK_lmnop?usp=sharing and again " \
        "https://drive.google.com/drive/u/0/folders/1AbCdEfGhIjK_lmnop"
    assert folder_links(t) == [("1AbCdEfGhIjK_lmnop", "https://drive.google.com/drive/folders/1AbCdEfGhIjK_lmnop?usp=sharing")]


def test_crawl_recurses_reads_and_is_searchable(tmp_path):
    kb = KnowledgeBase(tmp_path / "kb.db")
    d = FakeDrive(kb)
    assert d.add_folder("root", 1, "https://drive.google.com/drive/folders/root", "fire")
    assert not d.add_folder("root", 1, "x", "fire")  # no duplicates
    assert asyncio.run(d.crawl(d.folders(1)[0])) == "ok"
    f = d.folders(1)[0]
    assert (f["status"], f["name"], f["files"]) == ("ok", "Team Docs", 3)  # png skipped
    hits = kb.search("total mass budget", guild_id=1)
    assert hits and "Mass Budget" in hits[0].title and hits[0].channel == "Drive: Team Docs / Engineering"
    assert kb.search("radiator", guild_id=1)[0].channel == "Drive: Team Docs / Engineering / Thermal"
    margin = kb.search("margin", guild_id=1)  # second tab of the workbook is read too
    assert margin and "## Sheet: Margins" in margin[0].body
    assert len(kb.list_drive_files(1)) == 3
    # second crawl: nothing changed, nothing re-downloaded
    d.calls.clear()
    asyncio.run(d.crawl(d.folders(1)[0]))
    assert not any("/export" in c or c.count("/") == 2 and c != "/files" and "root" not in c for c in d.calls)
    # file removed from the folder disappears from search
    TREE["deep"] = []
    try:
        asyncio.run(d.crawl(d.folders(1)[0]))
        assert not kb.search("radiator", guild_id=1)
    finally:
        TREE["deep"] = [{"id": "notes", "name": "notes.txt", "mimeType": "text/plain", "modifiedTime": "1"}]


def test_no_access_and_no_credentials(tmp_path):
    kb = KnowledgeBase(tmp_path / "kb.db")
    d = FakeDrive(kb, deny=True)
    d.add_folder("root", 1, "u", "fire")
    assert asyncio.run(d.crawl(d.folders(1)[0])) == "no_access"
    nocreds = DriveIndexer(kb, DriveAuth(Path("/nonexistent"), api_key=""))
    nocreds.add_folder("other", 1, "u", "fire")
    assert asyncio.run(nocreds.crawl([f for f in nocreds.folders(1) if f["folder_id"] == "other"][0])) == "no_credentials"


def test_docx_text():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", "<w:document><w:p><w:r><w:t>Hello</w:t></w:r></w:p><w:p><w:t>Rover</w:t></w:p></w:document>")
    assert docx_text(buf.getvalue()).split() == ["Hello", "Rover"]


def test_discover_shared_needs_no_link(tmp_path, monkeypatch):
    kb = KnowledgeBase(tmp_path / "kb.db")
    d = FakeDrive(kb)
    monkeypatch.setattr(DriveAuth, "mode", property(lambda self: "service_account"))
    new_folders, files = asyncio.run(d.discover_shared(1))
    assert (new_folders, files) == (1, 1)
    assert any(f["folder_id"] == "sharedfolder" for f in d.folders(1))
    hit = kb.search("battery thermal runaway", guild_id=1)[0]
    assert hit.channel == "Drive: Shared with Rocky" and "MA Workbook" in hit.title
    assert asyncio.run(d.discover_shared(1)) == (0, 0)  # nothing new the second time
    assert asyncio.run(d.discover_shared(None)) == (0, 0)
