import time
from pathlib import Path

import numpy as np

from bot.assistant import format_find, parse_command, parse_wake, wake_pattern
from bot.knowledge import KnowledgeBase, chunks, fts_query, google_links

W = wake_pattern(["rocky"])


def test_wake_variants():
    assert parse_wake("Rocky, pull up the risk register.", W) == "pull up the risk register"
    assert parse_wake("Hey Rocky pull up the doc", W) == "pull up the doc"
    assert parse_wake("Rockie: stop recording", W) == "stop recording"
    assert parse_wake("rocky", W) == ""
    assert parse_wake("Rocky.", W) == ""
    assert parse_wake("Okay, Rocky, what did we decide?", W) == "what did we decide?"
    assert parse_wake("I think Rocky should do it", W) is None
    assert parse_wake("the rocket is ready", W) is None


def test_parse_commands():
    assert parse_command("pull up the google doc talking about computer systems").kind == "find"
    assert parse_command("pull up the google doc talking about computer systems").arg == "the google doc talking about computer systems"
    assert parse_command("can you bring up the risk register").kind == "find"
    assert parse_command("where's the thermal spreadsheet").kind == "find"
    c = parse_command("action item: Michael finishes the thermal model by Friday")
    assert (c.kind, c.arg) == ("action", "Michael finishes the thermal model by Friday")
    assert parse_command("add an action item Paige sends the budget").kind == "action"
    assert parse_command("stop recording").kind == "stop"
    assert parse_command("take notes now").kind == "notes"
    assert parse_command("post the minutes").kind == "notes"
    assert parse_command("help").kind == "help"
    assert parse_command("").kind == "help"
    c = parse_command("what did we decide about the launch date?")
    assert c.kind == "ask"


def test_google_links():
    t = ("see https://docs.google.com/document/d/1AbCdEfGhIjKlMnOpQrStUvWxYz0123456789/edit?usp=sharing and "
         "<https://docs.google.com/spreadsheets/d/1ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ/edit#gid=0>")
    links = google_links(t)
    assert [(k, i[:4]) for k, i, _ in links] == [("document", "1AbC"), ("spreadsheets", "1ZZZ")]
    assert not links[1][2].endswith(">")


def test_fts_query_drops_filler():
    q = fts_query("the google doc talking about computer systems")
    assert '"computer"*' in q and '"systems"*' in q and "google" not in q and "talking" not in q


def test_chunks():
    text = "\n".join(f"paragraph {i} " + "x" * 300 for i in range(20))
    parts = chunks(text, 1500)
    assert all(len(p) <= 1500 for p in parts) and len(parts) > 3


def test_search_prefers_files_and_dedupes(tmp_path: Path):
    kb = KnowledgeBase(tmp_path / "kb.db")
    now = time.time()
    kb.upsert("msg:1", "msg", 1, "general", "fire", "https://discord.com/x/1",
              "computer chat", "hey has anyone seen the computer systems doc", now)
    body = "Computer Systems Design\n" + "\n".join(f"The flight computer uses subsystem {i}." for i in range(200))
    kb.upsert("gdoc:abc", "gdoc", 1, "important-links", "mike", "https://docs.google.com/document/d/abc",
              "Google Doc: Computer Systems Design", body, now)
    kb.upsert("gdoc:zzz", "gdoc", 2, "x", "y", "https://docs.google.com/document/d/zzz",
              "Google Doc: Computer Systems (other server)", "computer systems", now)
    hits = kb.search("the google doc talking about computer systems", guild_id=1)
    assert hits[0].url == "https://docs.google.com/document/d/abc"
    assert sum(1 for h in hits if h.url.endswith("/abc")) == 1  # chunks deduped
    assert all("zzz" not in h.url for h in hits)  # other guild excluded
    files = kb.search("computer systems", guild_id=1, kinds=("gdoc",))
    assert len(files) == 1
    out = format_find("computer systems", hits)
    assert "Computer Systems Design" in out and "docs.google.com/document/d/abc" in out
    # re-upsert replaces, delete removes
    kb.upsert("msg:1", "msg", 1, "general", "fire", "u", "t", "totally different words", now)
    assert all(h.key != "msg:1" for h in kb.search("computer", guild_id=1))
    kb.delete("gdoc:abc")
    assert not kb.search("subsystem", guild_id=1)


def test_live_listener_endpointing():
    import asyncio
    import bot.live as live

    heard = []

    async def on_text(uid, name, t, text):
        heard.append((uid, name, text))

    async def run():
        loop = asyncio.get_running_loop()
        lst = live.LiveListener(loop, on_text, "unused", transcriber=lambda a: f"len={a.size}")
        frame = np.ones(320, dtype=np.int16) * 1000  # 20 ms at 16 kHz
        for _ in range(50):  # 1 s of speech from user 1
            lst.feed(1, "fire", frame)
        for _ in range(5):  # 0.1 s from user 2: too short, dropped
            lst.feed(2, "mike", frame)
        await asyncio.sleep(1.4)
        lst.close()
        await asyncio.sleep(0.2)
        return lst

    lst = asyncio.run(run())
    assert heard == [(1, "fire", "len=16000")]
    assert "fire: len=16000" in lst.transcript_so_far()


def test_docs_and_slash_prefix():
    for t in ("/ask docs", "docs", "list docs", "what docs do you have?", "show me all the files", "/ask list documents"):
        assert parse_command(t).kind == "docs", t
    assert parse_command("/ask pull up the risk register").kind == "find"
    assert parse_command("pull up the doc about docs").kind == "find"


def test_format_docs_and_mentions():
    from bot.assistant import format_docs, _clean
    assert "Drive folder link" in format_docs([])
    out = format_docs([("Google Doc: Org Chart", "https://docs.google.com/document/d/x", "ok"),
                       ("Google Sheet: Budget", "https://docs.google.com/spreadsheets/d/y", "private")])
    assert "[Google Doc: Org Chart](https://docs.google.com/document/d/x)" in out and "link only" in out
    assert _clean("<@123> hi <@&456>") == "@someone hi @someone"


def test_minutes_long_meeting_chunked_on_local_model(monkeypatch):
    import asyncio
    from datetime import datetime, timezone
    from bot import llm, minutes
    from bot.config import Config
    from bot.merge import Segment

    calls = []

    async def fake_complete(cfg, system, user, max_tokens=1500):
        calls.append(user)
        return "**Summary**\nok" if "Write the minutes" in user else "part notes"

    monkeypatch.setattr(llm, "complete", fake_complete)
    cfg = Config(discord_token="x", anthropic_api_key="", ollama_url="http://x", ollama_ctx=8192)
    transcript = "\n".join(f"[00:00:{i % 60:02d}] fire: " + "word " * 40 for i in range(400))
    md, mode = asyncio.run(minutes.generate_minutes(
        cfg, "T", datetime.now(timezone.utc), 60, ["fire"], transcript, [Segment(0, 1, "fire", "x")]))
    assert mode == "ollama" and "ok" in md
    assert len(calls) > 2 and "part notes" in calls[-1]  # parts first, then final minutes from the notes

    cfg2 = Config(discord_token="x", anthropic_api_key="", ollama_url="")
    assert not llm.enabled(cfg2)


def test_provider_order():
    from bot import llm
    from bot.config import Config
    base = dict(discord_token="x", ollama_url="http://x")
    assert llm.provider(Config(**base, anthropic_api_key="a", gemini_api_key="g")) == "claude"
    assert llm.provider(Config(**base, anthropic_api_key="", gemini_api_key="g")) == "gemini"
    assert llm.provider(Config(**base, anthropic_api_key="", gemini_api_key="")) == "ollama"


def test_cloud_failure_falls_back_to_ollama(monkeypatch):
    import asyncio
    from bot import llm
    from bot.config import Config
    seen = []

    async def fake(cfg, p, system, user, max_tokens, fast=False):
        seen.append(p)
        if p == "gemini":
            raise RuntimeError("404 model retired")
        return "local answer"

    monkeypatch.setattr(llm, "_complete", fake)
    cfg = Config(discord_token="x", anthropic_api_key="", gemini_api_key="g", ollama_url="http://x")
    assert asyncio.run(llm.complete(cfg, "s", "u")) == "local answer"
    assert seen == ["gemini", "gemini", "ollama"]  # main model, then lite model, then local


def test_sentences_split_and_glue():
    from bot.speech import sentences
    assert sentences("Found it. Link's in the chat. Ok.") == ["Found it. Link's in the chat. Ok."] or \
        sentences("Found it. Link's in the chat. Ok.") == ["Found it.", "Link's in the chat. Ok."]
    assert sentences("The rover weighs about 155 kilograms. We have twelve kilograms of margin left.") == [
        "The rover weighs about 155 kilograms.", "We have twelve kilograms of margin left."]


def test_streaming_source_frames():
    import threading
    from bot.speech import FRAME, StreamingPCM
    src = StreamingPCM(lead_ms=0)
    threading.Timer(0.05, lambda: (src.feed(b"\x01" * (FRAME + 100)), src.finish())).start()
    a, b, c = src.read(), src.read(), src.read()
    assert len(a) == FRAME and len(b) == FRAME and b.endswith(b"\x00") and c == b""


def test_speakable_strips_chat_formatting():
    from bot.speech import speakable
    s = speakable("The **total weight** is 155 kg [1], margin 12 kg. <https://docs.google.com/x> [doc](https://x.y) <@123>")
    assert s == "The total weight is 155 kg, margin 12 kg. doc someone"
