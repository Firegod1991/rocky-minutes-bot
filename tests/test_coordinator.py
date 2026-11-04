from types import SimpleNamespace

from bot.config import Config
from bot.main import Coordinator


def _bot(index):
    return SimpleNamespace(index=index, sessions={}, is_ready=lambda: True, get_guild=lambda g: True)


def test_claims_are_exclusive(tmp_path):
    coord = Coordinator(Config(discord_token="x", data_dir=tmp_path))
    a, b = _bot(0), _bot(1)
    assert coord.claim(1, 10, a)
    assert not coord.claim(1, 10, b)
    assert coord.claim(1, 10, a)  # re-claim by owner is fine
    assert coord.claim(1, 11, b)
    coord.release(1, 10, b)  # not the owner, no effect
    assert (1, 10) in coord.claims
    coord.release(1, 10, a)
    assert coord.claim(1, 10, b)


def test_free_bot_skips_busy(tmp_path):
    coord = Coordinator(Config(discord_token="x", data_dir=tmp_path))
    a, b = _bot(0), _bot(1)
    coord.bots.extend([a, b])
    a.sessions[1] = object()
    assert coord.free_bot(1) is b
    b.sessions[1] = object()
    assert coord.free_bot(1) is None


def test_all_tokens_dedup(monkeypatch):
    monkeypatch.setenv("DISCORD_TOKEN", "t1")
    monkeypatch.setenv("DISCORD_EXTRA_TOKENS", "t2, t1,,t3")
    assert Config().all_tokens == ["t1", "t2", "t3"]
