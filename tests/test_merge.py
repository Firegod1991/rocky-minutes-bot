from bot.merge import Segment, chunk_text, coalesce, fmt_ts, render_transcript, talk_time


def test_fmt_ts():
    assert fmt_ts(0) == "00:00:00"
    assert fmt_ts(65.9) == "00:01:05"
    assert fmt_ts(3661) == "01:01:01"


def test_coalesce_orders_and_joins_same_speaker():
    segs = [
        Segment(5.0, 6.0, "bob", "second"),
        Segment(0.0, 1.0, "fire", "hello"),
        Segment(1.2, 2.0, "fire", "there"),
        Segment(2.1, 3.0, "bob", "first"),
        Segment(10.0, 11.0, "fire", "  "),
    ]
    out = coalesce(segs)
    assert [(s.speaker, s.text) for s in out] == [
        ("fire", "hello there"),
        ("bob", "first"),
        ("bob", "second"),
    ]
    assert out[0].end == 2.0


def test_coalesce_respects_gap():
    segs = [Segment(0, 1, "a", "x"), Segment(5, 6, "a", "y")]
    assert len(coalesce(segs)) == 2


def test_render_transcript():
    segs = [Segment(61, 62, "fire", "go"), Segment(0, 1, "bob", "hi")]
    assert render_transcript(segs) == "[00:00:00] bob: hi\n[00:01:01] fire: go"


def test_talk_time_sorted_desc():
    segs = [Segment(0, 1, "a", "x"), Segment(0, 5, "b", "y"), Segment(10, 12, "a", "z")]
    assert list(talk_time(segs).items()) == [("b", 5.0), ("a", 3.0)]


def test_chunk_text_respects_limit_and_lines():
    text = "\n".join(f"line {i} " + "x" * 50 for i in range(100))
    chunks = chunk_text(text, 500)
    assert all(len(c) <= 500 for c in chunks)
    assert "".join(chunks) == text
    long = "y" * 1200
    assert [len(c) for c in chunk_text(long, 500)] == [500, 500, 200]
