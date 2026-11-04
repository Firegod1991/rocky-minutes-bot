"""Environment driven configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    v = os.getenv(name)
    try:
        return int(v) if v not in (None, "") else default
    except ValueError:
        return default


def _id_list(name: str) -> set[int]:
    raw = os.getenv(name, "")
    out: set[int] = set()
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if part.isdigit():
            out.add(int(part))
    return out


@dataclass
class Config:
    discord_token: str = field(default_factory=lambda: os.getenv("DISCORD_TOKEN", ""))
    # Extra bot tokens. Discord allows one voice channel per bot account per server,
    # so each extra token lets one more meeting be recorded at the same time.
    extra_tokens: list[str] = field(default_factory=lambda: [
        t.strip() for t in os.getenv("DISCORD_EXTRA_TOKENS", "").replace(";", ",").split(",") if t.strip()
    ])
    post_transcript_inline: bool = field(default_factory=lambda: _bool("POST_TRANSCRIPT_INLINE", True))
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    claude_model: str = field(default_factory=lambda: os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5"))

    # Whisper
    whisper_model: str = field(default_factory=lambda: os.getenv("WHISPER_MODEL", "small"))
    whisper_device: str = field(default_factory=lambda: os.getenv("WHISPER_DEVICE", "cpu"))
    whisper_compute_type: str = field(default_factory=lambda: os.getenv("WHISPER_COMPUTE_TYPE", "int8"))
    whisper_language: str | None = field(default_factory=lambda: os.getenv("WHISPER_LANGUAGE") or None)
    whisper_threads: int = field(default_factory=lambda: _int("WHISPER_THREADS", 0))
    # Vocabulary hint, e.g. "OSINT, Sherlock, Maigret, SpiderFoot, Firegod" helps with jargon and names
    whisper_prompt: str | None = field(default_factory=lambda: os.getenv("WHISPER_PROMPT") or None)

    # Behaviour
    minutes_channel_name: str = field(default_factory=lambda: os.getenv("MINUTES_CHANNEL_NAME", "meeting-minutes"))
    min_participants: int = field(default_factory=lambda: _int("MIN_PARTICIPANTS", 1))
    leave_grace_seconds: int = field(default_factory=lambda: _int("LEAVE_GRACE_SECONDS", 30))
    publish_delay_seconds: int = field(default_factory=lambda: _int("PUBLISH_DELAY_SECONDS", 120))
    min_meeting_seconds: int = field(default_factory=lambda: _int("MIN_MEETING_SECONDS", 60))
    max_meeting_hours: int = field(default_factory=lambda: _int("MAX_MEETING_HOURS", 6))
    announce_recording: bool = field(default_factory=lambda: _bool("ANNOUNCE_RECORDING", True))
    keep_audio: bool = field(default_factory=lambda: _bool("KEEP_AUDIO", False))
    voice_allowlist: set[int] = field(default_factory=lambda: _id_list("VOICE_CHANNEL_ALLOWLIST"))
    voice_denylist: set[int] = field(default_factory=lambda: _id_list("VOICE_CHANNEL_DENYLIST"))
    guild_ids: set[int] = field(default_factory=lambda: _id_list("GUILD_IDS"))
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "./data")))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    # Assistant: voice commands, @mentions, knowledge base
    voice_commands: bool = field(default_factory=lambda: _bool("VOICE_COMMANDS", True))
    text_commands: bool = field(default_factory=lambda: _bool("TEXT_COMMANDS", True))
    wake_words: list[str] = field(default_factory=lambda: [
        w.strip().lower() for w in os.getenv("WAKE_WORDS", "rocky").split(",") if w.strip()
    ])
    # small fast model for live wake word listening; the big one still does the final notes
    live_model: str = field(default_factory=lambda: os.getenv("WHISPER_LIVE_MODEL", "base.en"))
    index_history_limit: int = field(default_factory=lambda: _int("INDEX_HISTORY_LIMIT", 2000))
    doc_refresh_hours: int = field(default_factory=lambda: _int("DOC_REFRESH_HOURS", 6))

    def validate(self) -> None:
        if not self.all_tokens:
            raise SystemExit("DISCORD_TOKEN is required")
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @property
    def all_tokens(self) -> list[str]:
        """Primary token first, then extras; blanks and duplicates dropped.

        If DISCORD_TOKEN is empty but DISCORD_EXTRA_TOKENS has one, the first
        extra becomes the primary so a token on the wrong line still works.
        """
        seen: list[str] = []
        for t in [self.discord_token.strip(), *self.extra_tokens]:
            if t and t not in seen:
                seen.append(t)
        return seen

    def channel_allowed(self, channel_id: int) -> bool:
        if channel_id in self.voice_denylist:
            return False
        if self.voice_allowlist and channel_id not in self.voice_allowlist:
            return False
        return True
