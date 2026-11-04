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
    # Google Gemini (key from https://aistudio.google.com/apikey). Used when set and no Anthropic key.
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.5-flash"))
    # spoken answers use a lighter model for speed
    gemini_fast_model: str = field(default_factory=lambda: os.getenv("GEMINI_FAST_MODEL", "gemini-3.5-flash-lite"))
    # Free local fallback when no cloud key is set. Empty OLLAMA_URL disables it.
    ollama_url: str = field(default_factory=lambda: os.getenv("OLLAMA_URL", "http://host.docker.internal:11434"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen3:8b"))
    # local embedding model for meaning based search (served by Ollama); empty disables it
    embed_model: str = field(default_factory=lambda: os.getenv("EMBED_MODEL", "nomic-embed-text"))
    ollama_ctx: int = field(default_factory=lambda: _int("OLLAMA_CTX", 16384))

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
    # Rocky answers out loud in voice channels (local Piper text to speech)
    voice_replies: bool = field(default_factory=lambda: _bool("VOICE_REPLIES", True))
    # kokoro (natural, default) or piper (lighter); voice names differ per engine
    tts_engine: str = field(default_factory=lambda: os.getenv("TTS_ENGINE", "kokoro").strip().lower())
    tts_voice: str = field(default_factory=lambda: os.getenv("TTS_VOICE", ""))
    tts_speed: float = field(default_factory=lambda: float(os.getenv("TTS_SPEED", "1.05") or 1.05))
    tts_dir: Path = field(default_factory=lambda: Path(os.getenv("TTS_DIR", "/models/tts")))

    @property
    def voice_name(self) -> str:
        return self.tts_voice or ("am_michael" if self.tts_engine == "kokoro" else "en_US-ryan-high")
    # Google Drive folders (read only). Service account key file, or an API key for public folders.
    google_sa_file: Path = field(default_factory=lambda: Path(os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "/data/google-service-account.json")))
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    index_history_limit: int = field(default_factory=lambda: _int("INDEX_HISTORY_LIMIT", 2000))
    doc_refresh_hours: int = field(default_factory=lambda: _int("DOC_REFRESH_HOURS", 6))
    # Web research: "Rocky, research ..." searches the internet (Gemini with Google Search, or Claude web search)
    web_research: bool = field(default_factory=lambda: _bool("WEB_RESEARCH", True))
    # when the team's own docs and chats don't have the answer, look it up online instead
    web_fallback: bool = field(default_factory=lambda: _bool("WEB_FALLBACK", True))
    # optional search APIs used when Google Search grounding isn't available (Wikipedia is the free default)
    brave_api_key: str = field(default_factory=lambda: os.getenv("BRAVE_API_KEY", ""))
    tavily_api_key: str = field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))

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
