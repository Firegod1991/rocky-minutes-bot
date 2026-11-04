"""Find or create the minutes channel and post results there."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import discord

from .config import Config
from .merge import chunk_text

log = logging.getLogger(__name__)


class Settings:
    """Tiny per guild settings store (which channel gets the minutes)."""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, dict] = {}
        if path.exists():
            try:
                self.data = json.loads(path.read_text())
            except Exception:
                log.exception("could not read settings; starting fresh")

    def get_channel_id(self, guild_id: int) -> Optional[int]:
        return self.data.get(str(guild_id), {}).get("minutes_channel_id")

    def set_channel_id(self, guild_id: int, channel_id: int) -> None:
        self.data.setdefault(str(guild_id), {})["minutes_channel_id"] = channel_id
        self.path.write_text(json.dumps(self.data, indent=2))


async def get_minutes_channel(cfg: Config, settings: Settings, guild: discord.Guild) -> Optional[discord.TextChannel]:
    cid = settings.get_channel_id(guild.id)
    if cid:
        ch = guild.get_channel(cid)
        if isinstance(ch, discord.TextChannel):
            return ch
    for ch in guild.text_channels:
        if ch.name == cfg.minutes_channel_name:
            settings.set_channel_id(guild.id, ch.id)
            return ch
    try:
        overwrites = {
            guild.default_role: discord.PermissionOverwrite(send_messages=False, add_reactions=True),
            guild.me: discord.PermissionOverwrite(send_messages=True, attach_files=True, embed_links=True,
                                                  create_public_threads=True, send_messages_in_threads=True),
        }
        ch = await guild.create_text_channel(
            cfg.minutes_channel_name,
            topic="Automatic meeting notes from voice channels. Posted by the minutes bot.",
            overwrites=overwrites,
            reason="Meeting minutes bot: dedicated notes channel",
        )
        settings.set_channel_id(guild.id, ch.id)
        log.info("created #%s in %s", ch.name, guild.name)
        return ch
    except discord.Forbidden:
        log.error("missing Manage Channels in %s; cannot create #%s", guild.name, cfg.minutes_channel_name)
    except Exception:
        log.exception("failed creating minutes channel in %s", guild.name)
    return None


async def post_minutes(
    channel: discord.TextChannel,
    minutes_md: str,
    files: list[Path],
) -> Optional[discord.Message]:
    chunks = chunk_text(minutes_md, 1900)
    first: Optional[discord.Message] = None
    for i, chunk in enumerate(chunks):
        attachments = []
        if i == len(chunks) - 1:
            attachments = [discord.File(str(p), filename=p.name) for p in files if p.exists()]
        msg = await channel.send(chunk, files=attachments or discord.utils.MISSING,
                                 allowed_mentions=discord.AllowedMentions.none())
        first = first or msg
    return first


async def post_transcript_thread(anchor: discord.Message, title: str, transcript: str) -> None:
    """Put the full transcript in a thread under the minutes so the channel stays tidy."""
    if not transcript.strip():
        return
    try:
        thread = await anchor.create_thread(name=f"Transcript: {title}"[:100], auto_archive_duration=1440)
    except Exception:
        log.warning("could not create transcript thread", exc_info=True)
        return
    for chunk in chunk_text(transcript, 1900):
        try:
            await thread.send(chunk, allowed_mentions=discord.AllowedMentions.none())
        except Exception:
            log.warning("transcript chunk failed", exc_info=True)
            break


async def post_notice(channel: Optional[discord.TextChannel], text: str) -> None:
    if channel is None:
        return
    try:
        await channel.send(text, allowed_mentions=discord.AllowedMentions.none())
    except Exception:
        log.warning("could not post notice to #%s", channel.name, exc_info=True)
