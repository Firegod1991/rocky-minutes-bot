"""Discord meeting minutes bot: auto joins voice, records per speaker, transcribes, posts notes."""
from __future__ import annotations

import asyncio
import json
import logging
import re
import signal
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import discord
from discord import app_commands
from discord.ext import voice_recv

from .assistant import (HELP_TEXT, answer_question, answer_with_hits, embedder_for, format_docs, format_find, parse_command,
                        parse_wake, source_links, wake_pattern)
from .speech import Speaker, speakable
from .config import Config
from .drive import DriveAuth, DriveIndexer
from .indexer import Indexer
from .knowledge import KnowledgeBase, embed_pending
from .live import LiveListener, get_live_model
from .merge import Segment, chunk_text, coalesce, render_transcript
from .minutes import generate_minutes
from .publisher import Settings, get_minutes_channel, post_minutes, post_notice, post_transcript_thread
from .recorder import Session, TimelineSink
from .transcribe import transcribe_session_dir

FOLLOWUP_WINDOW_S = 8.0  # after a bare "Rocky", the same speaker's next sentence is the command

# voice_recv ships its UDP keepalive at 5000 *seconds*; without regular keepalives Discord stops
# sending audio shortly after the bot joins. Same bug as Pycord-Development/pycord#3388.
try:
    from discord.ext.voice_recv import reader as _vr_reader

    _vr_reader.UDPKeepAlive.delay = 5
except Exception:  # pragma: no cover
    pass

log = logging.getLogger("minutes")


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower() or "voice"


def _log_task_error(t: asyncio.Task) -> None:
    if not t.cancelled() and t.exception() is not None:
        log.error("background task failed", exc_info=t.exception())


def _humans(channel: discord.abc.Connectable) -> list[discord.Member]:
    return [m for m in getattr(channel, "members", []) if not m.bot]


class Coordinator:
    """Shared state between bot accounts running in this process.

    Discord allows one voice connection per bot account per server. With extra
    tokens, several accounts cover several simultaneous meetings; this object
    makes sure two accounts never record the same channel.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.bots: list["MinutesBot"] = []
        self.claims: dict[tuple[int, int], int] = {}  # (guild_id, channel_id) -> bot index
        self.settings = Settings(cfg.data_dir / "settings.json")
        self.channel_lock = asyncio.Lock()
        # one worker: whisper is CPU heavy, process meetings one at a time
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper")
        self.kb = KnowledgeBase(cfg.data_dir / "knowledge.db")
        self.indexer = Indexer(cfg, self.kb)
        self.drive = DriveIndexer(self.kb, DriveAuth(cfg.google_sa_file, cfg.google_api_key), cfg.doc_refresh_hours)
        self.indexer.drive = self.drive
        self.wake = wake_pattern(cfg.wake_words)
        # channels told to stop by voice/text; Rocky stays out until they empty
        self.paused: set[tuple[int, int]] = set()

    def claim(self, guild_id: int, channel_id: int, bot: "MinutesBot") -> bool:
        key = (guild_id, channel_id)
        if key in self.claims and self.claims[key] != bot.index:
            return False
        self.claims[key] = bot.index
        return True

    def release(self, guild_id: int, channel_id: int, bot: "MinutesBot") -> None:
        if self.claims.get((guild_id, channel_id)) == bot.index:
            del self.claims[(guild_id, channel_id)]

    def free_bot(self, guild_id: int) -> Optional["MinutesBot"]:
        for b in self.bots:
            if b.is_ready() and guild_id not in b.sessions and b.get_guild(guild_id):
                return b
        return None


class MinutesBot(discord.Client):
    def __init__(self, cfg: Config, coord: Coordinator, index: int = 0):
        intents = discord.Intents.none()
        intents.guilds = True
        intents.voice_states = True
        intents.members = True  # privileged: needed for display names in notes
        if index == 0 and cfg.text_commands:
            intents.guild_messages = True
            intents.message_content = True  # privileged: needed to index chat and answer @mentions
        super().__init__(intents=intents)
        self.cfg = cfg
        self.coord = coord
        self.index = index
        self.primary = index == 0
        self.tree = app_commands.CommandTree(self)
        self.sessions: dict[int, Session] = {}  # guild_id -> active session
        self.settings = coord.settings
        self.sessions_dir = cfg.data_dir / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.executor = coord.executor
        self.speaker = Speaker(cfg.tts_dir / cfg.tts_engine, cfg.voice_name, cfg.tts_engine, cfg.tts_speed)
        self._bg: set[asyncio.Task] = set()
        if self.primary:
            self._register_commands()
        coord.bots.append(self)

    # lifecycle ------------------------------------------------------------
    async def setup_hook(self) -> None:
        if not self.primary:
            return
        if self.cfg.guild_ids:
            for gid in self.cfg.guild_ids:
                g = discord.Object(id=gid)
                self.tree.copy_global_to(guild=g)
                await self.tree.sync(guild=g)
            log.info("slash commands synced to guilds %s", sorted(self.cfg.guild_ids))
        else:
            await self.tree.sync()
            log.info("slash commands synced globally (can take up to an hour to appear)")
        self.coord.indexer.start()
        self.coord.indexer.on_new_folder = self._on_new_folder
        self._spawn(self.coord.drive.refresh_loop(self._home_guild_id))
        self._spawn(self._embed_loop())
        self._spawn(self._resume_unprocessed())

    async def _ensure_avatar(self) -> None:
        """Give every Rocky account the Rocky logo if it still has Discord's default avatar."""
        path = Path(__file__).parent / "assets" / "rocky-avatar.png"
        if self.user.avatar is not None or not path.exists():
            return
        try:
            await self.user.edit(avatar=path.read_bytes())
            log.info("[bot %d] avatar set to the Rocky logo", self.index)
        except Exception as e:  # rate limited or not allowed: harmless, try again next start
            log.warning("[bot %d] could not set avatar: %s", self.index, e)

    async def on_ready(self) -> None:
        log.info("[bot %d] logged in as %s (%s); %d guilds", self.index, self.user, self.user.id, len(self.guilds))
        self._spawn(self._ensure_avatar())
        # catch meetings already in progress when the bot starts
        for guild in self.guilds:
            await self._scan_guild(guild)
        if not getattr(self, "_warmed", False):
            # load the voice and the live listener now so the first reply has no cold start
            self._warmed = True
            loop = asyncio.get_running_loop()
            if self.cfg.voice_replies:
                loop.run_in_executor(None, self.speaker.warm)
            if self.cfg.voice_commands:
                loop.run_in_executor(None, get_live_model, self.cfg.live_model)
        if self.primary and self.cfg.text_commands and not getattr(self, "_backfilled", False):
            self._backfilled = True
            for guild in self.guilds:
                self._spawn(self.coord.indexer.backfill(guild, self.settings.get_channel_id(guild.id)))

    async def _scan_guild(self, guild: discord.Guild) -> None:
        """Join the busiest eligible voice channel if we are idle in this guild."""
        if guild.id in self.sessions:
            return
        candidates = [
            vc for vc in guild.voice_channels + guild.stage_channels
            if self._eligible(guild, vc) and len(_humans(vc)) >= self.cfg.min_participants
            and (guild.id, vc.id) not in self.coord.claims and (guild.id, vc.id) not in self.coord.paused
        ]
        if candidates:
            await self.start_session(max(candidates, key=lambda c: len(_humans(c))))

    async def close(self) -> None:
        log.info("[bot %d] shutting down; finalizing %d active session(s)", self.index, len(self.sessions))
        for gid in list(self.sessions):
            try:
                await self.end_session(gid, reason="bot shutdown", process=False)
            except Exception:
                log.exception("finalizing session for guild %s", gid)
        if self.primary:
            await self.coord.indexer.close()
        await super().close()

    async def _minutes_channel(self, guild: discord.Guild):
        async with self.coord.channel_lock:
            return await get_minutes_channel(self.cfg, self.settings, guild)

    def _spawn(self, coro) -> asyncio.Task:
        t = asyncio.create_task(coro)
        self._bg.add(t)
        t.add_done_callback(self._bg.discard)
        t.add_done_callback(_log_task_error)
        return t

    # voice events ---------------------------------------------------------
    def _eligible(self, guild: discord.Guild, channel) -> bool:
        if not isinstance(channel, (discord.VoiceChannel, discord.StageChannel)):
            return False
        if guild.afk_channel and channel.id == guild.afk_channel.id:
            return False
        if not self.cfg.channel_allowed(channel.id):
            return False
        perms = channel.permissions_for(guild.me)
        return perms.connect and perms.view_channel

    async def on_voice_state_update(self, member: discord.Member, before: discord.VoiceState, after: discord.VoiceState) -> None:
        guild = member.guild
        session = self.sessions.get(guild.id)
        if member.id == self.user.id:
            # we got kicked, moved, or disconnected
            if session is None:
                return
            if after.channel is None:
                log.warning("bot removed from voice in %s", guild.name)
                await self.end_session(guild.id, reason="bot disconnected")
            elif after.channel.id != session.channel_id:
                log.info("bot was moved to #%s; following", after.channel.name)
                self._retarget(session, after.channel)
            return
        if member.bot or before.channel == after.channel:
            return

        # a channel emptied: forget any "stop recording" for it
        if before.channel is not None and not _humans(before.channel):
            self.coord.paused.discard((guild.id, before.channel.id))

        # someone joined a channel
        if after.channel is not None and self._eligible(guild, after.channel):
            if session is None:
                if (len(_humans(after.channel)) >= self.cfg.min_participants
                        and (guild.id, after.channel.id) not in self.coord.claims
                        and (guild.id, after.channel.id) not in self.coord.paused):
                    await self.start_session(after.channel)
            elif session.channel_id == after.channel.id:
                session.participants.setdefault(member.id, member.display_name)
                self._cancel_leave(session)
            else:
                # the meeting moved rooms: follow it if the room we record is now empty
                # and no other Rocky account already took the new room
                current = guild.get_channel(session.channel_id)
                taken = (guild.id, after.channel.id) in self.coord.claims
                if (current is None or not _humans(current)) and not taken:
                    await self._follow(session, after.channel)
                    return

        # someone left the channel we are recording
        if session and before.channel is not None and before.channel.id == session.channel_id:
            if not _humans(before.channel):
                self._schedule_leave(session)

    def _retarget(self, session: Session, channel) -> None:
        self.coord.release(session.guild_id, session.channel_id, self)
        session.channel_id = channel.id
        session.channel_name = channel.name
        self.coord.claim(session.guild_id, channel.id, self)
        self._cancel_leave(session)
        for m in _humans(channel):
            session.participants.setdefault(m.id, m.display_name)
        session.save_meta("recording")

    async def _follow(self, session: Session, channel) -> None:
        """Move the live recording to another channel without splitting the notes."""
        guild = channel.guild
        vc = guild.voice_client
        if vc is None:
            return
        try:
            log.info("following meeting from #%s to #%s", session.channel_name, channel.name)
            await vc.move_to(channel)
            self._retarget(session, channel)
        except Exception:
            log.exception("could not follow to #%s; ending session instead", channel.name)
            await self.end_session(session.guild_id, reason="follow failed")
            await self.start_session(channel)

    def _schedule_leave(self, session: Session) -> None:
        self._cancel_leave(session)

        async def _wait_then_leave():
            await asyncio.sleep(self.cfg.leave_grace_seconds)
            guild = self.get_guild(session.guild_id)
            ch = guild.get_channel(session.channel_id) if guild else None
            if ch is None or not _humans(ch):
                await self.end_session(session.guild_id, reason="channel empty")
                if guild:
                    await self._scan_guild(guild)

        session.leave_task = self._spawn(_wait_then_leave())

    @staticmethod
    def _cancel_leave(session: Session) -> None:
        if session.leave_task and not session.leave_task.done():
            session.leave_task.cancel()
        session.leave_task = None

    # session control ------------------------------------------------------
    async def start_session(self, channel: discord.VoiceChannel | discord.StageChannel, manual: bool = False) -> Optional[Session]:
        guild = channel.guild
        if guild.id in self.sessions:
            return self.sessions[guild.id]
        started = datetime.now(timezone.utc)
        sdir = self.sessions_dir / f"{started.strftime('%Y%m%d-%H%M%S')}-{guild.id}-{_slug(channel.name)}"
        session = Session(
            guild_id=guild.id, channel_id=channel.id, channel_name=channel.name,
            guild_name=guild.name, dir=sdir, started_at=started, manual=manual,
        )
        if not self.coord.claim(guild.id, channel.id, self):
            log.info("[bot %d] #%s already covered by another bot account", self.index, channel.name)
            return None
        self.sessions[guild.id] = session
        try:
            vc: voice_recv.VoiceRecvClient = await channel.connect(cls=voice_recv.VoiceRecvClient, self_deaf=False)  # type: ignore[assignment]
            if isinstance(channel, discord.StageChannel):
                try:
                    await guild.me.edit(suppress=False)
                except Exception:
                    pass
            sink = TimelineSink(sdir)
            session.sink = sink
            for m in _humans(channel):
                session.participants[m.id] = m.display_name
            if self.cfg.voice_commands:
                session.live = LiveListener(
                    asyncio.get_running_loop(),
                    lambda uid, name, t, text, s=session: self.on_live_text(s, uid, name, t, text),
                    self.cfg.live_model,
                    # glossary style priming: biases spelling without making Whisper skip the wake word
                    prompt="Glossary: " + ", ".join(
                        [w.title() for w in self.cfg.wake_words] + ([self.cfg.whisper_prompt] if self.cfg.whisper_prompt else [])
                    ) + ".",
                )
                sink.live = session.live
            vc.listen(sink)
            session.save_meta("recording")
            log.info("recording %s / #%s -> %s", guild.name, channel.name, sdir.name)
            if self.cfg.announce_recording:
                notes = await self._minutes_channel(guild)
                when = ("when the meeting ends" if self.cfg.publish_delay_seconds < 60 else
                        f"about {self.cfg.publish_delay_seconds // 60} minutes after the meeting ends")
                await post_notice(notes, f"Recording started in **{channel.name}**. Notes will be posted here {when}.")
            self._spawn(self._max_duration_guard(session))
            self._spawn(self._health(session))
            return session
        except Exception:
            log.exception("could not start recording in %s/#%s", guild.name, channel.name)
            self.sessions.pop(guild.id, None)
            self.coord.release(guild.id, channel.id, self)
            vc = guild.voice_client
            if vc:
                await vc.disconnect(force=True)
            return None

    async def _health(self, session: Session, every: float = 300.0) -> None:
        """Log whether audio is still arriving, so a silent bot is visible in the logs."""
        last = 0
        while True:
            await asyncio.sleep(every)
            if self.sessions.get(session.guild_id) is not session:
                return
            guild = self.get_guild(session.guild_id)
            vc = guild.voice_client if guild else None
            frames = session.sink.frames if session.sink else 0
            diag = ""
            try:
                d = vc.get_recv_diagnostics() if vc is not None else {}
                flat = {k: v for k, v in (d or {}).items() if isinstance(v, (int, float, str, bool))}
                ds = (d or {}).get("dave_session") or {}
                flat.update({f"dave_{k}": v for k, v in ds.items() if isinstance(v, (int, float, str, bool))})
                diag = json.dumps(flat)[:500]
            except Exception as e:
                diag = f"n/a ({type(e).__name__})"
            log.info("health #%s: audio frames +%d (total %d), connected=%s listening=%s, live=%s, recv=%s",
                     session.channel_name, frames - last, frames,
                     bool(vc and vc.is_connected()), bool(vc and getattr(vc, "is_listening", lambda: False)()),
                     session.live.stats if session.live else None, diag)
            last = frames

    async def _max_duration_guard(self, session: Session) -> None:
        await asyncio.sleep(self.cfg.max_meeting_hours * 3600)
        if self.sessions.get(session.guild_id) is session:
            log.info("max duration reached; rolling session in %s", session.guild_name)
            await self.end_session(session.guild_id, reason="max duration")
            guild = self.get_guild(session.guild_id)
            ch = guild.get_channel(session.channel_id) if guild else None
            if ch and _humans(ch):
                await self.start_session(ch)

    async def end_session(self, guild_id: int, reason: str = "", process: bool = True) -> Optional[Session]:
        session = self.sessions.pop(guild_id, None)
        if session is None:
            return None
        self.coord.release(guild_id, session.channel_id, self)
        self._cancel_leave(session)
        session.ended_at = datetime.now(timezone.utc)
        guild = self.get_guild(guild_id)
        vc = guild.voice_client if guild else None
        try:
            if isinstance(vc, voice_recv.VoiceRecvClient) and vc.is_listening():
                vc.stop_listening()  # runs sink.cleanup() synchronously
        except Exception:
            log.exception("stop_listening failed")
        if session.sink and not session.sink.finished:
            session.sink.cleanup()
        if session.live is not None:
            session.live.close()
        try:
            if vc:
                await vc.disconnect(force=True)
        except Exception:
            log.exception("disconnect failed")
        session.save_meta("recorded")
        log.info("ended session %s (%s) after %.0fs", session.dir.name, reason, session.duration)

        if session.duration < self.cfg.min_meeting_seconds and not session.manual:
            log.info("meeting shorter than %ss; discarding", self.cfg.min_meeting_seconds)
            session.save_meta("discarded")
            return session
        if process:
            self._spawn(self._delayed_process(session.dir, self.cfg.publish_delay_seconds))
        return session

    # processing -----------------------------------------------------------
    async def _delayed_process(self, sdir: Path, delay: int) -> None:
        if delay > 0:
            await asyncio.sleep(delay)
        await self.process_session_dir(sdir)

    async def _resume_unprocessed(self) -> None:
        await self.wait_until_ready()
        for sdir in sorted(self.sessions_dir.iterdir()):
            meta = sdir / "meta.json"
            if not meta.exists():
                continue
            try:
                status = json.loads(meta.read_text()).get("status")
            except Exception:
                continue
            if status in ("recorded", "recording", "processing"):
                if status == "recording" and not (sdir / "tracks.json").exists():
                    continue  # crashed mid recording with nothing usable
                log.info("resuming unprocessed session %s (%s)", sdir.name, status)
                await self.process_session_dir(sdir)

    async def process_session_dir(self, sdir: Path) -> None:
        meta_path = sdir / "meta.json"
        meta = json.loads(meta_path.read_text())
        meta["status"] = "processing"
        meta_path.write_text(json.dumps(meta, indent=2))
        guild = self.get_guild(int(meta["guild_id"]))
        if guild is None:
            log.error("guild %s not available; leaving %s for later", meta["guild_id"], sdir.name)
            return
        notes = await self._minutes_channel(guild)
        try:
            loop = asyncio.get_running_loop()
            segments: list[Segment] = await loop.run_in_executor(
                self.executor, transcribe_session_dir, self.cfg, sdir
            )
            merged = coalesce(segments)
            transcript = render_transcript(merged)
            (sdir / "segments.json").write_text(json.dumps([s.__dict__ for s in merged], indent=1))

            started = datetime.fromisoformat(meta["started_at"])
            title = f"Meeting notes: {meta['channel_name']}"
            participants = sorted(set(meta.get("participants", {}).values()) | {s.speaker for s in merged})
            minutes_md, mode = await generate_minutes(
                self.cfg, title, started, float(meta.get("duration_seconds", 0)), participants, transcript, merged,
                logged_items=meta.get("action_items") or [],
            )

            stamp = started.astimezone().strftime("%Y-%m-%d_%H%M")
            base = f"{stamp}_{_slug(meta['channel_name'])}"
            transcript_path = sdir / f"{base}_transcript.txt"
            minutes_path = sdir / f"{base}_minutes.txt"
            transcript_path.write_text(
                f"{title}\nStarted: {started.astimezone():%Y-%m-%d %H:%M %Z}\n"
                f"Participants: {', '.join(participants)}\n\n{transcript or '(no speech detected)'}\n"
            )
            minutes_path.write_text(minutes_md)

            anchor = None
            if notes is None:
                log.error("no minutes channel in %s; files saved in %s", guild.name, sdir)
                meta["status"] = "no_channel"
            else:
                anchor = await post_minutes(notes, minutes_md, [minutes_path, transcript_path])
                if anchor and self.cfg.post_transcript_inline:
                    await post_transcript_thread(anchor, meta["channel_name"], transcript)
                meta["status"] = "published"
                meta["summary_mode"] = mode
                log.info("published minutes for %s to #%s (%s)", sdir.name, notes.name, mode)
            # past meetings become searchable ("Rocky, what did we decide last week about...")
            self.coord.indexer.index_meeting(
                guild.id, sdir.name, meta["channel_name"], f"{title} ({started.astimezone():%b %d})",
                transcript, minutes_md, anchor.jump_url if anchor else "", started.timestamp(),
            )
            if not self.cfg.keep_audio:
                for wav in sdir.glob("*.wav"):
                    wav.unlink(missing_ok=True)
        except Exception:
            log.exception("processing failed for %s", sdir.name)
            meta["status"] = "failed"
            await post_notice(notes, f"Could not produce notes for the meeting in **{meta['channel_name']}** "
                                     f"({meta.get('started_at')}). Check the bot logs.")
        finally:
            meta_path.write_text(json.dumps(meta, indent=2))

    # assistant: voice + text commands ----------------------------------------
    async def on_live_text(self, session: Session, user_id: int, name: str, t: float, text: str) -> None:
        if self.sessions.get(session.guild_id) is not session:
            return
        rest = parse_wake(text, self.coord.wake)
        now = asyncio.get_running_loop().time()
        if rest is None:
            since = session.awaiting.get(user_id)
            if since is None or now - since > FOLLOWUP_WINDOW_S:
                return
            rest = text.strip()
        session.awaiting.pop(user_id, None)
        guild = self.get_guild(session.guild_id)
        channel = guild.get_channel(session.channel_id) if guild else None
        if channel is None:
            return
        say = self._sayer(guild)
        if not rest:
            session.awaiting[user_id] = now
            if say:
                self._spawn(say("Yes?"))
            else:
                await self._send(channel, f"Go ahead, {name}.")
            return
        log.info("voice command from %s in #%s: %s", name, session.channel_name, rest)
        cmd = parse_command(rest)

        async def reply(msg: str) -> None:
            await self._send(channel, f"> {name}: *{rest}*\n{msg}")

        await self.run_command(cmd, guild, reply, asker=name, owner=self, session=session, say=say)

    def _sayer(self, guild: discord.Guild):
        """Coroutine fn that speaks into this bot's voice connection, or None if voice replies are off."""
        if not self.cfg.voice_replies:
            return None
        vc = guild.voice_client

        async def say(text: str) -> None:
            try:
                await self.speaker.say(vc, text)
            except Exception:
                log.exception("speaking failed")

        return say

    async def run_command(self, cmd, guild: discord.Guild, reply, asker: str,
                          owner: Optional["MinutesBot"], session: Optional[Session], say=None) -> None:
        """reply posts text; say (voice only) speaks. Spoken lines stay short; details go to the chat."""
        kb = self.coord.kb

        def speak(text: str) -> None:
            if say:
                self._spawn(say(text))

        if cmd.kind == "help":
            speak("Say Rocky, then ask a question, ask me to pull up a doc, log an action item, "
                  "take notes now, or stop recording. Full list is in the chat.")
            return await reply(HELP_TEXT)
        if cmd.kind == "docs":
            files, drive_files = kb.list_files(guild.id), kb.list_drive_files(guild.id)
            folders = self.coord.drive.folders(guild.id)
            n = len(files) + len(drive_files)
            speak(f"I can search {n} file{'s' if n != 1 else ''}"
                  + (f" across {len(folders)} Drive folder{'s' if len(folders) != 1 else ''}" if folders else "")
                  + ". The list is in the chat." if n or folders else "I don't see any shared Google files yet.")
            return await reply(format_docs(files, folders, drive_files, self.coord.drive.auth.account_email()))
        if cmd.kind == "find":
            hits = kb.search(cmd.arg, guild_id=guild.id, kinds=("gdoc", "file"), limit=3)
            if not hits:
                hits = kb.search(cmd.arg, guild_id=guild.id, limit=3)
            if hits:
                top = speakable(hits[0].title if hits[0].is_file else "a message about it")
                speak(f"Found {top}. Link's in the chat.")
            else:
                speak("I couldn't find anything on that. If it's a Google doc, someone needs to post the link.")
            return await reply(format_find(cmd.arg, hits))
        if cmd.kind == "ask":
            so_far = session.live.transcript_so_far() if session and session.live else ""
            if say:
                text, hits = await answer_with_hits(self.cfg, kb, guild.id, cmd.arg, asker, so_far, spoken=True)
                speak(text)
                links = source_links(hits)
                return await reply(text + (f"\n{links}" if links else ""))
            return await reply(await answer_question(self.cfg, kb, guild.id, cmd.arg, asker, so_far))
        if session is None or owner is None:
            return await reply("I'm not recording a meeting you're in right now. Join the voice channel first.")
        if cmd.kind == "action":
            elapsed = (datetime.now(timezone.utc) - session.started_at).total_seconds()
            session.action_items.append({"by": asker, "text": cmd.arg, "t": elapsed})
            session.save_meta("recording")
            speak("Logged.")
            return await reply(f"Logged action item: **{cmd.arg}**. It'll be in the notes.")
        if cmd.kind == "stop":
            self.coord.paused.add((session.guild_id, session.channel_id))
            session.manual = True
            await reply("Stopped recording. Notes are on the way to #meeting-minutes. "
                        "I'll stay out of this channel until everyone leaves it.")
            if say:
                await say("Stopping. Notes are on the way.")  # finish talking before we hang up
            await owner.end_session(session.guild_id, reason="stopped by command")
            return
        if cmd.kind == "notes":
            ch = guild.get_channel(session.channel_id)
            session.manual = True
            await reply("Posting notes for the meeting so far to #meeting-minutes. Still recording.")
            if say:
                await say("Posting notes now. I'll keep recording.")
            await owner.end_session(session.guild_id, reason="notes requested")
            if ch is not None and _humans(ch):
                await owner.start_session(ch)

    async def _embed_loop(self) -> None:
        """Keep meaning based search up to date: embed new chunks in the background."""
        emb = embedder_for(self.cfg)
        if emb is None:
            return
        while True:
            try:
                n = await embed_pending(self.coord.kb, emb)
                if n:
                    log.info("embedded %d new chunks for meaning based search", n)
            except Exception:
                log.exception("embedding loop failed")
            await asyncio.sleep(30)

    def _home_guild_id(self) -> Optional[int]:
        """Guild that files shared directly with Rocky's Drive account belong to."""
        if self.cfg.guild_ids:
            return sorted(self.cfg.guild_ids)[0]
        return self.guilds[0].id if self.guilds else None

    async def _on_new_folder(self, folder: dict, message: discord.Message) -> None:
        """Someone posted a Drive folder link: read it and say how it went, in that channel."""
        drive = self.coord.drive
        if not drive.auth.mode:
            return await self._send(message.channel, "I see a Google Drive folder link, but Drive access isn't set up "
                                                     "for me yet, so I can only use files posted individually.")
        status = await drive.crawl(folder)
        info = next((f for f in drive.folders(folder["guild_id"]) if f["folder_id"] == folder["folder_id"]), folder)
        if status == "ok":
            await self._send(message.channel, f"Read the Drive folder **{info.get('name')}** and its subfolders: "
                                              f"{info.get('files', 0)} files are now searchable. I re-check it every "
                                              f"{self.cfg.doc_refresh_hours} hours. Read only, I can't edit anything.")
        elif status == "no_access":
            who = drive.auth.account_email()
            await self._send(message.channel, "I can't open that Drive folder. Share it with "
                                              f"`{who}` as **Viewer**, then post the link again." if who else
                                              "I can't open that Drive folder. Make sure it's shared with me, then post it again.")

    async def _send(self, channel, text: str) -> None:
        for chunk in chunk_text(text, 1900):
            try:
                await channel.send(chunk, allowed_mentions=discord.AllowedMentions.none(), suppress_embeds=True)
            except Exception:
                log.warning("could not send to #%s", getattr(channel, "name", "?"), exc_info=True)
                return

    def _session_for(self, member) -> tuple[Optional["MinutesBot"], Optional[Session]]:
        vc = getattr(getattr(member, "voice", None), "channel", None)
        if vc is None:
            return None, None
        for b in self.coord.bots:
            s = b.sessions.get(vc.guild.id)
            if s and s.channel_id == vc.id:
                return b, s
        return None, None

    def _addressed(self, message: discord.Message) -> Optional[str]:
        """Command text if the message @mentions Rocky (user or its role), else None."""
        me = message.guild.me if message.guild else None
        hit = self.user in message.mentions
        my_roles = {r.id for r in getattr(me, "roles", []) if r.managed} if me else set()
        if not hit and any(r.id in my_roles for r in message.role_mentions):
            hit = True
        if not hit:
            return None
        text = re.sub(rf"<@!?{self.user.id}>", "", message.content)
        for rid in my_roles:
            text = text.replace(f"<@&{rid}>", "")
        return text.strip(" ,:")

    async def on_message(self, message: discord.Message) -> None:
        if not self.primary or not self.cfg.text_commands or message.guild is None:
            return
        text = None if message.author.bot else self._addressed(message)
        if text is None:
            # questions to Rocky are not knowledge; only index everything else
            try:
                self.coord.indexer.index_message(message, self.settings.get_channel_id(message.guild.id))
            except Exception:
                log.exception("indexing failed")
            return
        owner, session = self._session_for(message.author)
        asker = message.author.display_name

        async def reply(msg: str) -> None:
            chunks_ = chunk_text(msg, 1900)
            try:
                await message.reply(chunks_[0], mention_author=False, suppress_embeds=True,
                                    allowed_mentions=discord.AllowedMentions.none())
            except Exception:
                await self._send(message.channel, chunks_[0])
            for c in chunks_[1:]:
                await self._send(message.channel, c)

        async with message.channel.typing():
            await self.run_command(parse_command(text), message.guild, reply, asker, owner, session)

    async def on_message_edit(self, before: discord.Message, after: discord.Message) -> None:
        if self.primary and self.cfg.text_commands and after.guild is not None and before.content != after.content:
            self.coord.indexer.index_message(after, self.settings.get_channel_id(after.guild.id))

    async def on_raw_message_delete(self, payload: discord.RawMessageDeleteEvent) -> None:
        if self.primary:
            self.coord.kb.delete(f"msg:{payload.message_id}")

    # slash commands -------------------------------------------------------
    def _register_commands(self) -> None:
        group = app_commands.Group(name="minutes", description="Meeting minutes bot",
                                   default_permissions=discord.Permissions(manage_guild=True))

        @group.command(name="start", description="Start recording a voice channel now")
        @app_commands.describe(channel="Voice channel (defaults to the one you are in)")
        async def start(inter: discord.Interaction, channel: Optional[discord.VoiceChannel] = None):
            await inter.response.defer(ephemeral=True)
            ch = channel or (inter.user.voice.channel if isinstance(inter.user, discord.Member) and inter.user.voice else None)
            if ch is None:
                return await inter.followup.send("Join a voice channel or pass one.", ephemeral=True)
            if (inter.guild_id, ch.id) in self.coord.claims:
                return await inter.followup.send(f"Already recording **{ch.name}**.", ephemeral=True)
            worker = self.coord.free_bot(inter.guild_id)
            if worker is None:
                busy = ", ".join(b.sessions[inter.guild_id].channel_name for b in self.coord.bots if inter.guild_id in b.sessions)
                return await inter.followup.send(f"All bot accounts are busy ({busy}). Add another token to record more channels at once.", ephemeral=True)
            s = await worker.start_session(ch, manual=True)
            await inter.followup.send(f"Recording **{ch.name}**." if s else "Could not join that channel (check permissions).", ephemeral=True)

        @group.command(name="stop", description="Stop the current recording and produce notes")
        @app_commands.describe(now="Skip the publish delay and post as soon as transcription finishes")
        @app_commands.describe(channel="Which recording to stop (defaults to the channel you are in, or the only one running)")
        async def stop(inter: discord.Interaction, now: bool = False, channel: Optional[discord.VoiceChannel] = None):
            await inter.response.defer(ephemeral=True)
            active = [(b, b.sessions[inter.guild_id]) for b in self.coord.bots if inter.guild_id in b.sessions]
            if not active:
                return await inter.followup.send("Not recording.", ephemeral=True)
            target = channel.id if channel else (
                inter.user.voice.channel.id if isinstance(inter.user, discord.Member) and inter.user.voice else None)
            pick = [(b, s) for b, s in active if target is None or s.channel_id == target]
            if len(pick) != 1:
                names = ", ".join(s.channel_name for _, s in active)
                return await inter.followup.send(f"Recording in: {names}. Pass the channel to stop.", ephemeral=True)
            worker, s = pick[0]
            s.manual = True
            if now:
                await worker.end_session(inter.guild_id, reason="manual stop", process=False)
                worker._spawn(worker.process_session_dir(s.dir))
            else:
                await worker.end_session(inter.guild_id, reason="manual stop")
            await inter.followup.send(f"Stopped. Notes for **{s.channel_name}** are being prepared.", ephemeral=True)

        @group.command(name="status", description="Show what the bot is doing")
        async def status(inter: discord.Interaction):
            notes = self.settings.get_channel_id(inter.guild_id)
            lines = []
            for b in self.coord.bots:
                s = b.sessions.get(inter.guild_id)
                if s:
                    spk = ", ".join(s.sink.names.values()) if s.sink and s.sink.names else "none yet"
                    lines.append(f"Recording **{s.channel_name}** for {int(s.duration // 60)} min. Speakers heard: {spk}.")
            txt = "\n".join(lines) or "Idle. Will auto join when someone enters a voice channel."
            txt += f"\nBot accounts: {len(self.coord.bots)} (max simultaneous meetings)"
            txt += f"\nNotes channel: {f'<#{notes}>' if notes else self.cfg.minutes_channel_name + ' (will be created)'}"
            await inter.response.send_message(txt, ephemeral=True)

        @group.command(name="channel", description="Set the channel where notes are posted")
        async def channel(inter: discord.Interaction, channel: discord.TextChannel):
            self.settings.set_channel_id(inter.guild_id, channel.id)
            await inter.response.send_message(f"Notes will be posted in {channel.mention}.", ephemeral=True)

        self.tree.add_command(group)

        @self.tree.command(name="ask", description="Ask Rocky: find a doc or answer from channels, shared files and past meetings")
        @app_commands.describe(question="e.g. 'pull up the computer systems doc' or 'what did we decide about the launch date?'")
        async def ask(inter: discord.Interaction, question: str):
            await inter.response.defer(thinking=True)
            owner, session = self._session_for(inter.user)

            async def reply(msg: str) -> None:
                chunks_ = chunk_text(f"> {question}\n{msg}", 1900)
                for c in chunks_:
                    await inter.followup.send(c, suppress_embeds=True, allowed_mentions=discord.AllowedMentions.none())

            await self.run_command(parse_command(question), inter.guild, reply,
                                   inter.user.display_name, owner, session)


def main() -> None:
    cfg = Config()
    cfg.validate()
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("discord").setLevel(logging.WARNING)
    asyncio.run(_run_all(cfg))


async def _run_all(cfg: Config) -> None:
    coord = Coordinator(cfg)
    bots = [MinutesBot(cfg, coord, i) for i in range(len(cfg.all_tokens))]
    log.info("starting %d bot account(s)", len(bots))
    tasks = [asyncio.create_task(b.start(tok)) for b, tok in zip(bots, cfg.all_tokens)]

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass

    waiter = asyncio.create_task(stop.wait())
    done, _ = await asyncio.wait([*tasks, waiter], return_when=asyncio.FIRST_COMPLETED)
    for t in done:
        if t is not waiter and t.exception():
            log.error("bot account crashed: %r", t.exception())
    log.info("shutting down")
    await asyncio.gather(*(b.close() for b in bots), return_exceptions=True)
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    coord.executor.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main()
