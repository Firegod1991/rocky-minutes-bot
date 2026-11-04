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

from .config import Config
from .merge import Segment, coalesce, render_transcript
from .minutes import generate_minutes
from .publisher import Settings, get_minutes_channel, post_minutes, post_notice, post_transcript_thread
from .recorder import Session, TimelineSink
from .transcribe import transcribe_session_dir

log = logging.getLogger("minutes")


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower() or "voice"


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
        self._spawn(self._resume_unprocessed())

    async def on_ready(self) -> None:
        log.info("[bot %d] logged in as %s (%s); %d guilds", self.index, self.user, self.user.id, len(self.guilds))
        # catch meetings already in progress when the bot starts
        for guild in self.guilds:
            if guild.id in self.sessions:
                continue
            for vc in guild.voice_channels + guild.stage_channels:
                if (self._eligible(guild, vc) and len(_humans(vc)) >= self.cfg.min_participants
                        and (guild.id, vc.id) not in self.coord.claims):
                    self._spawn(self.start_session(vc))
                    break

    async def close(self) -> None:
        log.info("[bot %d] shutting down; finalizing %d active session(s)", self.index, len(self.sessions))
        for gid in list(self.sessions):
            try:
                await self.end_session(gid, reason="bot shutdown", process=False)
            except Exception:
                log.exception("finalizing session for guild %s", gid)
        await super().close()

    async def _minutes_channel(self, guild: discord.Guild):
        async with self.coord.channel_lock:
            return await get_minutes_channel(self.cfg, self.settings, guild)

    def _spawn(self, coro) -> asyncio.Task:
        t = asyncio.create_task(coro)
        self._bg.add(t)
        t.add_done_callback(self._bg.discard)
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
        if member.id == self.user.id:
            # we got kicked, moved, or disconnected
            if after.channel is None and guild.id in self.sessions:
                log.warning("bot removed from voice in %s", guild.name)
                await self.end_session(guild.id, reason="bot disconnected")
            return
        if member.bot or before.channel == after.channel:
            return

        session = self.sessions.get(guild.id)

        # someone joined a channel
        if after.channel is not None and self._eligible(guild, after.channel):
            if session is None:
                if (len(_humans(after.channel)) >= self.cfg.min_participants
                        and (guild.id, after.channel.id) not in self.coord.claims):
                    await self.start_session(after.channel)
            elif session.channel_id == after.channel.id:
                session.participants.setdefault(member.id, member.display_name)
                self._cancel_leave(session)

        # someone left the channel we are recording
        if session and before.channel is not None and before.channel.id == session.channel_id:
            if not _humans(before.channel):
                self._schedule_leave(session)

    def _schedule_leave(self, session: Session) -> None:
        self._cancel_leave(session)

        async def _wait_then_leave():
            await asyncio.sleep(self.cfg.leave_grace_seconds)
            guild = self.get_guild(session.guild_id)
            ch = guild.get_channel(session.channel_id) if guild else None
            if ch is None or not _humans(ch):
                await self.end_session(session.guild_id, reason="channel empty")

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
            vc.listen(sink)
            session.save_meta("recording")
            log.info("recording %s / #%s -> %s", guild.name, channel.name, sdir.name)
            if self.cfg.announce_recording:
                notes = await self._minutes_channel(guild)
                await post_notice(notes, f"Recording started in **{channel.name}**. Notes will be posted here about "
                                         f"{self.cfg.publish_delay_seconds // 60} minutes after the meeting ends.")
            self._spawn(self._max_duration_guard(session))
            return session
        except Exception:
            log.exception("could not start recording in %s/#%s", guild.name, channel.name)
            self.sessions.pop(guild.id, None)
            self.coord.release(guild.id, channel.id, self)
            vc = guild.voice_client
            if vc:
                await vc.disconnect(force=True)
            return None

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
