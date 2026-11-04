# Rocky: Discord Meeting Minutes Bot

Joins a voice channel when people show up, records each speaker on their own track, transcribes locally with Whisper, asks Claude for minutes, and posts everything to a dedicated `#meeting-minutes` channel a couple of minutes after the meeting ends. Transcript and minutes are attached as `.txt` files, organized by username and timestamp, and the transcript is also posted in a thread under the minutes.

## What it does

1. Someone joins a voice channel. The bot joins too and starts recording (configurable minimum participant count, allow/deny lists, AFK channel ignored).
2. Each speaker is recorded separately and aligned on one timeline, so the transcript reads `[00:04:12] fire: ...` / `[00:04:20] tyler: ...` in the real order.
3. When the channel empties (30 s grace by default) the bot leaves, waits `PUBLISH_DELAY_SECONDS` (2 min default), transcribes with faster whisper on your own CPU/GPU, and sends the transcript to Claude for Summary, Key Points, Decisions, Action Items, Open Questions.
4. The result lands in `#meeting-minutes` (created automatically, read only for members) with `..._minutes.txt` and `..._transcript.txt` attached, plus the full transcript in a thread under the post. Talk time per person is included.
5. Meetings shorter than `MIN_MEETING_SECONDS` are discarded. Sessions interrupted by a restart are picked up and processed on the next boot.

Covering every voice channel: Discord lets one bot account be in one voice channel per server at a time. For overlapping meetings, create a second bot application, invite it with the same permissions, and put its token in `DISCORD_EXTRA_TOKENS`. The accounts run in one process and split channels between themselves automatically.

Slash commands (Manage Server permission): `/minutes start`, `/minutes stop [now]`, `/minutes status`, `/minutes channel #target`.

## Discord setup

1. https://discord.com/developers/applications, New Application, Bot.
2. Bot tab: copy the token. Under Privileged Gateway Intents enable **Server Members Intent**.
3. OAuth2 URL Generator: scopes `bot` + `applications.commands`. Permissions: View Channels, Connect, Send Messages, Attach Files, Create Public Threads, Send Messages in Threads, Manage Channels (permissions integer 309238811664).
4. Open the generated URL and add the bot to your server.
5. Put your server ID in `GUILD_IDS` so slash commands show up instantly.

## Run with Docker

```bash
cp .env.example .env     # fill in DISCORD_TOKEN and ANTHROPIC_API_KEY
docker compose up -d --build
docker compose logs -f
```

The image downloads the Whisper model at build time (`WHISPER_MODEL`, default `small`). Recordings, transcripts and minutes persist in `./data/sessions/<timestamp>-<guild>-<channel>/`.

On Linux hosts the container runs as uid 10001, so `mkdir -p data && sudo chown 10001 data` first. Docker Desktop on Windows and macOS handles this for you.

## Run without Docker

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env && set -a && . ./.env && set +a
python -m bot.main
```

Needs libopus on the host (`apt install libopus0`; on Windows discord.py bundles it).

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Why discord.py fork + voice recv instead of py-cord

Discord now end to end encrypts voice (DAVE). py-cord 2.8.1 ships `start_recording` but its receive pipeline is broken on release. The `zacker150/discord-ext-voice-recv` fork is the maintained Python path that decrypts DAVE audio, so `requirements.txt` pins it to a tested commit along with the discord.py build it needs.

## Privacy and consent

Recording calls has legal implications. The bot announces in the notes channel when a recording starts. Put a notice in your server rules. Audio is deleted after processing unless `KEEP_AUDIO=true`. Transcript text is sent to Anthropic only if `ANTHROPIC_API_KEY` is set; Whisper runs locally.
