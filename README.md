# Rocky: a Discord meeting assistant

Rocky sits in your Discord server, records voice meetings, writes the minutes, and answers questions about your team's work out loud or in chat. It reads the Google Docs, Sheets, Slides and Drive folders your team posts and answers from the latest data, with no file paths or citations in the way.

Transcription and voice run on your own machine. Only the text of a question (or a finished transcript) goes to the language model you pick, and you can run that locally too with Ollama.

## What it does

**Meeting minutes**
* Joins a voice channel when people show up and records each speaker on their own track
* Transcribes locally with faster whisper, in the real order of who said what
* Posts a summary, key points, decisions, action items and open questions to `#meeting-minutes` a couple of minutes after the call ends, with the full transcript as a `.txt` file and in a thread
* Several bot accounts can run at once, so overlapping meetings in different channels are all covered

**Voice and text assistant**
* Say "Rocky, what's the weight of the rover?" in a call and it answers out loud in a natural voice (Kokoro TTS)
* Or type `@Rocky what's the weight of the rover?` or `/ask` in any channel
* `Rocky, action item: Sam sends the budget by Friday` adds it to this meeting's notes
* `Rocky, take notes now` posts notes so far; `Rocky, stop recording` ends the meeting
* `@Rocky docs` lists every file it can read; `@Rocky pull up the risk log` posts the link
* `Rocky, research how much the Perseverance rover weighs` looks it up on the web

**Knowledge it uses**
* Every message in every channel, thread and forum post it can see
* Every Google Doc, Sheet (all tabs), Slides deck and Drive file (PDF, Word, Excel, PowerPoint) linked in any channel
* Whole Drive folders: post a folder link once and it crawls every file inside, including subfolders
* Files attached in Discord (PDF, Word, Excel, PowerPoint, txt, md, csv)
* Past meeting transcripts and minutes
* Keyword search plus meaning based search, so "how heavy is it" finds a sheet column called "Total mass (kg)"
* When sources disagree it uses the most recent value

Rocky only ever reads Google files. It never edits them.

**Web research**
* Say or type `research ...`, `google ...`, `search the web for ...` or `look up ... online`
* When your own docs and chats don't have the answer, Rocky looks it up online instead (turn off with `WEB_FALLBACK=false`)
* Search source, best first: Gemini with Google Search (needs billing on your AI Studio project), Claude web search, Brave or Tavily (free keys), then Wikipedia (free, no key)

## Quick start (Docker)

```bash
git clone https://github.com/Firegod1991/rocky-discord-bot.git
cd rocky-discord-bot
cp .env.example .env        # then fill in your own values
mkdir -p data
docker compose up -d --build
docker compose logs -f
```

On Linux, run `sudo chown 10001 data` once (the container runs as uid 10001). The first build downloads the Whisper and voice models, so give it a few minutes.

## Discord setup

1. Go to https://discord.com/developers/applications, create a New Application, then open the Bot tab.
2. Reset and copy the token into `DISCORD_TOKEN` in your `.env`. Treat it like a password.
3. Under Privileged Gateway Intents turn on **Server Members Intent** and **Message Content Intent**.
4. OAuth2, URL Generator: scopes `bot` and `applications.commands`. Permissions: View Channels, Read Message History, Send Messages, Attach Files, Create Public Threads, Send Messages in Threads, Connect, Speak, Manage Channels (only needed to create `#meeting-minutes`).
5. Open the generated link and add the bot to your server.
6. Right click your server, Copy Server ID, and put it in `GUILD_IDS`.

**More than one meeting at a time:** Discord allows one voice channel per bot account per server. Create a second application the same way, invite it, and add its token to `DISCORD_EXTRA_TOKENS` (comma separated). The accounts split channels automatically.

**Bot avatar:** drop your own image at `bot/assets/rocky-avatar.png` and each account uses it if it has no avatar yet.

## Language model

Set at least one of these in `.env`:

| Option | Setting | Notes |
|---|---|---|
| Google Gemini | `GEMINI_API_KEY` | Free tier at https://aistudio.google.com/apikey. Falls back to the lite model when rate limited |
| Anthropic Claude | `ANTHROPIC_API_KEY` | Optional |
| Ollama (local, free) | `OLLAMA_URL`, `OLLAMA_MODEL` | Runs on your machine. Also used for embeddings: `ollama pull nomic-embed-text` |

Rocky tries them in that order and falls back automatically.

## Google Docs and Drive

* **Linked files:** share each file as "Anyone with the link can view" and post the link in any channel. No key needed.
* **Public folders:** post the folder link. No key needed. Rocky reads everything inside and re reads every `DOC_REFRESH_HOURS`.
* **Private folders:** create a Google Cloud service account, save its JSON key as `data/google-service-account.json`, and share the folder with the service account email as Viewer.

## Running without Docker

```bash
sudo apt install libopus0 git espeak-ng
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env && set -a && . ./.env && set +a
python -m bot.main
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Keeping your secrets safe

* All keys and tokens live only in `.env`, which is in `.gitignore`. Never commit it, paste it in an issue, or put it in a screenshot.
* `data/` holds recordings, transcripts and the knowledge base and is ignored too.
* If a token ever leaks, reset it right away in the Discord developer portal or Google AI Studio.

## Privacy and consent

Recording calls has legal rules (two party consent states, GDPR, your server's own rules). Rocky announces in the notes channel when a recording starts. Tell your members in the server rules and channel topics. Audio is deleted after processing unless `KEEP_AUDIO=true`.

## Why the discord.py voice fork

Discord now end to end encrypts voice (DAVE). The `discord-ext-voice-recv` fork pinned in `requirements.txt` is the maintained Python path that can receive and decrypt it.

## Layout

```
bot/
  main.py        Discord clients, voice events, commands, multi account coordinator
  recorder.py    per speaker, timeline aligned audio recording
  transcribe.py  voice activity chunking and faster whisper
  live.py        live listening for "Rocky, ..." voice commands
  speech.py      text to speech (Kokoro or Piper) played into the call
  assistant.py   command parsing, search, answers
  knowledge.py   SQLite search index, Google file reading, embeddings
  indexer.py     reads channel history, threads and linked files
  drive.py       Google Drive folder crawler (public or service account)
  minutes.py     minutes generation
  llm.py         Gemini, Claude and Ollama with automatic fallback, web research
  publisher.py   notes channel and posting
  config.py      settings from .env
tests/
```

## License

MIT. See `LICENSE`.
