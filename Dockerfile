FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/data \
    HF_HOME=/models

# libopus: Discord audio decode. git: pip install from GitHub. ffmpeg: not required, PyAV is bundled.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libopus0 git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Pre download the Whisper model so first meeting is not slow. Override at build time:
#   docker build --build-arg WHISPER_MODEL=medium .
ARG WHISPER_MODEL=small
ARG WHISPER_LIVE_MODEL=base.en
RUN python -c "from faster_whisper import WhisperModel; [WhisperModel(m, device='cpu', compute_type='int8') for m in ('${WHISPER_MODEL}', '${WHISPER_LIVE_MODEL}')]"

# Rocky's speaking voice (local Piper text to speech)
ARG TTS_VOICE=en_US-ryan-high
RUN python -m piper.download_voices --download-dir /models/piper ${TTS_VOICE}

# code last, so code changes don't re-download the models
COPY bot ./bot

RUN useradd -r -u 10001 bot && mkdir -p /data && chown -R bot:bot /data /models
USER bot
VOLUME ["/data"]

CMD ["python", "-m", "bot.main"]
