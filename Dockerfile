FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/data \
    HF_HOME=/models

# libopus: Discord audio decode. git: pip install from GitHub.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libopus0 git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY bot ./bot

# Pre download the Whisper model so the first meeting is not slow. Override at build time:
#   docker build --build-arg WHISPER_MODEL=medium .
ARG WHISPER_MODEL=small
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('${WHISPER_MODEL}', device='cpu', compute_type='int8')"

RUN useradd -r -u 10001 bot && mkdir -p /data && chown -R bot:bot /data /models
USER bot
VOLUME ["/data"]

CMD ["python", "-m", "bot.main"]
