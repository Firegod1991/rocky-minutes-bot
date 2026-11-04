"""One call for "ask a language model": Claude if a key is set, otherwise local Ollama."""
from __future__ import annotations

import asyncio
import logging

import aiohttp

from .config import Config

log = logging.getLogger(__name__)


def provider(cfg: Config) -> str:
    if cfg.anthropic_api_key:
        return "claude"
    if cfg.gemini_api_key:
        return "gemini"
    if cfg.ollama_url and cfg.ollama_model:
        return "ollama"
    return ""


def enabled(cfg: Config) -> bool:
    return bool(provider(cfg))


async def complete(cfg: Config, system: str, user: str, max_tokens: int = 1500) -> str:
    """Ask the configured cloud model; if it fails, fall back to local Ollama when available."""
    p = provider(cfg)
    try:
        return await _complete(cfg, p, system, user, max_tokens)
    except Exception:
        if p != "ollama" and cfg.ollama_url and cfg.ollama_model:
            log.exception("%s failed; falling back to local Ollama", p)
            return await _complete(cfg, "ollama", system, user, max_tokens)
        raise


async def _complete(cfg: Config, p: str, system: str, user: str, max_tokens: int) -> str:
    if p == "claude":
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(api_key=cfg.anthropic_api_key, max_retries=2, timeout=180)
        resp = await client.messages.create(
            model=cfg.claude_model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(getattr(b, "text", "") for b in resp.content).strip()
    if p == "gemini":
        # Gemini's OpenAI compatible endpoint
        url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        body = {
            "model": cfg.gemini_model,
            # thinking tokens count against max_tokens on Gemini 3.x: keep thinking light and leave headroom
            "reasoning_effort": "low",
            "max_tokens": max_tokens + 2048,
            "temperature": 0.2,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        headers = {"Authorization": f"Bearer {cfg.gemini_api_key}"}
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as s:
            for attempt, wait in enumerate((2, 5, 0)):
                async with s.post(url, json=body, headers=headers) as r:
                    if r.status == 200:
                        data = await r.json()
                        break
                    err = f"gemini HTTP {r.status}: {(await r.text())[:300]}"
                # busy or rate limited: brief backoff before giving up to the fallback
                if r.status not in (429, 500, 503) or not wait:
                    raise RuntimeError(err)
                await asyncio.sleep(wait)
        return ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "").strip()
    if p == "ollama":
        body = {
            "model": cfg.ollama_model,
            "stream": False,
            "think": False,  # qwen3 and similar: answer directly, no hidden reasoning pass
            "keep_alive": "30m",
            "options": {"temperature": 0.2, "num_ctx": cfg.ollama_ctx, "num_predict": max_tokens},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        timeout = aiohttp.ClientTimeout(total=600)
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.post(cfg.ollama_url.rstrip("/") + "/api/chat", json=body) as r:
                if r.status != 200:
                    raise RuntimeError(f"ollama HTTP {r.status}: {(await r.text())[:200]}")
                data = await r.json()
        text = (data.get("message") or {}).get("content", "")
        # strip any <think> block a reasoning model might still emit
        if "</think>" in text:
            text = text.split("</think>", 1)[1]
        return text.strip()
    raise RuntimeError("no language model configured")
