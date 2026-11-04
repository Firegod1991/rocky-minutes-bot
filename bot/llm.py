"""Language model calls: Claude, Gemini or local Ollama with automatic fallback, plus web research."""
from __future__ import annotations

import asyncio
import logging

import aiohttp

from .config import Config

log = logging.getLogger(__name__)


def _short(body: str) -> str:
    """First line of an API error message, for readable logs."""
    import json

    try:
        data = json.loads(body)
        err = (data[0] if isinstance(data, list) else data).get("error", {})
        return str(err.get("message", body)).split(". ")[0][:120]
    except Exception:
        return " ".join(body.split())[:120]


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


_main_off_until = 0.0


async def complete(cfg: Config, system: str, user: str, max_tokens: int = 1500, fast: bool = False) -> str:
    """Ask the configured cloud model; if it fails, fall back to local Ollama when available."""
    global _main_off_until
    import time

    p = provider(cfg)
    # main Gemini model recently out of quota: go straight to the lite model instead of waiting on retries
    if p == "gemini" and not fast and time.time() < _main_off_until and cfg.gemini_fast_model:
        fast = True
    try:
        return await _complete(cfg, p, system, user, max_tokens, fast)
    except Exception as e:
        if p == "gemini" and not fast and "429" in str(e):
            _main_off_until = time.time() + 600
        # main Gemini model busy or out of free quota: the lite model has its own quota, try it first
        if p == "gemini" and not fast and cfg.gemini_fast_model and cfg.gemini_fast_model != cfg.gemini_model:
            log.warning("gemini %s failed (%s); trying %s", cfg.gemini_model, str(e)[:60], cfg.gemini_fast_model)
            try:
                return await _complete(cfg, p, system, user, max_tokens, True)
            except Exception:
                pass
        if p != "ollama" and cfg.ollama_url and cfg.ollama_model:
            log.warning("%s failed; falling back to local Ollama", p)
            return await _complete(cfg, "ollama", system, user, max_tokens, fast)
        raise


async def _complete(cfg: Config, p: str, system: str, user: str, max_tokens: int, fast: bool = False) -> str:
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
            # fast = spoken replies: smaller model, no thinking pass
            "model": cfg.gemini_fast_model if fast else cfg.gemini_model,
            # thinking tokens count against max_tokens on Gemini 3.x: keep thinking light and leave headroom
            "reasoning_effort": "low",  # "none" is rejected by the lite models
            "max_tokens": max_tokens + 2048,
            "temperature": 0.2,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        headers = {"Authorization": f"Bearer {cfg.gemini_api_key}"}
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as s:
            # spoken replies can't wait on backoff: one try, then the local fallback
            for wait in ((0,) if fast else (2, 5, 0)):
                async with s.post(url, json=body, headers=headers) as r:
                    if r.status == 200:
                        data = await r.json()
                        break
                    err = f"gemini HTTP {r.status}: {_short(await r.text())}"
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


def can_research(cfg: Config) -> bool:
    return bool(cfg.web_research and enabled(cfg))


_grounding_off_until = 0.0  # Google Search grounding needs billing; skip it for a while after a quota error


async def research(cfg: Config, system: str, question: str, max_tokens: int = 700, fast: bool = False) -> str:
    """Answer from the web. Order: Gemini with Google Search, Claude web search, then a search API
    (Brave or Tavily key, else free Wikipedia) whose results any configured model summarizes."""
    global _grounding_off_until
    import time

    if cfg.gemini_api_key and time.time() > _grounding_off_until:
        models = [cfg.gemini_fast_model] if fast else [cfg.gemini_model, cfg.gemini_fast_model]
        for model in dict.fromkeys(m for m in models if m):
            try:
                return await _gemini_search(cfg, model, system, question, max_tokens)
            except Exception as e:
                log.warning("google search grounding with %s failed: %s", model, str(e)[:100])
                if "429" in str(e):
                    _grounding_off_until = time.time() + 3600
    if cfg.anthropic_api_key:
        try:
            from anthropic import AsyncAnthropic

            client = AsyncAnthropic(api_key=cfg.anthropic_api_key, max_retries=1, timeout=120)
            resp = await client.messages.create(
                model=cfg.claude_model, max_tokens=max_tokens, system=system,
                tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}],
                messages=[{"role": "user", "content": question}],
            )
            text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text").strip()
            if text:
                return text
        except Exception as e:
            log.warning("claude web search failed: %s", str(e)[:100])
    results = await web_search(cfg, question)
    if not results:
        return "I couldn't find that online."
    src = "\n\n".join(f"[{i}] {t}\n{body[:5000]}" for i, (t, _, body) in enumerate(results, 1))
    return await complete(cfg, system + " Use only these search results; they are current.",
                          f"Question: {question}\n\n<search_results>\n{src}\n</search_results>", max_tokens, fast)


async def web_search(cfg: Config, query: str, n: int = 5) -> list[tuple[str, str, str]]:
    """(title, url, text) results from Brave or Tavily when a key is set, else Wikipedia (free, no key)."""
    timeout = aiohttp.ClientTimeout(total=20)
    headers = {"User-Agent": "RockyBot/1.0 (https://github.com/Firegod1991/rocky-discord-bot)"}
    async with aiohttp.ClientSession(timeout=timeout, headers=headers) as s:
        try:
            if cfg.brave_api_key:
                async with s.get("https://api.search.brave.com/res/v1/web/search",
                                 params={"q": query, "count": n, "extra_snippets": "true"},
                                 headers={"X-Subscription-Token": cfg.brave_api_key, "Accept": "application/json"}) as r:
                    r.raise_for_status()
                    data = await r.json()
                return [(w.get("title", ""), w.get("url", ""),
                         " ".join([w.get("description", "")] + w.get("extra_snippets", [])))
                        for w in (data.get("web") or {}).get("results", [])[:n]]
            if cfg.tavily_api_key:
                async with s.post("https://api.tavily.com/search",
                                  json={"query": query, "max_results": n, "include_answer": True},
                                  headers={"Authorization": f"Bearer {cfg.tavily_api_key}"}) as r:
                    r.raise_for_status()
                    data = await r.json()
                out = [("Summary", "", data["answer"])] if data.get("answer") else []
                return out + [(x.get("title", ""), x.get("url", ""), x.get("content", "")) for x in data.get("results", [])[:n]]
        except Exception as e:
            log.warning("search api failed, using Wikipedia: %s", str(e)[:100])
        return await _wikipedia(s, query, n)


def _focus(text: str, query: str, limit: int = 5000) -> str:
    """Intro plus the paragraphs that share the most words with the question (long articles bury facts)."""
    import re

    paras = [p.strip() for p in re.split(r"\n+", text) if len(p.strip()) > 40]
    if sum(map(len, paras)) <= limit:
        return "\n".join(paras)
    words = {w for w in re.findall(r"[a-z0-9]{3,}", query.lower())} - {"the", "what", "current", "how", "much", "many", "does", "who", "when"}
    syn = {"mass": "weigh weighs weighing weight kg kilograms", "weight": "weigh weighing mass kg kilograms",
           "heavy": "weigh weighing mass kg", "height": "tall high elevation km", "tall": "height high km",
           "cost": "price usd million billion", "price": "cost usd", "speed": "mph km/h fast", "size": "diameter wide km"}
    for w in list(words):
        words |= set(syn.get(w, "").split())
    scored = sorted(range(1, len(paras)), key=lambda i: -sum(w in paras[i].lower() for w in words))
    keep, used = {0}, len(paras[0])
    for i in scored:
        if used + len(paras[i]) > limit:
            continue
        keep.add(i)
        used += len(paras[i])
    return "\n".join(paras[i] for i in sorted(keep))


async def _wikipedia(s, query: str, n: int) -> list[tuple[str, str, str]]:
    import re

    q = re.sub(r"^(?:what(?:'s| is| are| was)|who(?:'s| is| was)|how (?:much|many|tall|big|heavy|far)|when (?:is|was|did))\s+", "",
               query.strip().rstrip("?"), flags=re.I)
    api = "https://en.wikipedia.org/w/api.php"
    try:
        async with s.get(api, params={"action": "query", "list": "search", "srsearch": q, "srlimit": 3,
                                      "format": "json"}) as r:
            titles = [x["title"] for x in (await r.json()).get("query", {}).get("search", [])]
        if not titles:
            return []
        async def extract(t: str) -> tuple[str, str, str]:
            async with s.get(api, params={"action": "query", "prop": "extracts", "explaintext": 1,
                                          "titles": t, "redirects": 1, "format": "json"}) as r:
                pages = list((await r.json()).get("query", {}).get("pages", {}).values())
            text = _focus((pages[0].get("extract", "") if pages else "").strip(), query)
            return t, "https://en.wikipedia.org/wiki/" + t.replace(" ", "_"), text

        found = await asyncio.gather(*(extract(t) for t in titles[:2]), return_exceptions=True)
        return [x for x in found if isinstance(x, tuple) and x[2]][:n]
    except Exception as e:
        log.warning("wikipedia search failed: %s", str(e)[:100])
        return []


async def _gemini_search(cfg: Config, model: str, system: str, question: str, max_tokens: int) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": question}]}],
        "tools": [{"google_search": {}}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": max_tokens + 2048,
                             "thinkingConfig": {"thinkingLevel": "low"}},
    }
    headers = {"x-goog-api-key": cfg.gemini_api_key}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as s:
        async with s.post(url, json=body, headers=headers) as r:
            if r.status != 200:
                txt = await r.text()
                if r.status == 400 and "thinking" in txt.lower():
                    body["generationConfig"].pop("thinkingConfig", None)  # model without thinking levels
                    async with s.post(url, json=body, headers=headers) as r2:
                        if r2.status != 200:
                            raise RuntimeError(f"gemini HTTP {r2.status}: {_short(await r2.text())}")
                        data = await r2.json()
                else:
                    raise RuntimeError(f"gemini HTTP {r.status}: {_short(txt)}")
            else:
                data = await r.json()
    parts = ((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
    if not text:
        raise RuntimeError("empty answer")
    return text
