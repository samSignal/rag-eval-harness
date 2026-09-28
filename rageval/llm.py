"""Minimal LLM clients returning (text, prompt_tokens, completion_tokens)."""
from __future__ import annotations

import os

import requests


class OpenAIClient:
    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None, base_url: str | None = None):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"), base_url=base_url)
        self.model, self.name = model, f"openai:{model}"

    def __call__(self, prompt: str) -> tuple[str, int, int]:
        r = self.client.chat.completions.create(model=self.model, temperature=0,
                                                messages=[{"role": "user", "content": prompt}])
        u = r.usage
        return r.choices[0].message.content or "", (u.prompt_tokens if u else 0), (u.completion_tokens if u else 0)


class OllamaClient:
    def __init__(self, model: str = "llama3.1", url: str = "http://localhost:11434"):
        self.url, self.model, self.name = url.rstrip("/"), model, f"ollama:{model}"

    def __call__(self, prompt: str) -> tuple[str, int, int]:
        r = requests.post(f"{self.url}/api/generate", timeout=300,
                          json={"model": self.model, "prompt": prompt, "stream": False, "options": {"temperature": 0}})
        r.raise_for_status()
        d = r.json()
        return d.get("response", ""), d.get("prompt_eval_count", 0), d.get("eval_count", 0)


def get_client(spec: str):
    """'openai:gpt-4o-mini' or 'ollama:llama3.1'."""
    kind, _, model = spec.partition(":")
    if kind == "openai":
        return OpenAIClient(model or "gpt-4o-mini")
    if kind == "ollama":
        return OllamaClient(model or "llama3.1")
    raise ValueError(f"Unknown LLM spec {spec!r}")


def text_only(client):
    """Adapt a client to the prompt -> text signature used by judges."""
    return lambda prompt: client(prompt)[0]
