"""
llm.py
======
A provider-agnostic wrapper so the rest of the codebase never hard-codes a
vendor. Swap providers by changing one env var (LLM_PROVIDER) -- the pipeline,
the LLM-as-judge, everything keeps working. That decoupling is deliberate good
engineering, and it means a future employer's stack (whatever it is) is a small
config change, not a rewrite.

Supported backends:
  - "anthropic"  (Claude)         -> needs ANTHROPIC_API_KEY
  - "openai"     (GPT)            -> needs OPENAI_API_KEY
  - "ollama"     (local, free)    -> needs a running ollama server, no key

SDKs are imported lazily so you only need the one you actually use installed.
"""

from __future__ import annotations

import os


class LLMClient:
    def __init__(self, provider: str | None = None, model: str | None = None,
                 temperature: float = 0.0):
        self.provider = (provider or os.environ.get("LLM_PROVIDER", "anthropic")).lower()
        self.temperature = temperature
        # Sensible default models per provider; override via the `model` arg.
        self.model = model or {
            "anthropic": "claude-sonnet-4-6",
            "openai": "gpt-4o-mini",
            "ollama": "llama3.1",
        }.get(self.provider, "claude-sonnet-4-6")

    def generate(self, prompt: str, system: str = "") -> str:
        """Send one prompt, get the text completion back as a plain string."""
        if self.provider == "anthropic":
            return self._anthropic(prompt, system)
        if self.provider == "openai":
            return self._openai(prompt, system)
        if self.provider == "ollama":
            return self._ollama(prompt, system)
        raise ValueError(f"Unknown LLM provider: {self.provider}")

    # ---- backends (lazy imports keep dependencies optional) ----
    def _anthropic(self, prompt: str, system: str) -> str:
        import anthropic
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        resp = client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=self.temperature,
            system=system or "You are a careful biomedical assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")

    def _openai(self, prompt: str, system: str) -> str:
        from openai import OpenAI
        client = OpenAI()  # reads OPENAI_API_KEY from env
        resp = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system or "You are a careful biomedical assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content or ""

    def _ollama(self, prompt: str, system: str) -> str:
        import requests
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": self.model, "system": system, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")
