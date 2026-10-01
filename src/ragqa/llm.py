"""A thin client for any OpenAI-compatible chat API.

By default it talks to a local Ollama server, so open-source models (Llama, Qwen, Gemma) run on the
laptop with no key. Point LLM_BASE_URL and LLM_API_KEY at a hosted provider to use that instead.
"""
import os

import httpx

DEFAULT_BASE_URL = "http://localhost:11434/v1"


class LLM:
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 client: httpx.Client | None = None):
        self.model = model
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "local")
        self.client = client or httpx.Client(timeout=180)

    def chat(self, messages: list[dict], temperature: float = 0.0, top_p: float = 1.0, max_tokens: int = 300,
             json_mode: bool = False, seed: int | None = 0) -> str:
        """temperature 0 picks the likeliest token each step; higher values and top_p < 1 sample more freely."""
        body = {"model": self.model, "messages": messages, "temperature": temperature, "top_p": top_p,
                "max_tokens": max_tokens}
        if seed is not None:
            body["seed"] = seed
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        response = self.client.post(f"{self.base_url}/chat/completions", json=body,
                                    headers={"Authorization": f"Bearer {self.api_key}"})
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
