from __future__ import annotations

import base64
from typing import Any

import requests

from src.config import Settings


class NvidiaClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.settings.nvidia_api_key}",
            "Content-Type": "application/json",
        }

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.settings.nvidia_base_url}/{path.lstrip('/')}"
        response = requests.post(url, headers=self._headers(), json=payload, timeout=60)
        response.raise_for_status()
        return response.json()

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.2,
        response_format_json: bool = False,
    ) -> str:
        if self.settings.is_mock_mode:
            return '{"mock_mode": true}'
        payload: dict[str, Any] = {
            "model": model or self.settings.nvidia_chat_model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format_json:
            payload["response_format"] = {"type": "json_object"}
        data = self._post("/chat/completions", payload)
        return data["choices"][0]["message"]["content"]

    def embed_texts(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        if not self.settings.embed_enabled:
            raise RuntimeError("Embedding model is not configured.")
        payload = {"model": model or self.settings.nvidia_embed_model, "input": texts}
        data = self._post("/embeddings", payload)
        return [item["embedding"] for item in data.get("data", [])]

    def analyze_image(
        self,
        image_bytes: bytes,
        prompt: str,
        model: str | None = None,
    ) -> str:
        if not self.settings.vision_enabled:
            raise RuntimeError("Vision model is not configured.")
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        payload = {
            "model": model or self.settings.nvidia_vision_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                    ],
                }
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }
        data = self._post("/chat/completions", payload)
        return data["choices"][0]["message"]["content"]

    def rerank(
        self,
        query: str,
        documents: list[str],
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        if not self.settings.rerank_enabled:
            return [
                {"index": index, "relevance_score": 1.0 / (index + 1)}
                for index, _ in enumerate(documents)
            ]
        payload = {
            "model": model or self.settings.nvidia_rerank_model,
            "query": query,
            "documents": documents,
        }
        data = self._post("/reranking", payload)
        return data.get("results", [])
