from __future__ import annotations

import base64
from typing import Any

import requests
from requests import Response
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import Settings


class NvidiaClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.settings.nvidia_api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _raise_for_api_error(response: Response) -> None:
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = ""
            try:
                payload = response.json()
                detail = payload.get("error", {}).get("message") or payload.get("message") or ""
            except ValueError:
                detail = response.text[:300]
            message = f"NVIDIA API request failed with status {response.status_code}."
            if detail:
                message += f" Details: {detail}"
            raise RuntimeError(message) from exc

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.settings.nvidia_base_url}/{path.lstrip('/')}"
        try:
            response = self.session.post(
                url,
                headers=self._headers(),
                json=payload,
                timeout=60,
            )
        except requests.RequestException as exc:
            raise RuntimeError(
                "Could not reach the NVIDIA API. Check the base URL, key, model names, and network connection."
            ) from exc
        self._raise_for_api_error(response)
        try:
            return response.json()
        except ValueError as exc:
            raise RuntimeError("NVIDIA API returned a non-JSON response.") from exc

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
