from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(slots=True)
class Settings:
    nvidia_api_key: str = os.getenv("NVIDIA_API_KEY", "")
    nvidia_base_url: str = os.getenv(
        "NVIDIA_BASE_URL",
        "https://integrate.api.nvidia.com/v1",
    ).rstrip("/")
    nvidia_chat_model: str = os.getenv("NVIDIA_CHAT_MODEL", "")
    nvidia_embed_model: str = os.getenv("NVIDIA_EMBED_MODEL", "")
    nvidia_vision_model: str = os.getenv("NVIDIA_VISION_MODEL", "")
    nvidia_rerank_model: str = os.getenv("NVIDIA_RERANK_MODEL", "")

    @property
    def is_mock_mode(self) -> bool:
        return not (self.nvidia_api_key and self.nvidia_chat_model)

    @property
    def embed_enabled(self) -> bool:
        return bool(self.nvidia_api_key and self.nvidia_embed_model)

    @property
    def vision_enabled(self) -> bool:
        return bool(self.nvidia_api_key and self.nvidia_vision_model)

    @property
    def rerank_enabled(self) -> bool:
        return bool(self.nvidia_api_key and self.nvidia_rerank_model)


def get_settings() -> Settings:
    return Settings()
