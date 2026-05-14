from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KV_", env_file=".env", extra="ignore")

    serve_env: Literal["dev", "staging", "prod"] = Field(default="dev", alias="KV_SERVE_ENV")
    api_tokens: str = Field(
        default="dev:dev-token",
        alias="KV_API_TOKENS",
        description="Comma-separated tenant:token pairs.",
    )
    model_registry: Path = Field(default=Path("config/model_registry.json"), alias="KV_MODEL_REGISTRY")
    rate_limit_rpm: int = Field(default=600, alias="KV_RATE_LIMIT_RPM")
    default_policy_mode: Literal[
        "quality_first", "balanced", "memory_first", "throughput_first"
    ] = Field(default="balanced", alias="KV_DEFAULT_POLICY_MODE")
    nvme_cache_dir: Path = Field(default=Path(".kv-cache"), alias="KV_NVME_CACHE_DIR")
    require_cuda: bool = Field(default=False, alias="KV_REQUIRE_CUDA")
    warm_cuda: bool = Field(default=False, alias="KV_WARM_CUDA")
    prefix_near_match_hamming: int = Field(default=3, alias="KV_PREFIX_NEAR_MATCH_HAMMING")
    max_request_tokens: int = Field(default=32768, alias="KV_MAX_REQUEST_TOKENS")

    def token_map(self) -> dict[str, str]:
        pairs: dict[str, str] = {}
        for raw_pair in self.api_tokens.split(","):
            raw_pair = raw_pair.strip()
            if not raw_pair:
                continue
            tenant, sep, token = raw_pair.partition(":")
            if not sep or not tenant or not token:
                raise ValueError("KV_API_TOKENS must contain comma-separated tenant:token pairs")
            pairs[tenant] = token
        if self.serve_env == "prod" and pairs == {"dev": "dev-token"}:
            raise ValueError("KV_API_TOKENS must be configured explicitly in production")
        return pairs


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
