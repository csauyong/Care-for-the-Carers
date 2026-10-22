"""Settings come from environment variables, optionally loaded from a .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROVIDERS = ("anthropic", "openai", "scripted")
DETECTORS = ("keyword", "llm")


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Settings:
    telegram_token: str | None = None
    provider: str = "anthropic"
    model: str | None = None
    effort: str = "low"
    openai_base_url: str | None = None
    detector: str = "keyword"
    data_dir: Path = Path("data")
    trace_content: bool = True
    resources_file: Path | None = None
    stress_threshold: float = 3.0
    history_turns: int = 20
    webhook_url: str | None = None
    host: str = "0.0.0.0"
    port: int = 8080
    judge_model: str = "claude-opus-5-5"
    simulator_model: str = "claude-sonnet-5-5"

    @classmethod
    def from_env(cls, env_file: str | None = None) -> "Settings":
        load_dotenv(env_file or ".env")
        load_dotenv("keys.env")  # the file name used by version 1
        env = os.environ.get
        port = env("CAREBOT_PORT") or env("PORT")  # PORT is what most PaaS hosts set
        resources_file = env("CAREBOT_RESOURCES_FILE")
        return cls(
            telegram_token=env("TELEGRAM_BOT_TOKEN") or env("TELEGRAM_API_KEY"),
            provider=env("CAREBOT_PROVIDER", "anthropic").lower(),
            model=env("CAREBOT_MODEL") or None,
            effort=env("CAREBOT_EFFORT", "low"),
            openai_base_url=env("OPENAI_BASE_URL") or None,
            detector=env("CAREBOT_DETECTOR", "keyword").lower(),
            data_dir=Path(env("CAREBOT_DATA_DIR", "data")),
            trace_content=env("CAREBOT_TRACE_CONTENT", "true").lower() not in ("0", "false", "no"),
            resources_file=Path(resources_file) if resources_file else None,
            stress_threshold=float(env("CAREBOT_STRESS_THRESHOLD", "3")),
            history_turns=int(env("CAREBOT_HISTORY_TURNS", "20")),
            webhook_url=env("CAREBOT_WEBHOOK_URL") or None,
            host=env("CAREBOT_HOST", "0.0.0.0"),
            port=int(port or 8080),
            judge_model=env("CAREBOT_JUDGE_MODEL", "claude-opus-5-5"),
            simulator_model=env("CAREBOT_SIMULATOR_MODEL", "claude-sonnet-5-5"),
        )

    def validate(self, need_telegram: bool = False) -> None:
        problems = []
        if self.provider not in PROVIDERS:
            problems.append(f"CAREBOT_PROVIDER must be one of {', '.join(PROVIDERS)} (got {self.provider!r})")
        if self.detector not in DETECTORS:
            problems.append(f"CAREBOT_DETECTOR must be one of {', '.join(DETECTORS)} (got {self.detector!r})")
        if self.provider == "openai" and not self.model:
            problems.append("CAREBOT_MODEL is required with the openai provider (for example llama3.1 on Ollama)")
        if need_telegram and not self.telegram_token:
            problems.append("TELEGRAM_BOT_TOKEN is not set. Get one from @BotFather on Telegram.")
        if problems:
            raise ConfigError("\n".join(problems))
