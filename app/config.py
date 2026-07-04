import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    d360_api_key: str
    d360_api_base_url: str
    webhook_auth_mode: str
    webhook_bearer_token: str
    webhook_basic_user: str
    webhook_basic_pass: str
    openai_api_key: str
    openai_transcribe_model: str
    log_level: str
    openai_model: str
    langsmith_api_key: str
    langsmith_project: str
    langsmith_tracing: bool
    conversation_max_messages: int

    @classmethod
    def from_env(cls) -> "Settings":
        api_key = os.getenv("D360_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Missing D360_API_KEY. Copy example.env to .env and set it.")

        auth_mode = os.getenv("WEBHOOK_AUTH_MODE", "none").strip().lower()
        if auth_mode not in {"none", "bearer", "basic"}:
            raise RuntimeError("WEBHOOK_AUTH_MODE must be one of: none, bearer, basic")

        return cls(
            d360_api_key=api_key,
            d360_api_base_url=os.getenv(
                "D360_API_BASE_URL",
                "https://waba-v2.360dialog.io",
            ).rstrip("/"),
            webhook_auth_mode=auth_mode,
            webhook_bearer_token=os.getenv("WEBHOOK_BEARER_TOKEN", "").strip(),
            webhook_basic_user=os.getenv("WEBHOOK_BASIC_USER", "").strip(),
            webhook_basic_pass=os.getenv("WEBHOOK_BASIC_PASS", "").strip(),
            openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            openai_transcribe_model=os.getenv(
                "OPENAI_TRANSCRIBE_MODEL",
                "gpt-4o-transcribe",
            ).strip(),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini").strip(),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            langsmith_api_key=os.getenv("LANGSMITH_API_KEY", "").strip(),
            langsmith_project=os.getenv("LANGSMITH_PROJECT", "echo2").strip(),
            langsmith_tracing=os.getenv("LANGSMITH_TRACING_V2", "false").strip().lower() == "true",
            conversation_max_messages=int(os.getenv("CONVERSATION_MAX_MESSAGES", "20")),
        )

settings = Settings.from_env()