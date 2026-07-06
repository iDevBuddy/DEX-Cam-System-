"""Load and persist config.yaml + .env settings."""
import os
import threading
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
# DEX_CONFIG lets tests run against a scratch config without touching the real one.
CONFIG_PATH = Path(os.environ.get("DEX_CONFIG") or ROOT / "config.yaml")

load_dotenv(ROOT / ".env")

_lock = threading.Lock()


def load() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save(cfg: dict) -> None:
    with _lock:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()
