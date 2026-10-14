"""Centralized application configuration loader strictly reading from config directory."""

from pathlib import Path
from typing import Any, Mapping
import yaml


class AppConfig:
    _cached_config: Mapping[str, Any] | None = None

    @classmethod
    def load(cls) -> Mapping[str, Any]:
        if cls._cached_config is None:
            config_path = Path(__file__).resolve().parent.parent.parent.parent / "config" / "default.yaml"
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    cls._cached_config = yaml.safe_load(f) or {}
            else:
                cls._cached_config = {}
        return cls._cached_config

    @classmethod
    def get_endpoint(cls, key: str, default: str = "") -> str:
        cfg = cls.load()
        routes = cfg.get("routes", {})
        return routes.get(key, default)
