import os
import sys
from pathlib import Path
import uvicorn
from src.api.rest.v1.app import app
from src.infra.config.config_loader import AppConfig


def get_server_config() -> tuple[str, int]:
    cfg = AppConfig.load()
    server_cfg = cfg.get("server", {})
    host = os.getenv("HOST") or server_cfg.get("host") or "0.0.0.0"
    port_str = os.getenv("PORT") or str(server_cfg.get("port", "8080"))
    try:
        port = int(port_str)
    except ValueError:
        port = 8080
    return host, port


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    host, port = get_server_config()
    uvicorn.run(
        "src.api.rest.v1.app:app",
        host=host,
        port=port,
        reload=os.getenv("ENV", "production").lower() in ("development", "dev", "local"),
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )


if __name__ == "__main__":
    main()
