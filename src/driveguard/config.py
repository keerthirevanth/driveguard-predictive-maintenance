"""Load the project config. Single source of truth for all experiment knobs."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Read config.yaml into a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    import json

    print(json.dumps(load_config(), indent=2, default=str))
