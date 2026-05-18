from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json_config(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a JSON object: {config_path}")
    return data


def _parser_defaults(parser: argparse.ArgumentParser) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for action in parser._actions:
        if not action.dest or action.dest == "help":
            continue
        defaults[action.dest] = action.default
    return defaults


def apply_config_defaults(args: argparse.Namespace, parser: argparse.ArgumentParser) -> tuple[argparse.Namespace, dict[str, Any]]:
    config_data = load_json_config(getattr(args, "config", None))
    if not config_data:
        return args, {}

    defaults = _parser_defaults(parser)
    for key, value in config_data.items():
        if not hasattr(args, key):
            continue
        if getattr(args, key) == defaults.get(key):
            setattr(args, key, value)
    return args, config_data


def namespace_to_dict(args: argparse.Namespace, *, exclude: set[str] | None = None) -> dict[str, Any]:
    exclude = exclude or set()
    output: dict[str, Any] = {}
    for key, value in vars(args).items():
        if key in exclude:
            continue
        output[key] = json_safe(value)
    return output


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value

