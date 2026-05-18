from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from .config import json_safe


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(path: str | Path, payload: Any) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2)
    return output_path


def write_csv_rows(path: str | Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_safe(value) for key, value in row.items()})
    return output_path


def append_csv_row(path: str | Path, fieldnames: list[str], row: dict[str, Any]) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    file_exists = output_path.is_file()
    with output_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: json_safe(value) for key, value in row.items()})
    return output_path

