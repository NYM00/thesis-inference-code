from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def _safe_stem(path: Path) -> str:
    return path.stem.replace(" ", "_")


def compiled_model_path(
    *,
    weights_path: str | Path,
    image_size: int,
    clip_length: int,
    cache_root: str | Path,
) -> Path:
    weights_path = Path(weights_path)
    cache_root = Path(cache_root)
    file_name = f"{_safe_stem(weights_path)}_img{image_size}_clip{clip_length}.pt"
    return cache_root / file_name


def load_or_compile_neuron_model(
    *,
    eager_model: torch.nn.Module,
    weights_path: str | Path,
    image_size: int,
    clip_length: int,
    cache_root: str | Path,
    force_recompile: bool = False,
) -> tuple[Any, Path]:
    try:
        import torch_neuronx  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "torch_neuronx is not installed. Run this only inside the Neuron-enabled environment."
        ) from exc

    compiled_path = compiled_model_path(
        weights_path=weights_path,
        image_size=image_size,
        clip_length=clip_length,
        cache_root=cache_root,
    )
    compiler_workdir = compiled_path.parent / f"{compiled_path.stem}_compile_workdir"

    compiled_path.parent.mkdir(parents=True, exist_ok=True)

    if force_recompile and compiled_path.exists():
        compiled_path.unlink()

    if not compiled_path.exists():
        example_input = torch.randn(
            1,
            3,
            clip_length,
            image_size,
            image_size,
            dtype=torch.float32,
        )
        eager_model = eager_model.cpu().eval()
        traced = torch_neuronx.trace(
            eager_model,
            (example_input,),
            compiler_workdir=str(compiler_workdir),
        )
        torch.jit.save(traced, str(compiled_path))

    loaded = torch.jit.load(str(compiled_path))
    return loaded, compiled_path
