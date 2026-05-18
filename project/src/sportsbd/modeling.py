from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r2plus1d_18
from .neuron_backend import load_or_compile_neuron_model


DEFAULT_IMAGE_SIZE = 112
DEFAULT_MEAN = (0.43216, 0.394666, 0.37645)
DEFAULT_STD = (0.22803, 0.22145, 0.216989)
CANONICAL_CLASS_NAMES = ["hard", "fade_in", "clip_scene", "negative"]

LABEL_ALIASES = {
    "hard": "hard",
    "cut": "hard",
    "fadein": "fade_in",
    "fade_in": "fade_in",
    "fade-in": "fade_in",
    "logo": "clip_scene",
    "clip_scene": "clip_scene",
    "clipscene": "clip_scene",
    "replay_logo": "clip_scene",
    "negative": "negative",
    "nan": "negative",
    "none": "negative",
    "background": "negative",
}


def normalize_label_name(name: str | None) -> str | None:
    if name is None:
        return None
    normalized = name.strip().lower().replace(" ", "_")
    return LABEL_ALIASES.get(normalized, normalized)


def normalize_class_names(class_names: list[str] | tuple[str, ...] | None) -> list[str]:
    if not class_names:
        return list(CANONICAL_CLASS_NAMES)
    normalized = [normalize_label_name(name) or "" for name in class_names]
    return [name if name else CANONICAL_CLASS_NAMES[index] for index, name in enumerate(normalized)]


def _torch_load(path: Path, map_location: str | torch.device) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def build_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name != "r2plus1d_18":
        raise ValueError(f"Unsupported SportSBD model: {model_name}")
    model = r2plus1d_18(weights=None)
    in_features = model.fc.in_features  # type: ignore[attr-defined]
    model.fc = nn.Linear(in_features, num_classes)  # type: ignore[attr-defined]
    return model


def load_checkpoint_bundle(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    *,
    clip_length: int | None = None,
    compile_artifacts_dir: str | Path | None = None,
    force_recompile: bool = False,
) -> dict[str, Any]:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    normalized_device = str(device).strip().lower()
    map_location = "cpu" if normalized_device == "neuron" else device

    checkpoint = _torch_load(path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unexpected checkpoint structure: {path}")

    state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
    if state_dict is None:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint state dict missing in: {path}")

    config = checkpoint.get("config", {})
    if not isinstance(config, dict):
        raise ValueError(f"Checkpoint config must be a dict: {path}")

    class_names = normalize_class_names(config.get("CLASS_NAMES"))
    image_size = int(config.get("IMG_SIZE", DEFAULT_IMAGE_SIZE))
    num_classes = int(config.get("NUM_CLASSES", len(class_names)))
    model_name = str(config.get("MODEL_NAME", "r2plus1d_18"))

    cleaned_state_dict: dict[str, Any] = {}
    for key, value in state_dict.items():
        cleaned_state_dict[key[7:] if key.startswith("module.") else key] = value

    model = build_model(model_name=model_name, num_classes=num_classes)
    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected state dict keys in {path}: {unexpected}")
    if missing:
        raise ValueError(f"Missing state dict keys in {path}: {missing}")

    if normalized_device == "neuron":
        if clip_length is None:
            raise ValueError("clip_length must be provided for Neuron compilation.")
        if compile_artifacts_dir is None:
            raise ValueError("compile_artifacts_dir must be provided for Neuron compilation.")

        compiled_model, compiled_path = load_or_compile_neuron_model(
            eager_model=model,
            weights_path=path,
            image_size=image_size,
            clip_length=clip_length,
            cache_root=compile_artifacts_dir,
            force_recompile=force_recompile,
        )

        return {
            "model": compiled_model,
            "device": "neuron",
            "backend": "neuron",
            "config": config,
            "class_names": class_names,
            "image_size": image_size,
            "num_classes": num_classes,
            "mean": tuple(DEFAULT_MEAN),
            "std": tuple(DEFAULT_STD),
            "checkpoint_path": str(path),
            "compiled_model_path": str(compiled_path),
        }

    torch_device = torch.device(device)
    model.to(torch_device)
    model.eval()

    return {
        "model": model,
        "device": torch_device,
        "backend": "eager",
        "config": config,
        "class_names": class_names,
        "image_size": image_size,
        "num_classes": num_classes,
        "mean": tuple(DEFAULT_MEAN),
        "std": tuple(DEFAULT_STD),
        "checkpoint_path": str(path),
    }


def preprocess_bgr_frame(
    frame_bgr: np.ndarray,
    image_size: int,
    mean: tuple[float, float, float] = DEFAULT_MEAN,
    std: tuple[float, float, float] = DEFAULT_STD,
) -> torch.Tensor:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    mean_tensor = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_tensor = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean_tensor) / std_tensor


def predict_clip_tensor(
    clip_frames: list[torch.Tensor],
    model: nn.Module,
    class_names: list[str],
    device: torch.device | str,
    backend: str = "eager",
) -> dict[str, Any]:
    if not clip_frames:
        raise ValueError("predict_clip_tensor requires at least one frame")

    clip = torch.stack(clip_frames, dim=0).permute(1, 0, 2, 3).unsqueeze(0)

    if backend != "neuron":
        clip = clip.to(device)

    with torch.no_grad():
        logits = model(clip)
        probabilities = F.softmax(logits, dim=1)[0].detach().cpu().numpy()

    predicted_index = int(np.argmax(probabilities))
    predicted_class = class_names[predicted_index] if predicted_index < len(class_names) else str(predicted_index)
    return {
        "class_probs": [float(item) for item in probabilities.tolist()],
        "predicted_index": predicted_index,
        "predicted_class": predicted_class,
        "confidence": float(probabilities[predicted_index]),
        "any_boundary_prob": float(np.sum(probabilities[: max(len(class_names) - 1, 0)])),
    }


