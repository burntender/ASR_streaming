import json
import os
import threading
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration


DTYPE_MAP = {
    "auto": None,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def configure_torch_backend() -> None:
    if os.environ.get("ASR_DISABLE_MKLDNN", "1").lower() not in ("0", "false", "no"):
        torch.backends.mkldnn.enabled = False

    cpu_threads = os.environ.get("ASR_CPU_THREADS")
    if cpu_threads:
        torch.set_num_threads(int(cpu_threads))


def validate_local_model(model_dir: Path) -> None:
    required = [
        model_dir / "config.json",
        model_dir / "model.safetensors.index.json",
        model_dir / "tokenizer.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Local model directory is incomplete: " + ", ".join(missing)
        )

    shard = model_dir / "model-00001-of-00008.safetensors"
    if shard.exists() and shard.stat().st_size < 1024:
        raise RuntimeError(
            "Model shards still look like Git LFS pointers. Download the real safetensors files first."
        )


def resolve_device(requested_device: str) -> str:
    if requested_device == "auto" and not torch.cuda.is_available():
        return "cpu"
    return requested_device


def resolve_dtype(device: str, dtype_name: str) -> torch.dtype | None:
    if dtype_name != "auto":
        return DTYPE_MAP[dtype_name]
    if device == "cpu":
        return torch.float32
    if torch.cuda.is_available():
        return torch.bfloat16
    return torch.float32


def load_model(model_dir: str, device: str, dtype: torch.dtype | None):
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    kwargs = {"local_files_only": True}
    if dtype is not None:
        kwargs["torch_dtype"] = dtype

    if device == "auto":
        kwargs["device_map"] = "auto"
        model = VibeVoiceAsrForConditionalGeneration.from_pretrained(model_dir, **kwargs)
        return processor, model

    model = VibeVoiceAsrForConditionalGeneration.from_pretrained(model_dir, **kwargs)
    if dtype is None:
        model = model.to(device)
    else:
        model = model.to(device=device, dtype=dtype)
    return processor, model


def extract_text(decoded: Any) -> str:
    if isinstance(decoded, str):
        return decoded
    if isinstance(decoded, dict):
        for key in ("text", "transcription", "transcript"):
            value = decoded.get(key)
            if isinstance(value, str):
                return value
        return json.dumps(decoded, ensure_ascii=False)
    if isinstance(decoded, list):
        parts = [extract_text(item) for item in decoded]
        return "\n".join(part for part in parts if part)
    return str(decoded)


class VibeVoiceAsr:
    def __init__(
        self,
        model_dir: str | None = None,
        device: str | None = None,
        dtype_name: str | None = None,
        max_new_tokens: int | None = None,
    ) -> None:
        configure_torch_backend()
        self.model_dir = Path(model_dir or os.environ.get("MODEL_DIR", "/models/vibevoice"))
        self.requested_device = device or os.environ.get("ASR_DEVICE", "auto")
        self.dtype_name = dtype_name or os.environ.get("ASR_DTYPE", "auto")
        self.max_new_tokens = int(
            max_new_tokens or os.environ.get("ASR_MAX_NEW_TOKENS", "2048")
        )
        self._processor = None
        self._model = None
        self._model_device = None
        self._model_dtype = None
        self._load_lock = threading.Lock()
        self._generate_lock = threading.Lock()

    def load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return

            validate_local_model(self.model_dir)
            resolved_device = resolve_device(self.requested_device)
            dtype = resolve_dtype(resolved_device, self.dtype_name)
            self._processor, self._model = load_model(
                str(self.model_dir), resolved_device, dtype
            )
            self._model_device = self._model.device
            self._model_dtype = getattr(self._model, "dtype", dtype or torch.float32)

    @property
    def info(self) -> dict[str, str]:
        self.load()
        return {
            "model_dir": str(self.model_dir),
            "device": str(self._model_device),
            "dtype": str(self._model_dtype),
        }

    def transcribe(
        self,
        audio_path: str | Path,
        prompt: str | None = None,
        return_format: str = "transcription_only",
        max_new_tokens: int | None = None,
    ):
        self.load()
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        inputs = self._processor.apply_transcription_request(
            audio=str(audio_path),
            prompt=prompt,
        )
        inputs = inputs.to(self._model_device, self._model_dtype)

        with self._generate_lock, torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
            )

        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]

        if return_format == "raw":
            return self._processor.decode(generated_ids)[0]

        return self._processor.decode(generated_ids, return_format=return_format)[0]
