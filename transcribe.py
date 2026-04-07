import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration


DTYPE_MAP = {
    "auto": None,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VibeVoice-ASR from a local model directory."
    )
    parser.add_argument(
        "--audio",
        required=True,
        help="Path to an audio file inside the container, for example /data/audio/sample.wav",
    )
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("MODEL_DIR", "/models/vibevoice"),
        help="Mounted local model directory",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Optional transcription context / hotwords prompt",
    )
    parser.add_argument(
        "--format",
        choices=["raw", "parsed", "transcription_only"],
        default="parsed",
        help="Output format from the processor decoder",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Target device. Examples: auto, cuda, cuda:0, cpu",
    )
    parser.add_argument(
        "--dtype",
        choices=sorted(DTYPE_MAP.keys()),
        default="auto",
        help="Torch dtype override",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Maximum number of generated tokens",
    )
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()

    model_dir = Path(args.model_dir)
    audio_path = Path(args.audio)
    validate_local_model(model_dir)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    requested_device = args.device
    if requested_device == "auto" and not torch.cuda.is_available():
        requested_device = "cpu"

    dtype = resolve_dtype(requested_device, args.dtype)
    processor, model = load_model(str(model_dir), args.device, dtype)

    model_device = model.device
    model_dtype = getattr(model, "dtype", dtype or torch.float32)

    print(
        json.dumps(
            {
                "model_dir": str(model_dir),
                "audio": str(audio_path),
                "device": str(model_device),
                "dtype": str(model_dtype),
            },
            ensure_ascii=True,
        )
    )

    inputs = processor.apply_transcription_request(
        audio=str(audio_path),
        prompt=args.prompt,
    )
    inputs = inputs.to(model_device, model_dtype)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

    generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]

    if args.format == "raw":
        decoded = processor.decode(generated_ids)[0]
        print(decoded)
        return

    decoded = processor.decode(generated_ids, return_format=args.format)[0]
    if isinstance(decoded, (dict, list)):
        print(json.dumps(decoded, ensure_ascii=False, indent=2))
    else:
        print(decoded)


if __name__ == "__main__":
    main()
