import argparse
import json
import os
from pathlib import Path

from asr_core import DTYPE_MAP, VibeVoiceAsr, extract_text


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


def main() -> None:
    args = parse_args()

    audio_path = Path(args.audio)
    asr = VibeVoiceAsr(
        model_dir=args.model_dir,
        device=args.device,
        dtype_name=args.dtype,
        max_new_tokens=args.max_new_tokens,
    )

    print(
        json.dumps(
            {
                **asr.info,
                "audio": str(audio_path),
            },
            ensure_ascii=True,
        )
    )

    decoded = asr.transcribe(
        audio_path=audio_path,
        prompt=args.prompt,
        return_format=args.format,
        max_new_tokens=args.max_new_tokens,
    )

    if isinstance(decoded, (dict, list)):
        print(json.dumps(decoded, ensure_ascii=False, indent=2))
    else:
        print(extract_text(decoded))


if __name__ == "__main__":
    main()
