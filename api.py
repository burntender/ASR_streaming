import os
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from asr_core import VibeVoiceAsr, extract_text


API_MODEL_NAME = os.environ.get("OPENAI_COMPAT_MODEL", "vibevoice-asr")

app = FastAPI(title="VibeVoice-ASR OpenAI-compatible API")
asr = VibeVoiceAsr()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
def list_models() -> dict[str, list[dict[str, str]]]:
    return {
        "object": "list",
        "data": [
            {
                "id": API_MODEL_NAME,
                "object": "model",
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/audio/transcriptions")
async def create_transcription(
    file: Annotated[UploadFile, File()],
    model: Annotated[str, Form()],
    language: Annotated[str | None, Form()] = None,
    prompt: Annotated[str | None, Form()] = None,
    response_format: Annotated[str | None, Form()] = None,
    temperature: Annotated[float | None, Form()] = None,
) -> JSONResponse:
    del language, response_format, temperature

    if model != API_MODEL_NAME:
        raise HTTPException(status_code=404, detail=f"Model not found: {model}")

    suffix = Path(file.filename or "").suffix or ".audio"
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)

        decoded = asr.transcribe(
            tmp_path,
            prompt=prompt,
            return_format="transcription_only",
        )
        return JSONResponse({"text": extract_text(decoded)})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if "tmp_path" in locals():
            tmp_path.unlink(missing_ok=True)
