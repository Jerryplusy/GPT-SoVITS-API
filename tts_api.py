import argparse
import base64
import io
import os
import re
import sys
import uuid
import wave
import yaml
from contextlib import asynccontextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Header, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from GPT_SoVITS.TTS_infer_pack.TTS import TTS


INPUT_DIR = PROJECT_ROOT / "input"
REF_AUDIO_PATH = INPUT_DIR / "ref.mp3"
REF_TEXT_PATH = INPUT_DIR / "ref_text.txt"


def load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def detect_lang(text: str) -> str:
    if re.search(r"[\u3040-\u30ff]", text):
        return "all_ja"
    if re.search(r"[\u4e00-\u9fff]", text):
        return "all_zh"
    return "en"


def read_ref_text(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""


class TTSService:
    def __init__(self, config: dict):
        self.config = config
        self.tts_config = {
            "custom": {
                "device": config["tts"]["device"],
                "is_half": config["tts"]["is_half"],
                "version": config["tts"]["version"],
                "t2s_weights_path": config["tts"]["t2s_weights_path"],
                "vits_weights_path": config["tts"]["vits_weights_path"],
                "bert_base_path": config["tts"]["bert_base_path"],
                "cnhuhbert_base_path": config["tts"]["cnhuhbert_base_path"],
            }
        }
        self.tts = None

    def init_tts(self):
        if self.tts is None:
            print("Initializing TTS model...")
            self.tts = TTS(self.tts_config)
            print("TTS model initialized.")

    def synthesize(
        self,
        text: str,
        text_lang: str = None,
    ) -> tuple[int, np.ndarray]:
        self.init_tts()

        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        if not REF_AUDIO_PATH.exists():
            raise FileNotFoundError(f"Reference audio not found: {REF_AUDIO_PATH}")

        prompt_text = read_ref_text(REF_TEXT_PATH)
        if not prompt_text:
            raise FileNotFoundError(f"Prompt text not found: {REF_TEXT_PATH}")

        if text_lang is None:
            text_lang = detect_lang(text)
        prompt_lang = detect_lang(prompt_text)

        result = list(
            self.tts.run(
                {
                    "text": text.strip(),
                    "text_lang": text_lang,
                    "ref_audio_path": str(REF_AUDIO_PATH),
                    "prompt_text": prompt_text,
                    "prompt_lang": prompt_lang,
                    "top_k": self.config["tts"].get("top_k", 15),
                    "top_p": self.config["tts"].get("top_p", 1.0),
                    "temperature": self.config["tts"].get("temperature", 1.0),
                    "batch_size": self.config["tts"].get("batch_size", 1),
                    "batch_threshold": self.config["tts"].get("batch_threshold", 0.75),
                    "split_bucket": self.config["tts"].get("split_bucket", True),
                    "text_split_method": "cut5",
                    "speed_factor": self.config["tts"].get("speed_factor", 1.0),
                    "fragment_interval": self.config["tts"].get("fragment_interval", 0.3),
                    "seed": self.config["tts"].get("seed", 42),
                    "parallel_infer": self.config["tts"].get("parallel_infer", True),
                    "repetition_penalty": self.config["tts"].get("repetition_penalty", 1.35),
                    "sample_steps": self.config["tts"].get("sample_steps", 12),
                    "super_sampling": self.config["tts"].get("super_sampling", False),
                    "streaming_mode": False,
                }
            )
        )

        if not result:
            raise RuntimeError("TTS inference returned no audio")

        return result[-1]


tts_service: TTSService = None


def verify_api_key(x_api_key: str = None, config: dict = None):
    if config is None:
        config = load_config()
    expected_key = config.get("server", {}).get("api_key", "")
    if expected_key and x_api_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts_service
    config = load_config()
    tts_service = TTSService(config)
    yield


app = FastAPI(
    title="GPT-SoVITS TTS API",
    description="Text-to-Speech API using GPT-SoVITS",
    version="1.0.0",
    lifespan=lifespan,
)


class TTSRequest(BaseModel):
    text: str
    text_lang: str | None = None


class TTSResponse(BaseModel):
    audio_base64: str
    sample_rate: int
    text: str


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/info")
async def get_info():
    return {
        "ref_audio_exists": REF_AUDIO_PATH.exists(),
        "ref_audio_path": str(REF_AUDIO_PATH),
        "ref_text_exists": REF_TEXT_PATH.exists(),
        "ref_text_path": str(REF_TEXT_PATH),
    }


@app.post("/tts", response_model=TTSResponse)
async def synthesize_speech(
    request: TTSRequest,
    x_api_key: str = Header(None, alias="X-API-Key"),
):
    config = load_config()
    verify_api_key(x_api_key, config)

    try:
        sr, audio = tts_service.synthesize(
            text=request.text,
            text_lang=request.text_lang,
        )

        buffer = io.BytesIO()
        sf.write(buffer, audio, sr, format="WAV")
        buffer.seek(0)

        audio_base64 = base64.b64encode(buffer.read()).decode("utf-8")

        return TTSResponse(
            audio_base64=audio_base64,
            sample_rate=sr,
            text=request.text,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/tts/wav")
async def synthesize_speech_wav(
    text: str,
    text_lang: str | None = None,
    x_api_key: str = Header(None, alias="X-API-Key"),
):
    config = load_config()
    verify_api_key(x_api_key, config)

    try:
        sr, audio = tts_service.synthesize(
            text=text,
            text_lang=text_lang,
        )

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sr)
            wav_file.writeframes(audio.tobytes())

        buffer.seek(0)

        filename = f"tts_{uuid.uuid4().hex[:8]}.wav"

        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


def main():
    parser = argparse.ArgumentParser(description="GPT-SoVITS TTS API Server")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--host", type=str, default=None, help="Host to bind")
    parser.add_argument("--port", type=int, default=None, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    config = load_config(args.config)

    import uvicorn

    host = args.host or config.get("server", {}).get("host", "0.0.0.0")
    port = args.port or config.get("server", {}).get("port", 8000)

    print(f"Starting server on {host}:{port}")
    print(f"API Key: {config.get('server', {}).get('api_key', 'NOT SET')}")
    print(f"Reference audio: {REF_AUDIO_PATH}")
    print(f"Reference text: {REF_TEXT_PATH}")

    uvicorn.run(
        "tts_api:app",
        host=host,
        port=port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
