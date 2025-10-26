# src/asr_whisper.py
"""
Lightweight Whisper ASR wrapper for DocScribe.
- Works with Streamlit UploadedFile, BytesIO, bytes, or file paths.
- Caches Whisper models so repeated calls are fast.
- Returns text, language, and segments (start/end/txt).

Install deps (once):
    pip install openai-whisper ffmpeg-python
    # and ensure ffmpeg is installed on your system
"""

from __future__ import annotations

import io
import os
import tempfile
from typing import Any, Dict, List, Union, Optional

try:
    import whisper  # OpenAI Whisper
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Whisper is not installed. Please run: `pip install openai-whisper ffmpeg-python` "
        "and ensure ffmpeg is installed on your system."
    ) from e


# -----------------------------
# Model caching / configuration
# -----------------------------
DEFAULT_MODEL = os.environ.get("DOCSCRIBE_WHISPER_MODEL", "base")
_MODEL_CACHE: Dict[str, Any] = {}


def _get_model(name: str = DEFAULT_MODEL):
    """Load a Whisper model by name, cached."""
    name = name or DEFAULT_MODEL
    if name not in _MODEL_CACHE:
        _MODEL_CACHE[name] = whisper.load_model(name)
    return _MODEL_CACHE[name]


# -----------------------------
# I/O helpers
# -----------------------------
def _to_wav_path(
    audio_source: Union[str, bytes, io.BytesIO, Any],
    suffix: str = ".wav",
) -> str:
    """
    Normalize various inputs (path/bytes/BytesIO/UploadedFile) to a temp WAV file path.
    The temp file is NOT deleted automatically; caller controls lifecycle.
    """
    # If it's already a path string, trust caller
    if isinstance(audio_source, str):
        return audio_source

    # If it's raw bytes
    if isinstance(audio_source, (bytes, bytearray)):
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_source)
            return tmp.name

    # If it's a BytesIO-like or Streamlit UploadedFile (has .read)
    if hasattr(audio_source, "read"):
        data = audio_source.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            return tmp.name

    # If it's a file-like object without .read()
    if isinstance(audio_source, io.IOBase):
        data = audio_source.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            return tmp.name

    raise TypeError(
        "Unsupported audio_source type. Provide a path (str), bytes/BytesIO, "
        "or a Streamlit UploadedFile."
    )


# -----------------------------
# Public transcription function
# -----------------------------
def transcribe_audio(
    audio_source: Union[str, bytes, io.BytesIO, Any],
    model_name: str = DEFAULT_MODEL,
    *,
    language: Optional[str] = None,
    task: str = "transcribe",
    fp16: Optional[bool] = None,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    Transcribe audio using Whisper.

    Args:
        audio_source: str path, bytes, BytesIO, or Streamlit UploadedFile
        model_name: whisper model size (tiny, base, small, medium, large, large-v2, etc.)
        language: force language code (e.g., "en") or None for auto
        task: "transcribe" or "translate"
        fp16: override fp16 setting (None lets Whisper choose based on device)
        temperature: decoding temperature (0.0 for deterministic)

    Returns:
        dict with keys:
            - text: str
            - language: str
            - segments: List[Dict[str, Any]] (start/end/text)
            - info: Dict[str, Any] (model_name, task, temperature)
    """
    model = _get_model(model_name)
    wav_path = _to_wav_path(audio_source, suffix=".wav")

    # Whisper transcribe options
    options = {
        "task": task,
        "temperature": temperature,
    }
    if language:
        options["language"] = language
    if fp16 is not None:
        options["fp16"] = fp16

    result = model.transcribe(wav_path, **options)

    segments_out: List[Dict[str, Any]] = []
    for seg in result.get("segments", []) or []:
        segments_out.append(
            {
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": (seg.get("text") or "").strip(),
            }
        )

    return {
        "text": (result.get("text") or "").strip(),
        "language": result.get("language", "unknown"),
        "segments": segments_out,
        "info": {
            "model_name": model_name,
            "task": task,
            "temperature": temperature,
        },
    }


# -----------------------------
# CLI quick test
# -----------------------------
if __name__ == "__main__":  # pragma: no cover
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.asr_whisper path/to/audio.wav [model_name]")
        sys.exit(1)

    path = sys.argv[1]
    mname = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL
    out = transcribe_audio(path, model_name=mname)
    print("Language:", out["language"])
    print("Text:", out["text"])
    print("Segments:", len(out["segments"]))