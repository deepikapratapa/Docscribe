"""
asr_whisper.py
----------------
A simple wrapper around OpenAI's Whisper ASR model.

Usage:
    - Can transcribe local audio files (e.g., .wav)
    - Can handle Streamlit-uploaded files via st.file_uploader

Output:
    Returns a dictionary: {"text": <transcribed_text>}
"""

import tempfile
import whisper
# import openai_whisper
from typing import Union
import streamlit as st


def transcribe_audio(audio_source: Union[str, "UploadedFile"], model_name: str = "base") -> dict:
    """
    Transcribe audio using OpenAI Whisper.

    Args:
        audio_source: Path to a .wav file (str) or a Streamlit UploadedFile object.
        model_name: Whisper model size (tiny, base, small, medium, large)

    Returns:
        dict: {"text": <transcribed_text>}
    """
    # Load model once — Whisper automatically caches models
    model = whisper.load_model(model_name)

    # Determine if it's a local file path or a Streamlit UploadedFile
    if isinstance(audio_source, str):
        audio_path = audio_source

    else:
        # Create a temporary file to save the uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_source.read())
            audio_path = tmp.name

    # Perform transcription
    result = model.transcribe(audio_path)

    return {"text": result.get("text", "").strip()}


# ---------------------------------------------------------------------
# ✅ Quick test (run manually)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    print("🔍 Whisper ASR quick test")

    if len(sys.argv) < 2:
        print("Usage: python src/asr_whisper.py path/to/audio.wav")
        sys.exit(1)

    audio_file = sys.argv[1]
    output = transcribe_audio(audio_file, model_name="base")
    print("📝 Transcription result:")
    print(output)
