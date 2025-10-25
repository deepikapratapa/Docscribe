# import streamlit as st

# st.set_page_config(page_title="DocScribe", page_icon="🩺")
# st.title("DocScribe — Hello, doctors!")
# st.write("Welcome to DocScribe. This is a minimal hello-world Streamlit app.")
# if st.button("Say hello"):
#     st.success("👋 Hello from DocScribe!")

# uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])
# if uploaded_file:
#     from src.asr_whisper import transcribe_audio
#     result = transcribe_audio(uploaded_file)
#     st.write(result["text"])


import streamlit as st
from asr_whisper import transcribe_audio

st.set_page_config(page_title="DocScribe", page_icon="🩺", layout="centered")

st.title("🩺 DocScribe: Medical Voice Transcriber")
st.markdown("Record or upload a `.wav` file to transcribe notes, dictations, or consultations.")

# --- Audio upload ---
uploaded_audio = st.file_uploader("🎙️ Upload an audio file", type=["wav"])

# --- Model selection ---
model_choice = st.selectbox(
    "Select Whisper model size:",
    ["tiny", "base", "small", "medium", "large"],
    index=1,
)

# --- Transcription trigger ---
if uploaded_audio is not None:
    st.audio(uploaded_audio, format="audio/wav")
    if st.button("🔍 Transcribe Audio"):
        with st.spinner("Transcribing... Please wait ⏳"):
            result = transcribe_audio(uploaded_audio, model_name=model_choice)
        st.success("✅ Transcription complete!")
        st.subheader("📝 Transcript:")
        st.write(result["text"])
else:
    st.info("Please upload a .wav file to get started.")
