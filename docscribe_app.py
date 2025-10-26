# app.py
import io
import json
import time
import tempfile
import re
from typing import Dict, Any, List, Tuple

import streamlit as st

# --- Optional mic recorder (graceful fallback if missing) ---
try:
    from audio_recorder_streamlit import audio_recorder
except Exception:
    audio_recorder = None

# --- Optional ASR (keep if you already have it) ---
try:
    from src.asr_whisper import transcribe_audio
except Exception:
    transcribe_audio = None

# --- Clinical extractor + composer ---
from src.extract_clinical import extract_note  # self-contained module
from src.compose_note import compose_note

st.set_page_config(page_title="DocScribe", page_icon="🩺", layout="centered")
st.title("🩺 DocScribe — Speech-to-Note")

# -------------------------
# Caching & warm-up helpers
# -------------------------
@st.cache_data(show_spinner=False)
def _warm_up() -> float:
    t0 = time.time()
    _ = extract_note("hello")  # warms model; returns empty-ish note
    return round(time.time() - t0, 2)

lat = _warm_up()
with st.expander("ℹ️ Model status", expanded=False):
    st.write(f"Extractor warmed in ~{lat}s (cached).")

# -------------------------
# Input: audio upload / live record / text
# -------------------------
tab1, tab2, tab3 = st.tabs(["📤 Upload .wav", "🎙️ Record Live", "⌨️ Text"])

# ===== Upload .wav =====
with tab1:
    uploaded_audio = st.file_uploader("Choose a .wav file", type=["wav"])
    if uploaded_audio is not None:
        st.audio(uploaded_audio, format="audio/wav")
        if transcribe_audio is None:
            st.warning("ASR module not found. Switch to the Text tab or add `src/asr_whisper.py`.")
        else:
            model_choice = st.selectbox(
                "Whisper model:",
                ["tiny", "base", "small", "medium", "large"],
                index=1,
                help="Smaller = faster, larger = more accurate.",
            )
            if st.button("🔍 Transcribe (Upload)"):
                with st.spinner("Transcribing…"):
                    asr = transcribe_audio(uploaded_audio, model_name=model_choice)
                st.success("✅ Transcription complete")
                st.session_state["transcript_text"] = asr.get("text", "").strip()

# ===== Record Live =====
with tab2:
    if audio_recorder is None:
        st.info(
            "Install live mic component:\n\n"
            "`pip install audio-recorder-streamlit`\n\n"
            "Also ensure FFmpeg is installed for Whisper."
        )
    else:
        st.caption("Click to start/stop. Auto-stops after a short pause.")
        audio_bytes = audio_recorder(
            text="Click to record",
            recording_color="#e74c3c",
            neutral_color="#2ecc71",
            icon_name="microphone",
            icon_size="3x",
            sample_rate=16000,
            pause_threshold=2.0,
        )
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            if transcribe_audio is None:
                st.warning("ASR module not found. Switch to the Text tab or add `src/asr_whisper.py`.")
            else:
                model_choice_rec = st.selectbox(
                    "Whisper model (recording):",
                    ["tiny", "base", "small", "medium", "large"],
                    index=1,
                    key="rec_model_select",
                )
                if st.button("🔍 Transcribe (Recording)"):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(audio_bytes)
                        temp_wav = tmp.name
                    with st.spinner("Transcribing…"):
                        asr = transcribe_audio(temp_wav, model_name=model_choice_rec)
                    st.success("✅ Transcription complete")
                    st.session_state["transcript_text"] = asr.get("text", "").strip()

# ===== Text =====
with tab3:
    default_text = st.session_state.get("transcript_text", "")
    txt = st.text_area(
        "Paste/enter transcript",
        value=default_text,
        height=180,
        placeholder="e.g. Fever and cough for 3 days. Mild SOB. Likely CAP. Order chest X-ray…",
    )
    if st.button("🧠 Extract note from text", type="primary"):
        st.session_state["transcript_text"] = txt

# -------------------------
# Run extraction if we have text
# -------------------------
transcript = st.session_state.get("transcript_text", "").strip()
if transcript:
    st.markdown("### 🧪 Extraction")
    with st.spinner("Extracting structured note…"):
        t0 = time.time()
        note_dict, raw = extract_note(transcript)
        soap, summary = compose_note(note_dict)
        dt = round(time.time() - t0, 2)

    colA, colB = st.columns(2)
    with colA:
        st.markdown("#### 🧾 SOAP")
        st.code(soap)
    with colB:
        st.markdown("#### 🗒️ Visit summary")
        st.write(summary)
        st.caption(f"⏱ Latency: ~{dt}s")

    st.markdown("#### 📦 JSON")
    st.code(json.dumps(note_dict, indent=2))

    # -------------------------
    # Downloads
    # -------------------------
    soap_bytes = io.BytesIO(soap.encode("utf-8"))
    json_bytes = io.BytesIO(json.dumps(note_dict, indent=2).encode("utf-8"))
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("⬇️ Download SOAP (.txt)", data=soap_bytes, file_name="soap.txt", mime="text/plain")
    with col2:
        st.download_button("⬇️ Download JSON (.json)", data=json_bytes, file_name="note.json", mime="application/json")

    # -------------------------
    # Quick inline edits → re-compose
    # -------------------------
    with st.expander("✏️ Quick edit and re-compose"):
        cc = st.text_input("Chief complaint", note_dict.get("chief_complaint",""))
        a  = st.text_input("Assessment", note_dict.get("assessment",""))
        d  = st.text_area("Diagnosis (one per line)", "\n".join(note_dict.get("diagnosis", [])))
        o  = st.text_area("Orders (one per line)", "\n".join(note_dict.get("orders", [])))
        p  = st.text_area("Plan (one per line)", "\n".join(note_dict.get("plan", [])))
        fu = st.text_input("Follow-up", note_dict.get("follow_up",""))
        if st.button("🔁 Re-compose"):
            edited = {
                "chief_complaint": cc,
                "assessment": a,
                "diagnosis": [x.strip() for x in d.splitlines() if x.strip()],
                "orders":    [x.strip() for x in o.splitlines() if x.strip()],
                "plan":      [x.strip() for x in p.splitlines() if x.strip()],
                "follow_up": fu,
            }
            e_soap, e_summary = compose_note(edited)
            st.markdown("##### Updated SOAP")
            st.code(e_soap)
            st.markdown("##### Updated summary")
            st.write(e_summary)

# -------------------------
# Tiny in-app metrics (3 demo cases)
# -------------------------
def _normalize_text(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (x or "").lower()).strip()

def field_overlap_score(pred, gold) -> Tuple[float,float,float]:
    if isinstance(pred, list) and isinstance(gold, list):
        pset = {_normalize_text(x) for x in pred if x}
        gset = {_normalize_text(x) for x in gold if x}
    else:
        pset = {_normalize_text(pred)} if pred else set()
        gset = {_normalize_text(gold)} if gold else set()
    if not gset and not pset: return (1.0, 1.0, 1.0)
    if not gset or not pset:  return (0.0, 0.0, 0.0)
    inter = len(pset & gset)
    prec = inter / max(len(pset), 1)
    rec  = inter / max(len(gset), 1)
    f1   = 2*prec*rec / max((prec+rec), 1e-9)
    return (prec, rec, f1)

gold_refs = [
    {
        "chief_complaint": "Fever and cough",
        "assessment": "Likely CAP",
        "diagnosis": ["CAP"],
        "orders": ["chest X-ray"],
        "plan": ["azithromycin 500 mg daily x5"],
        "follow_up": "2 days",
    },
    {
        "chief_complaint": "Left ankle pain",
        "assessment": "Likely lateral ankle sprain",
        "diagnosis": ["lateral ankle sprain"],
        "orders": ["ankle X-ray"],
        "plan": ["RICE", "ibuprofen 400 mg PRN"],
        "follow_up": "",
    },
    {
        "chief_complaint": "Dysuria",
        "assessment": "Uncomplicated UTI",
        "diagnosis": ["uncomplicated UTI"],
        "orders": ["Urinalysis"],
        "plan": ["nitrofurantoin 100 mg BID x5 days"],
        "follow_up": "2 days",
    },
]

st.markdown("---")
if st.checkbox("📊 Show quick metrics on demo cases"):
    demos = [
        "Fever and cough for 3 days. Mild shortness of breath. Likely CAP. Order chest X-ray and start azithromycin 500 mg daily x5. Follow up in 2 days.",
        "Left ankle pain after inversion injury yesterday. Likely lateral ankle sprain. X-ray ankle to rule out fracture. RICE and ibuprofen 400 mg PRN.",
        "Dysuria and urinary frequency for 2 days. No fever or flank pain. Likely uncomplicated UTI. Urinalysis and nitrofurantoin 100 mg BID x5 days.",
    ]
    rows = []
    for i, demo in enumerate(demos):
        pred, _ = extract_note(demo)
        g = gold_refs[i]
        row = {"case": i+1}
        for k in ["chief_complaint","assessment","diagnosis","orders","plan","follow_up"]:
            _,_,f1 = field_overlap_score(pred.get(k, []), g.get(k, []))
            row[k] = round(f1, 2)
        rows.append(row)
    st.table(rows)

# Footer
st.caption("DocScribe • Whisper ASR + FLAN extraction • For demo use only (not medical advice).")