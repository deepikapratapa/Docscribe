# 🧠 DocScribe — Voice to Structured Clinical Notes  
**Gator Hack IV | AI in Healthcare**

> “When clinicians speak, the record writes itself.”  
> *DocScribe* transforms clinician voice input into structured medical documentation — generating clean, explainable, and EMR-ready notes in seconds.

---

## 🌟 Overview
Healthcare professionals spend up to **40–50%** of their workday documenting care.  
**DocScribe** reduces this burden by using AI to automatically:
- 🎙️ Transcribe spoken diagnostic reasoning (via Whisper)  
- 🧩 Extract structured fields like diagnosis, orders, and plan (via Flan-T5)  
- 📋 Generate standardized SOAP notes and patient summaries  
- 🔍 Highlight transcript phrases that support each section  
- 📤 Export ready-to-review notes in JSON, Markdown, or PDF  


---

## 🧱 Architecture

```text
🎤 Voice Input (Clinician Dictation)
     ↓ Whisper (ASR)
📝 Transcript (editable)
     ↓ Flan-T5 Extraction (Few-shot prompt)
{ chief_complaint, assessment, diagnosis[], orders[], plan[], follow_up }
     ↓
📋 Note Composer (SOAP + Patient Summary)
     ↓
🔍 Span Highlighter → Traceable Output
     ↓
⬇ Export (JSON / PDF / Markdown)


## 🧠 Core Features

| Feature | Description |
|----------|--------------|
| 🎙️ **Speech-to-Text** | Whisper converts spoken dictation into text |
| 🧩 **Structured Extraction** | LLM extracts JSON with clear clinical fields |
| 📋 **SOAP Note Generator** | Automatically formats clinician notes (S/O/A/P sections) |
| 🔍 **Explainable Output** | Highlights transcript phrases used in each note section |
| 📤 **Exports** | Generate JSON, Markdown, and PDF (with safety disclaimer) |
| ⚖️ **Responsible AI** | Guardrails prevent hallucinated medications or diagnoses |

## 📚 Datasets

| Role | Dataset | Description | Source |
|------|----------|--------------|---------|
| **Primary** | [`abisee/medical_dialogue`](https://huggingface.co/datasets/abisee/medical_dialogue) | Doctor–patient dictations with assessments & plans | Hugging Face |
| **Formatting** | [`medical_meadow/clinical_notes_synth`](https://huggingface.co/datasets/medical_meadow/clinical_notes_synth) | Synthetic SOAP/EMR-style notes | Hugging Face |
| **Optional** | [`openlifescienceai/medmcqa`](https://huggingface.co/datasets/openlifescienceai/medmcqa) | Clinical reasoning Q&A for decision support | Hugging Face |

## ⚙️ Installation

Follow the steps below to set up **DocScribe** on your system.  
Supports both **macOS** 🖥️ and **Windows** 💻 environments.

---

### 🧩 Step 1 — Clone the Repository
```bash
git clone https://github.com/<your-username>/docscribe.git
cd docscribe
```
### 🍏 macOS Setup

1. Create a Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```
2. Install Dependencies
```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets accelerate pydantic streamlit openai-whisper reportlab sounddevice numpy
```

3. Install FFmpeg (required for Whisper)
```bash
brew install ffmpeg
```
💡 If you don’t have Homebrew installed:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
🪟 Windows Setup
1. Create a Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install Dependencies
```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets accelerate pydantic streamlit openai-whisper reportlab sounddevice numpy
```

3. Install FFmpeg (required for Whisper)
	1.	Download FFmpeg from the official website: https://ffmpeg.org/download.html
	2.	Extract it and copy the path to the bin folder.
	3.	Add that path to your System Environment Variables → PATH.
	4.	Restart your terminal or VS Code before continuing.
	
### Step 4 — Verify Installation	
```bash
python -c "import torch, whisper; print('PyTorch:', torch.__version__); print('Whisper OK ✅')"
```
Expected Output:
```bash
PyTorch: <version>
Whisper OK ✅
```
If you see this message, your environment is ready!

### Step 5 — Run DocScribe

Test the clinical extractor and composer:
```bash
python src/extract_clinical.py
python src/compose_note.py
```
Launch the Streamlit App (after UI integration):
That’s it! 🎉
You now have a fully working environment for DocScribe

🧩 Quick Start (Command Line Demo)

Run the clinical extractor + SOAP composer from your terminal:
```bash
python -c "from src.extract_clinical import extract_info; \
from src.compose_note import compose_note; \
note = extract_info('Fever and cough for 3 days. Suspect pneumonia. Order chest X-ray and start azithromycin.'); \
print(compose_note(note)[0])"
```

Expected Output:
```bash
S: Fever and cough for 3 days.
O: Chest X-ray.
A: Suspected pneumonia (Pneumonia).
P: Start azithromycin 500 mg daily.
Follow-up: Re-evaluate in 2 days.
```

### 💻 Run the Streamlit App
```bash
streamlit run app.py
```
You’ll see:
	•	🎙️ Audio upload or record
	•	🧩 Structured JSON
	•	📋 SOAP Note + Patient Summary
	•	🔍 Highlighted transcript phrases
	•	📤 Export buttons
	
## 🧪 Evaluation (Example)

| Metric | Result | Notes |
|---------|--------|-------|
| **Diagnosis F1** | 0.88 | Evaluated on 15 test cases |
| **Orders F1** | 0.85 | From synthetic transcripts |
| **Latency** | 4.3 s | End-to-end: Audio → JSON → SOAP |
| **Hallucination Rate** | 0 % | Guardrails successfully applied |

## 📁 Project Structure
```
docscribe/
├─ app.py                      # Streamlit UI 
├─ prompts/
│  ├─ extractor_fewshot.md     # Few-shot examples for extraction
│  └─ soap_fewshot.md          # SOAP layout exemplars
├─ src/
│  ├─ extract_clinical.py      # LLM extractor
│  ├─ compose_note.py          # SOAP & summary composer
│  ├─ schema.py                # Pydantic schema
│  ├─ asr_whisper.py           # Audio transcription 
│  └─ highlight_spans.py       # Keyword highlighter 
├─ data/
│  └─ samples_audio/           # Demo recordings (.wav)
├─ eval/
│  ├─ eval_transcripts.jsonl   # Evaluation set
│  └─ run_eval.py              # F1 scoring script
└─ README.md                   # This file
```

🧩 Example Audio Scripts

1️⃣ Pneumonia

“Fever and cough for three days with mild shortness of breath. I suspect community-acquired pneumonia. Order chest X-ray and start azithromycin five hundred milligrams daily. Follow-up in two days.”

2️⃣ Ankle Sprain

“Left ankle pain after inversion injury yesterday. Likely lateral ankle sprain. X-ray ankle to rule out fracture. RICE and ibuprofen four hundred milligrams as needed.”

3️⃣ UTI

“Dysuria and urinary frequency for two days. No fever or flank pain. Likely uncomplicated UTI. Urinalysis and nitrofurantoin one hundred milligrams twice daily for five days.”

🎯 Roadmap
	•	Audio → Transcript pipeline
	•	JSON extraction with Flan-T5
	•	SOAP note composer
	•	Streamlit UI (in progress)
	•	Span-level highlighting
	•	ICD-10 auto-coding (top 50)
	•	Bilingual mode (EN ↔ ES)


⚖️ Ethics & Limitations
	•	No real patient data used.
	•	Outputs are drafts, not clinical decisions.
	•	Always review and verify before use in practice.
	•	For research and hackathon demonstration only.

👥 Team Roles
| Member | Role | Focus |
|---------|------|--------|
| **[Deepika Sarala Pratapa]** | AI & Clinical Intelligence Lead | Prompt design, LLM extraction, evaluation |
| **[Rohit Bogulla]** | Full-Stack & UI/UX Lead | Whisper integration, Streamlit design, highlighting, exports |

📜 License

MIT License © 2025 THEDIVERGENTS
For academic and research use only.

⸻

🩺 Acknowledgments
	•	Hugging Face Datasets: abisee/medical_dialogue, medical_meadow/clinical_notes_synth
	•	Models: google/flan-t5-base, openai/whisper-tiny
	•	Hackathon: Gator Hack IV – AI Days, University of Florida













