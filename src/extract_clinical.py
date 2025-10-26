# src/extract_clinical.py
import re, json, os
from typing import Dict, Any, List, Tuple

# -----------------------
# Model bootstrap (FLAN)
# -----------------------
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

MODEL_NAME = os.environ.get("DOCSCRIBE_MODEL", "google/flan-t5-large")
TOKENIZERS_PARALLELISM = os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    import torch
    DEVICE = 0 if torch.cuda.is_available() else -1
except Exception:
    DEVICE = -1

GEN_KW = dict(
    do_sample=False,
    num_beams=4,
    temperature=0.0,
    max_new_tokens=420,
    early_stopping=True,
)

# Load once when module is imported
_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
_t5 = pipeline("text2text-generation", model=_model, tokenizer=_tokenizer, device=DEVICE)

# -----------------------
# Prompt (schema-only)
# -----------------------
FEWSHOT = """You are a documentation assistant.

Return ONE valid JSON object ONLY. Start with '{' and end with '}'.
Use EXACTLY these keys and types:
- "chief_complaint": string
- "assessment": string
- "diagnosis": array of strings
- "orders": array of strings
- "plan": array of strings
- "follow_up": string

STRICT RULES:
- Derive content ONLY from the TRANSCRIPT text.
- Every value MUST be a verbatim substring of the TRANSCRIPT (case-insensitive allowed).
- If a value is not present, leave it "" (for strings) or [] (for arrays).
- Do NOT add any text before or after the JSON.

TRANSCRIPT:
{transcript}

JSON:
"""

# -----------------------
# Regex / helpers
# -----------------------
KEYS_ORDER = ["chief_complaint","assessment","diagnosis","orders","plan","follow_up"]
KEYS_SET   = set(KEYS_ORDER)
KEY_START_RE = re.compile(r'(?:"?(chief_complaint|assessment|diagnosis|orders|plan|follow_up)"?\s*:)', re.I)

_LEAD_VERBS = re.compile(r"^\s*(?:start|begin|initiate|recommend|advise|continue|order|obtain|get|perform|schedule)\s+", re.IGNORECASE)
_DETERMINERS = re.compile(r"^\s*(?:to|the|a|an)\s+", re.IGNORECASE)

_HEDGES = re.compile(
    r"\b(likely|suspected|suspect|possible|probable|prob|consistent with|r/o|rule out)\b[:\s,-]*",
    re.IGNORECASE,
)

_TIME_RE = re.compile(
    r"\b(?:(?:in\s+)?\d+\s*(?:day|days|week|weeks|wk|wks|month|months)|"
    r"\d+-\d+\s*(?:days|weeks)|"
    r"(?:return if worse|return if worsening|follow up))\b",
    re.IGNORECASE
)

_DOSAGE_PHRASE_RE = re.compile(
    r"\b([A-Za-z][A-Za-z\-]*(?:\s[A-Za-z][A-Za-z\-]*)*\s+"
    r"(?:\d+\s*(?:mg|mcg|g|ml|units)\b(?:\s*(?:daily|q\d+h|BID|TID|QID|PRN))?"
    r"(?:\s*x\d+\s*(?:day|days|week|weeks)?)?))",
    re.IGNORECASE
)

# modality canonicalization
_MODALITY_CANON = {
    r"(?:x[\-\s]?ray|xr|xray)": "X-ray",
    r"(?:ct\s*scan|ct)": "CT",
    r"(?:mri)": "MRI",
    r"(?:ultra\s*sound|us)": "Ultrasound",
    r"(?:ekg|ecg)": "ECG",
    r"(?:echo|echocardiogram)": "Echo",
}

ORDER_VERBS = r"(?:order|obtain|get|perform|schedule)"
PLAN_VERBS  = r"(?:start|begin|initiate|recommend|advise|continue)"

HEURISTICS = {
    "imaging": {"x-ray", "xray", "ct", "mri", "ultrasound", "ekg", "ecg", "echo"},
    "labs": {"cbc", "cmp", "a1c", "bmp", "urinalysis", "culture", "strep test"},
}
MIRROR_MEDS_TO_ORDERS = True  # demo behavior

# -----------------------
# Small utils
# -----------------------
def _gen_text(prompt: str) -> str:
    return _t5(prompt, **GEN_KW)[0]["generated_text"].strip()

def _canonical(s: str) -> str:
    if not s:
        return ""
    x = s.strip().rstrip(".")
    x = _LEAD_VERBS.sub("", x)
    x = _DETERMINERS.sub("", x)
    x = re.sub(r"\s+", " ", x)
    x = re.sub(r"\bx(\d+)\s*day\b", r"x\1 days", x, flags=re.IGNORECASE)
    return x.lower()

def _dehedge(s: str) -> str:
    return _HEDGES.sub("", s or "").strip(" .,:;")

def _loose_contains(transcript: str, phrase: str) -> bool:
    if not phrase:
        return False
    t_raw = (transcript or "").lower()
    p_raw = phrase.strip().lower().rstrip(".")
    if p_raw and p_raw in t_raw:
        return True
    t_can = _canonical(transcript)
    p_can = _canonical(phrase)
    if p_can and p_can in t_can:
        return True
    p_alt = re.sub(r"\bx(\d+)\s*day\b", r"x\1 days", p_raw)
    return p_alt in t_raw

def _extract_time_phrase(text: str) -> str:
    if not text:
        return ""
    m = _TIME_RE.search(text)
    return (m.group(0).strip() if m else "").rstrip(".")

def _extract_dosage_phrases(text: str) -> List[str]:
    return [m.group(1).strip().rstrip(".") for m in _DOSAGE_PHRASE_RE.finditer(text or "")]

def _find_key_spans(txt: str) -> Dict[str, slice]:
    spans, positions = {}, []
    for m in KEY_START_RE.finditer(txt):
        k = m.group(1).lower()
        positions.append((k, m.start(), m.end()))
    for i, (k, s, e) in enumerate(positions):
        nxt = positions[i+1][1] if i+1 < len(positions) else len(txt)
        spans[k] = slice(e, nxt)
    return spans

def _grab_string_val(chunk: str) -> str:
    m = re.search(r'"\s*([^"]*?)\s*"', chunk)  # "value"
    if m: return m.group(1).strip()
    m = re.search(r':\s*([^,\]\}]+)', chunk)    # : value
    return m.group(1).strip() if m else ""

def _grab_list_val(chunk: str) -> List[str]:
    m = re.search(r'\[\s*([^\]]*?)\s*\]', chunk)
    inside = m.group(1) if m else chunk
    items = re.findall(r'"([^"]+)"', inside) or [x.strip() for x in re.split(r'[;,]', inside) if x.strip()]
    cleaned, seen = [], set()
    for it in items:
        it = it.strip()
        if not it or it.lower() in KEYS_SET or len(it) <= 1:
            continue
        low = it.lower()
        if low not in seen:
            seen.add(low); cleaned.append(it)
    return cleaned

def _parse_array(s: str) -> List[str]:
    m = re.search(r"\[[\s\S]*\]", s)
    if m:
        try:
            arr = json.loads(m.group(0))
            if isinstance(arr, list):
                return [x for x in arr if isinstance(x, str)]
        except Exception:
            pass
    items = re.findall(r'"([^"]+)"', s)
    return items or [x.strip() for x in re.split(r"[;,]", s) if x.strip()]

def parse_fields_with_boundaries(raw_txt: str) -> Dict[str, Any]:
    t = (raw_txt or "").replace("“", '"').replace("”", '"').replace("’", "'")
    t = re.sub(r"\s+", " ", t).strip()

    # Try JSON first
    mjson = re.search(r"\{[\s\S]*\}", t)
    if mjson:
        block = mjson.group(0)
        try:
            data = json.loads(block)
            data = {k: v for k, v in data.items() if k in KEYS_SET}
            for k in KEYS_ORDER:
                data.setdefault(k, [] if k in ("diagnosis","orders","plan") else "")
            return data
        except Exception:
            pass

    # Boundary parse
    spans = _find_key_spans(t)
    data: Dict[str, Any] = {k: ([] if k in ("diagnosis","orders","plan") else "") for k in KEYS_ORDER}
    for k in KEYS_ORDER:
        if k not in spans:
            continue
        chunk = t[spans[k]]
        data[k] = _grab_list_val(chunk) if k in ("diagnosis","orders","plan") else _grab_string_val(chunk)
    return data

def _normalize_order_phrase(s: str) -> str:
    """Flip '<modality> <target>' → '<target> <Modality>' and tidy spacing."""
    txt = re.sub(r"\s+", " ", s or "").strip().rstrip(".")
    for mod_re, canon in _MODALITY_CANON.items():
        m = re.match(rf"(?i)^{mod_re}\s+(.+)$", txt)
        if m:
            target = m.group(1).strip()
            if not re.search(rf"(?i)\b{canon}\b", target):
                return f"{target} {canon}".strip()
    return txt

def _split_conjunctions(items: List[str], transcript: str) -> List[str]:
    parts: List[str] = []
    for it in items:
        s = (it or "").strip().rstrip(".")
        if not s:
            continue
        chunks = re.split(r"\b(?:and|then|,|;)\b", s, flags=re.IGNORECASE)
        for c in chunks:
            c2 = c.strip().strip(",;.").rstrip(".")
            if not c2:
                continue
            # keep follow-up phrases OUT of arrays
            low = c2.lower()
            if low.startswith("follow up") or _TIME_RE.search(c2):
                continue
            if _loose_contains(transcript, c2):
                parts.append(c2)

    seen, out = set(), []
    for p in parts:
        key = _canonical(p)
        if key and key not in seen:
            seen.add(key)
            out.append(p)
    return out or [x for x in items if x]

def _clip_action_core(s: str, target: str) -> str:
    txt = _LEAD_VERBS.sub("", (s or "").strip().rstrip("."))
    if target == "orders":
        m = re.search(rf"\b{ORDER_VERBS}\b\s+(.*)$", txt, flags=re.IGNORECASE)
        if m: return m.group(1).strip().rstrip(".")
    elif target == "plan":
        m = re.search(rf"\b{PLAN_VERBS}\b\s+(.*)$", txt, flags=re.IGNORECASE)
        if m: return m.group(1).strip().rstrip(".")
    parts = [p.strip() for p in re.split(r"[.]", txt) if p.strip()]
    return parts[-1] if parts else txt

def _is_dosage_like(s: str) -> bool:
    return bool(_DOSAGE_PHRASE_RE.search(s or ""))

def _keep_minimal(s: str) -> bool:
    n_words = len((s or "").split())
    return n_words <= 12 or _is_dosage_like(s)

def _looks_like_imaging_or_lab(s: str) -> bool:
    w = (s or "").lower()
    return any(tok in w for tok in (HEURISTICS["imaging"] | HEURISTICS["labs"]))

def _route_items(orders: List[str], plan: List[str]) -> Tuple[List[str], List[str]]:
    o2, p2 = [], []
    for it in orders:
        s = (it or "").strip().rstrip(".")
        if not s: continue
        if _looks_like_imaging_or_lab(s):
            o2.append(s)
        elif _is_dosage_like(s):
            if MIRROR_MEDS_TO_ORDERS: o2.append(s)
            p2.append(s)
        else:
            o2.append(s)
    for it in plan:
        s = (it or "").strip().rstrip(".")
        if not s: continue
        if _looks_like_imaging_or_lab(s):
            o2.append(s)
        elif _is_dosage_like(s):
            if MIRROR_MEDS_TO_ORDERS: o2.append(s)
            p2.append(s)
        else:
            p2.append(s)

    def dedup(xs: List[str]) -> List[str]:
        seen, out = set(), []
        for x in xs:
            key = _canonical(x)
            if key and key not in seen:
                seen.add(key); out.append(x.strip().rstrip("."))
        return out

    return dedup(o2), dedup(p2)

def _merge_unique(dst: List[str], src: List[str]) -> List[str]:
    seen = {_canonical(x) for x in dst if x}
    out = [d.strip().rstrip(".") for d in dst if d and _canonical(d)]
    for s in src:
        t = (s or "").strip().rstrip(".")
        key = _canonical(t)
        if t and key and key not in seen:
            seen.add(key)
            out.append(t)
    return out

def _extract_before_rule_out(text: str) -> List[str]:
    out = []
    for m in re.finditer(r"([^\.]*?)\s+to\s+rule\s+out\b", text or "", flags=re.IGNORECASE):
        lhs = m.group(1).strip().rstrip(".")
        chunks = re.split(r"\b(?:and|then|,|;)\b", lhs, flags=re.IGNORECASE)
        out.extend([c.strip().strip(",;.") for c in chunks if c.strip()])
    seen, dedup = set(), []
    for x in out:
        lx = _canonical(x)
        if lx not in seen:
            seen.add(lx); dedup.append(x)
    return dedup

def _post_clean_orders(items: List[str]) -> List[str]:
    """Keep Orders to tests/procedures only; strip meds/therapy artifacts."""
    cleaned = []
    for s in items:
        if not s:
            continue
        s2 = s.strip().rstrip(".")
        # drop med/dosage phrases from Orders (they belong in Plan)
        if _is_dosage_like(s2):
            continue
        # drop conjunction/verb artifacts like "and start ..."
        if re.search(r"^\s*(and\s+start|and\s+begin)\b", s2, re.IGNORECASE):
            continue
        # drop obvious therapy words (go to Plan)
        if re.search(r"\b(rice|rest|ice|compression|elevation|lifestyle|counseling)\b", s2, re.IGNORECASE):
            continue
        cleaned.append(s2)
    return cleaned


def _first_imaging_or_lab_token(s: str) -> str:
    """Return the first imaging/lab token present, else ''."""
    w = (s or "").lower()
    for tok in (HEURISTICS["imaging"] | HEURISTICS["labs"]):
        if tok in w:
            return tok
    return ""


def _split_mixed_test_and_med(s: str) -> List[str]:
    """
    If an item contains BOTH a test (e.g., 'urinalysis', 'x-ray') AND a dosage-like med,
    split into ['<test>', '<med dosing>']. Otherwise return [s].
    """
    s2 = (s or "").strip().rstrip(".")
    if not s2:
        return []

    has_dose = _is_dosage_like(s2)
    test_tok = _first_imaging_or_lab_token(s2)
    if has_dose and test_tok:
        # test fragment: keep the token minimally as the order (e.g., 'urinalysis', 'chest x-ray')
        test_phrase = test_tok
        # lift the first dosage phrase as the med
        doses = _extract_dosage_phrases(s2)
        med_phrase = doses[0] if doses else ""
        out = []
        if test_phrase:
            # small normalization for x-ray side (e.g., 'ankle x-ray' / 'chest x-ray')
            # if there is a body part, leave normalization to existing _normalize_order_phrase
            out.append(test_phrase)
        if med_phrase:
            out.append(med_phrase)
        return [x for x in out if x]
    return [s2] if s2 else []  

ACTION_PATTERNS = [
    ("orders", rf"\b{ORDER_VERBS}\b\s+([^\.]+)"),
    ("plan",   rf"\b{PLAN_VERBS}\b\s+([^\.]+)"),
]

def _derive_actions_from_transcript(transcript: str) -> Dict[str, List[str]]:
    text = re.sub(r"\s+", " ", transcript or "").strip()
    derived = {"orders": [], "plan": []}

    # 1) Verb-led extraction
    for target, pat in ACTION_PATTERNS:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            segment = m.group(1)
            chunks = re.split(r"\b(?:and|then|,|;)\b", segment, flags=re.IGNORECASE)
            for c in chunks:
                c2 = re.sub(r"^\s*(to\s+)", "", c, flags=re.IGNORECASE).strip().rstrip(".")
                if c2:
                    derived[target].append(c2)

    # 2) Dosage phrases → Plan candidates
    for phr in _extract_dosage_phrases(text):
        derived["plan"].append(phr)

    # 3) Before "to rule out ..." → Orders
    for lhs in _extract_before_rule_out(text):
        derived["orders"].append(lhs)

    # De-dup canonical
    for k in derived:
        seen, out = set(), []
        for it in derived[k]:
            key = _canonical(it)
            if key and it:
                if key not in seen:
                    seen.add(key); out.append(it.strip())
        derived[k] = out

    return derived

def _ground_to_transcript(data: Dict[str, Any], transcript: str) -> Dict[str, Any]:
    out = {}
    for k in KEYS_ORDER:
        v = data.get(k, [] if k in ("diagnosis","orders","plan") else "")
        if isinstance(v, list):
            kept, seen = [], set()
            for s in v:
                s2 = (s or "").strip().rstrip(".")
                key = _canonical(s2)
                if s2 and key and key not in seen and _loose_contains(transcript, s2):
                    seen.add(key); kept.append(s2)
            out[k] = kept
        else:
            s2 = (v or "").strip().rstrip(".")
            out[k] = s2 if s2 and _loose_contains(transcript, s2) else ""
    return out

def _raw_output_is_bad_list(raw: str, transcript: str) -> bool:
    s = (raw or "").strip()
    if s.startswith("[") and s.endswith("]") and len(s) < 4000:
        inner = re.sub(r'^\[\s*"?|\s*"?\]$', "", s).strip()
        return len(inner) >= 20 and inner.lower() in (transcript or "").lower()
    return False

FIELD_PROMPTS = {
    "chief_complaint": (
        "From the TRANSCRIPT, return the chief complaint as a verbatim substring.\n"
        "Return ONLY the phrase, no quotes, no extra text. If none, return nothing.\n\n"
        "TRANSCRIPT:\n{transcript}\n\nPHRASE:"
    ),
    "assessment": (
        "From the TRANSCRIPT, return the assessment/impression as a verbatim substring.\n"
        "Return ONLY the phrase, no quotes, no extra text. If none, return nothing.\n\n"
        "TRANSCRIPT:\n{transcript}\n\nPHRASE:"
    ),
    "follow_up": (
        "From the TRANSCRIPT, return ONLY the follow-up timing as a verbatim substring "
        "(e.g., '2 days', '1 week', 'return if worsening'). Do not include medications or 'PRN'. "
        "Return ONLY the phrase, no quotes, no extra text. If none, return nothing.\n\n"
        "TRANSCRIPT:\n{transcript}\n\nPHRASE:"
    ),
    "diagnosis": (
        "From the TRANSCRIPT, list diagnoses as a JSON array of verbatim substrings.\n"
        "Return ONLY the JSON array (e.g., [\"...\"]). If none, return [].\n\n"
        "TRANSCRIPT:\n{transcript}\n\nARRAY:"
    ),
    "orders": (
        "From the TRANSCRIPT, extract tests/procedures/medications that are explicitly ordered "
        "as a JSON array of verbatim substrings (minimal phrases only, e.g., 'chest X-ray', "
        "'azithromycin 500 mg daily x5'). If multiple are in one sentence, split into separate items. "
        "Return ONLY the JSON array. If none, return [].\n\nTRANSCRIPT:\n{transcript}\n\nARRAY:"
    ),
    "plan": (
        "From the TRANSCRIPT, extract planned interventions/instructions as a JSON array of verbatim substrings "
        "(minimal phrases only, e.g., 'RICE', 'ibuprofen 400 mg PRN'). If multiple are in one sentence, split into "
        "separate items. Return ONLY the JSON array. If none, return [].\n\nTRANSCRIPT:\n{transcript}\n\nARRAY:"
    ),
}

# -----------------------
# Refinement pipeline
# -----------------------
def _refine_empty_fields(transcript: str, data: Dict[str, Any]) -> Dict[str, Any]:
    filled = dict(data)

    # Strings
    for k in ["chief_complaint", "assessment"]:
        if not filled.get(k):
            val = _gen_text(FIELD_PROMPTS[k].format(transcript=transcript)).strip()
            filled[k] = val

    # Follow-up normalize (PRN not captured)
    fu = filled.get("follow_up", "")
    if not fu:
        fu = _gen_text(FIELD_PROMPTS["follow_up"].format(transcript=transcript)).strip()
    filled["follow_up"] = _extract_time_phrase(fu)

    # Arrays
    for k in ["diagnosis", "orders", "plan"]:
        arr = filled.get(k, [])
        if not arr:
            raw = _gen_text(FIELD_PROMPTS[k].format(transcript=transcript))
            arr = _parse_array(raw)

        # generic split (and/then/,/;)
        arr = _split_conjunctions(arr, transcript)

        # strip lead verbs & normalize modality phrasing for orders/plan
        if k in ("orders", "plan"):
            arr = [_clip_action_core(x, k) for x in arr]
            arr = [_normalize_order_phrase(x) for x in arr]

        # de-hedge diagnosis labels
        if k == "diagnosis":
            arr = [_dehedge(x) for x in arr if _dehedge(x)]

        # de-dup & minimality
        seen, clean = set(), []
        for it in arr:
            s = (it or "").strip().rstrip(".")
            if not s:
                continue
            if k in ("orders", "plan") and not _keep_minimal(s):
                continue
            key = _canonical(s)
            if key and key not in seen:
                seen.add(key)
                clean.append(s)
        filled[k] = clean

    # De-hedge again (safety)
    filled["diagnosis"] = [_dehedge(x) for x in filled.get("diagnosis", []) if _dehedge(x)]

    # Always derive & merge
    derived = _derive_actions_from_transcript(transcript)
    filled["orders"] = _merge_unique(filled.get("orders", []), derived.get("orders", []))
    filled["plan"]   = _merge_unique(filled.get("plan",   []), derived.get("plan",   []))

    # Prune Plan noise
    pruned_plan = []
    for s in filled.get("plan", []):
        s2 = s.strip()
        if s2.count(".") > 0:
            continue
        if re.search(r"\blikely\b|\border\b", s2, re.IGNORECASE):
            continue
        pruned_plan.append(s2)
    filled["plan"] = pruned_plan

    # Canonical routing
    filled["orders"], filled["plan"] = _route_items(
        filled.get("orders", []), filled.get("plan", [])
    )

    # Ground to transcript
    return _ground_to_transcript(filled, transcript)

# -----------------------
# Public entry point
# -----------------------
def extract_note(transcript: str, gen_kwargs: Dict[str, Any] = GEN_KW) -> Tuple[Dict[str, Any], str]:
    """
    Returns (note_dict, raw_model_output)
    note_dict has keys:
      chief_complaint (str), assessment (str), diagnosis (list[str]),
      orders (list[str]), plan (list[str]), follow_up (str)
    """
    # Pass A — schema-only prompt
    prompt = FEWSHOT.replace("{transcript}", (transcript or "").strip())
    result = _t5(prompt, **gen_kwargs)[0]
    raw = result["generated_text"]

    # If raw is "bad list", force empty so backoff fully runs
    if _raw_output_is_bad_list(raw, transcript):
        data = {k: ([] if k in ("diagnosis","orders","plan") else "") for k in KEYS_ORDER}
    else:
        data = _ground_to_transcript(parse_fields_with_boundaries(raw), transcript)

    # Pass B — refine + salvage + routing + grounding
    filled = _refine_empty_fields(transcript, data)

    # Final cleaning
    out: Dict[str, Any] = {
        "chief_complaint": filled.get("chief_complaint", "").strip(),
        "assessment": filled.get("assessment", "").strip(),
        "diagnosis": [x.strip() for x in filled.get("diagnosis", []) if x and x.strip()],
        "orders": [x.strip() for x in filled.get("orders", []) if x and x.strip()],
        "plan": [x.strip() for x in filled.get("plan", []) if x and x.strip()],
        "follow_up": filled.get("follow_up", "").strip(),
    }
    return out, raw