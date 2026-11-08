"""Evidence linking: record where each extracted value appears in the transcript."""
import html
import re
from typing import Any, Dict, List, Optional


def _find_span(transcript: str, value: str) -> Optional[Dict[str, Any]]:
    """Find a value in the transcript (case-insensitive, flexible whitespace)."""
    v = (value or "").strip().rstrip(".")
    if not v:
        return None
    pattern = r"\s+".join(re.escape(w) for w in v.split())
    m = re.search(pattern, transcript, flags=re.IGNORECASE)
    if not m:
        return None
    return {"start": m.start(), "end": m.end(), "text": transcript[m.start():m.end()]}


def build_evidence(note: Dict[str, Any], transcript: str) -> Dict[str, List[Dict[str, Any]]]:
    """For every extracted value, return its character span in the transcript (None if no exact match)."""
    evidence: Dict[str, List[Dict[str, Any]]] = {}
    for field, val in note.items():
        values = val if isinstance(val, list) else [val]
        spans = []
        for v in values:
            if not v:
                continue
            span = _find_span(transcript, v)
            spans.append({"value": v, **(span or {"start": None, "end": None, "text": None})})
        evidence[field] = spans
    return evidence


def highlight_html(transcript: str, evidence: Dict[str, List[Dict[str, Any]]]) -> str:
    """Return the transcript as HTML with each evidence span wrapped in <mark>, labelled by field."""
    spans = sorted({(s["start"], s["end"], f)
                    for f, items in evidence.items() for s in items if s["start"] is not None})
    out, pos = [], 0
    for start, end, field in spans:
        if start < pos:  # skip overlapping spans
            continue
        out.append(html.escape(transcript[pos:start]))
        out.append(f'<mark title="{field}">{html.escape(transcript[start:end])}</mark>')
        pos = end
    out.append(html.escape(transcript[pos:]))
    return "".join(out)
