from typing import Tuple, Dict, Any

def _to_dict(note) -> Dict[str, Any]:
    if isinstance(note, dict):
        return note
    if hasattr(note, "dict"):
        return note.dict()
    fields = ["chief_complaint","assessment","diagnosis","orders","plan","follow_up"]
    return {k: getattr(note, k, "" if k in ("chief_complaint","assessment","follow_up") else []) for k in fields}

def compose_note(note) -> Tuple[str, str]:
    data = _to_dict(note)
    s = data.get("chief_complaint") or "—"
    o = ", ".join(data.get("orders") or []) or "—"
    a = data.get("assessment") or (", ".join(data.get("diagnosis") or []) or "—")
    p = "; ".join(data.get("plan") or []) or "—"
    f = data.get("follow_up") or "—"
    soap = f"S: {s}\nO: {o}\nA: {a}\nP: {p}\nFollow-up: {f}"
    summary = f"Visit summary: {s}. Assessment: {a}. Plan: {p}. Follow-up: {f}."
    return soap, summary
