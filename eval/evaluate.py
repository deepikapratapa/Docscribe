"""Evaluate DocScribe extraction against gold labels: per-field precision, recall and F1.

Usage (from the repo root):  python -m eval.evaluate
"""
import json
import re
import sys
from pathlib import Path

FIELDS = ["chief_complaint", "assessment", "diagnosis", "orders", "plan", "follow_up"]


def _norm(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(x).lower()).strip()


def field_prf(pred, gold):
    """Exact match after normalization. Returns (p, r, f1), or None when both sides are empty."""
    to_set = lambda v: {_norm(x) for x in (v if isinstance(v, list) else [v]) if str(x).strip()}
    p_set, g_set = to_set(pred), to_set(gold)
    if not p_set and not g_set:
        return None                      # nothing to extract and nothing predicted: not scored
    hit = len(p_set & g_set)
    p = hit / len(p_set) if p_set else 0.0
    r = hit / len(g_set) if g_set else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def evaluate(preds, golds, count_empty_as_correct=False):
    """Return per-case mean F1 and per-field mean P/R/F1."""
    per_field = {f: [] for f in FIELDS}
    case_f1 = []
    for pred, gold in zip(preds, golds):
        f1s = []
        for f in FIELDS:
            s = field_prf(pred.get(f, []), gold.get(f, []))
            if s is None:
                if not count_empty_as_correct:
                    continue
                s = (1.0, 1.0, 1.0)
            per_field[f].append(s)
            f1s.append(s[2])
        case_f1.append(sum(f1s) / len(f1s) if f1s else 0.0)
    field_means = {f: tuple(round(sum(x[i] for x in v) / len(v), 2) for i in range(3))
                   for f, v in per_field.items() if v}
    return case_f1, field_means


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from src.extract_clinical import extract_note

    cases = json.loads((root / "eval" / "gold_validation.json").read_text())
    preds = [extract_note(c["transcript"])[0] for c in cases]
    golds = [c["gold"] for c in cases]

    for label, flag in [("Both-empty fields counted as correct (original)", True),
                        ("Both-empty fields not scored (stricter)", False)]:
        case_f1, fields = evaluate(preds, golds, count_empty_as_correct=flag)
        print(f"\n== {label} ==")
        for i, f1 in enumerate(case_f1, 1):
            print(f"case {i}: mean F1 = {f1:.2f}")
        print(f"average F1 = {sum(case_f1) / len(case_f1):.2f}")
        print(f"{'field':16s}  P     R     F1")
        for f, (p, r, f1) in fields.items():
            print(f"{f:16s} {p:.2f}  {r:.2f}  {f1:.2f}")

    (root / "eval" / "predictions.json").write_text(json.dumps(preds, indent=2))


if __name__ == "__main__":
    main()
