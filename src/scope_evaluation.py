"""Optional scope-router evaluation for labeled in-scope/out-of-scope test cases.

The current repository does not ship an OOD/scope-labeled benchmark, so these
metrics are provided as reusable evaluation support rather than fabricated
results. Each row must contain ``expected_in_scope`` and ``predicted_in_scope``.
"""

from __future__ import annotations


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def evaluate_scope_predictions(rows: list[dict]) -> dict:
    if not rows:
        return {
            "Scope Accuracy": 0.0,
            "In-Scope Recall": 0.0,
            "Out-of-Scope Detection Recall": 0.0,
            "Scope Macro F1": 0.0,
            "Over-Refusal Rate": 0.0,
            "Under-Refusal Rate": 0.0,
            "Evaluated Scope Cases": 0,
        }

    tp = tn = fp = fn = 0
    for row in rows:
        expected = bool(row["expected_in_scope"])
        predicted = bool(row["predicted_in_scope"])
        if expected and predicted:
            tp += 1
        elif not expected and not predicted:
            tn += 1
        elif not expected and predicted:
            fp += 1
        else:
            fn += 1

    in_precision = _safe_div(tp, tp + fp)
    in_recall = _safe_div(tp, tp + fn)
    in_f1 = _safe_div(2 * in_precision * in_recall, in_precision + in_recall)

    out_precision = _safe_div(tn, tn + fn)
    out_recall = _safe_div(tn, tn + fp)
    out_f1 = _safe_div(2 * out_precision * out_recall, out_precision + out_recall)

    total = len(rows)
    return {
        "Scope Accuracy": round((tp + tn) / total, 4),
        "In-Scope Recall": round(in_recall, 4),
        "Out-of-Scope Detection Recall": round(out_recall, 4),
        "Scope Macro F1": round((in_f1 + out_f1) / 2, 4),
        "Over-Refusal Rate": round(_safe_div(fn, tp + fn), 4),
        "Under-Refusal Rate": round(_safe_div(fp, tn + fp), 4),
        "Evaluated Scope Cases": total,
    }
