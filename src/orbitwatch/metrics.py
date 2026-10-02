from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support


def intervals(mask: np.ndarray, offset: int = 0) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1:
        raise ValueError("Event mask must be one-dimensional.")
    changes = np.diff(np.r_[False, values, False].astype(np.int8))
    return [
        (int(start + offset), int(end - 1 + offset))
        for start, end in zip(
            np.flatnonzero(changes == 1), np.flatnonzero(changes == -1), strict=True
        )
    ]


def detection_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    labels, scores = np.asarray(labels), np.asarray(scores)
    if labels.shape != scores.shape or labels.ndim != 1:
        raise ValueError("Labels and scores must be matching one-dimensional arrays.")
    if not np.isfinite(scores).all() or not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Evaluation requires finite scores and a positive finite threshold.")
    if not np.isin(labels, [0, 1]).all() or len(labels) == 0:
        raise ValueError("Evaluation requires nonempty binary labels.")
    predicted = scores > threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted, average="binary", zero_division=0
    )
    actual_events, predicted_events = intervals(labels), intervals(predicted)
    unmatched = set(range(len(actual_events)))
    delays: list[int] = []
    for start, end in predicted_events:
        matches = [
            i
            for i in sorted(unmatched)
            if actual_events[i][0] <= end and actual_events[i][1] >= start
        ]
        if matches:
            match = matches[0]
            unmatched.remove(match)
            delays.append(max(0, start - actual_events[match][0]))
    hits = len(delays)
    event_precision = hits / len(predicted_events) if predicted_events else 0.0
    event_recall = hits / len(actual_events) if actual_events else 0.0
    event_f1 = (
        2 * event_precision * event_recall / (event_precision + event_recall)
        if event_precision + event_recall
        else 0.0
    )
    normal = labels == 0
    false_positive_points = int(np.sum(predicted & normal))
    return {
        "point_precision": float(precision),
        "point_recall": float(recall),
        "point_f1": float(f1),
        "average_precision": float(average_precision_score(labels, scores))
        if np.any(labels)
        else None,
        "event_precision": event_precision,
        "event_recall": event_recall,
        "event_f1": event_f1,
        "true_events": len(actual_events),
        "predicted_events": len(predicted_events),
        "matched_events": hits,
        "mean_delay_samples": float(np.mean(delays)) if delays else None,
        "normal_samples": int(normal.sum()),
        "false_positive_samples": false_positive_points,
        "false_alarms_per_1000_normal_samples": (
            1000 * false_positive_points / int(normal.sum()) if normal.any() else None
        ),
        "evaluated_samples": len(labels),
        "anomalous_samples": int(np.sum(labels)),
    }
