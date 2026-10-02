from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from orbitwatch.config import DATASET_CAVEAT, ExperimentConfig
from orbitwatch.data import available_channels, load_channel, validate_channel
from orbitwatch.detectors import fit_and_score
from orbitwatch.metrics import detection_metrics

DETECTORS = ("rolling_median", "gru")


def source_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def resolve_run(root: Path, run: str | None = None) -> Path:
    if run is None:
        latest = root / "runs" / "latest.json"
        if not latest.is_file():
            raise FileNotFoundError("No completed experiment. Run `orbitwatch train` first.")
        run = json.loads(latest.read_text())["run_id"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", run):
        raise ValueError("Invalid experiment identifier.")
    path = root / "runs" / run
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Experiment not found: {run}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete":
        raise ValueError(f"Experiment {run} is not complete: {manifest.get('status')}.")
    return path


def summarize(metrics: pd.DataFrame) -> list[dict]:
    results = []
    for mission in ["ALL", *sorted(metrics["spacecraft"].unique())]:
        subset = metrics if mission == "ALL" else metrics[metrics["spacecraft"] == mission]
        for detector, group in subset.groupby("detector"):
            normal = int(group["normal_samples"].sum())
            true_events = int(group["true_events"].sum())
            predicted_events = int(group["predicted_events"].sum())
            matched = int(group["matched_events"].sum())
            average_precision = group["average_precision"].mean()
            precision = matched / predicted_events if predicted_events else 0.0
            recall = matched / true_events if true_events else 0.0
            results.append(
                {
                    "spacecraft": mission,
                    "detector": detector,
                    "channels": len(group),
                    "macro_point_f1": float(group["point_f1"].mean()),
                    "macro_average_precision": float(average_precision)
                    if pd.notna(average_precision)
                    else None,
                    "event_precision": precision,
                    "event_recall": recall,
                    "event_f1": 2 * precision * recall / (precision + recall)
                    if precision + recall
                    else 0.0,
                    "false_alarms_per_1000_normal_samples": 1000
                    * int(group["false_positive_samples"].sum())
                    / normal
                    if normal
                    else None,
                    "mean_channel_p95_latency_ms": float(group["p95_window_latency_ms"].mean()),
                    "training_seconds": float(group["training_seconds"].sum()),
                }
            )
    return results


def run_experiment(
    root: Path,
    config: ExperimentConfig,
    channels: list[str] | None = None,
    run_id: str | None = None,
    progress: Callable[[str], None] = print,
) -> Path:
    metadata = available_channels(root / "data")
    selected = channels or metadata["chan_id"].tolist()
    if len(selected) != len(set(selected)):
        raise ValueError("Duplicate channels are not allowed.")
    for channel in selected:
        validate_channel(channel)
        if channel not in set(metadata["chan_id"]):
            raise ValueError(f"Unknown channel: {channel}")
    run_id = run_id or datetime.now(UTC).strftime("nasa-%Y%m%dT%H%M%SZ")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", run_id):
        raise ValueError("Invalid experiment identifier.")
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "run_id": run_id,
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "channels": selected,
        "config": config.model_dump(),
        "device": "CPU",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "pandas", "scikit-learn")
        },
        "source_sha256": source_fingerprint(),
        "dataset": json.loads((root / "data" / "manifest.json").read_text()),
        "limitations": [
            DATASET_CAVEAT,
            "Single-seed measurements are not statistical significance claims.",
            "Model configuration was specified before this evaluation; test labels do not calibrate thresholds.",
            "All computed alarm events are included. No point adjustment or ground-truth-based alarm expansion is used.",
            "Latency is CPU forward-pass timing, not spacecraft hardware qualification.",
        ],
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    rows: list[dict] = []
    started = time.perf_counter()
    for index, channel_id in enumerate(selected, 1):
        progress(f"[{index}/{len(selected)}] Training and evaluating {channel_id}")
        channel = load_channel(root / "data", channel_id)
        output = fit_and_score(channel.train, channel.test, config)
        channel_dir = run_dir / channel_id
        channel_dir.mkdir()
        np.savez_compressed(
            channel_dir / "scores.npz",
            observed=channel.test[:, 0],
            labels=channel.labels,
            rolling_median_prediction=output.baseline_prediction,
            gru_prediction=output.gru_prediction,
            rolling_median_scores=output.baseline_scores,
            gru_scores=output.gru_scores,
        )
        torch.save(output.checkpoint, channel_dir / "gru.pt")
        channel_metadata = {
            "channel": channel_id,
            "spacecraft": channel.spacecraft,
            "features": channel.train.shape[1],
            "train_samples": len(channel.train),
            "fit_samples": output.fit_samples,
            "calibration_samples": output.calibration_samples,
            "test_samples": len(channel.test),
            "warmup_samples": config.window,
            "thresholds": output.thresholds,
            "training_losses": output.training_losses,
        }
        (channel_dir / "metadata.json").write_text(json.dumps(channel_metadata, indent=2))
        for detector, scores, latency in (
            ("rolling_median", output.baseline_scores, output.baseline_p95_latency_ms),
            ("gru", output.gru_scores, output.gru_p95_latency_ms),
        ):
            result = detection_metrics(
                channel.labels[config.window :],
                scores[config.window :],
                output.thresholds[detector],
            )
            rows.append(
                {
                    "channel": channel_id,
                    "spacecraft": channel.spacecraft,
                    "detector": detector,
                    "threshold": output.thresholds[detector],
                    "training_seconds": output.training_seconds if detector == "gru" else 0.0,
                    "p95_window_latency_ms": latency,
                    **result,
                }
            )
        pd.DataFrame(rows).to_csv(run_dir / "metrics.csv", index=False)
    summary = summarize(pd.DataFrame(rows))
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    manifest.update(
        status="complete",
        completed_at=datetime.now(UTC).isoformat(),
        duration_seconds=time.perf_counter() - started,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False))
    latest_temp = root / "runs" / "latest.json.tmp"
    latest_temp.write_text(json.dumps({"run_id": run_id}))
    latest_temp.replace(root / "runs" / "latest.json")
    progress(f"Completed {len(selected)} channels. Results: {run_dir}")
    return run_dir
