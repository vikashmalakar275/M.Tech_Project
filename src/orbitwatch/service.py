from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from orbitwatch.data import validate_channel
from orbitwatch.evidence import EvidencePacket, evidence_identifier
from orbitwatch.experiment import DETECTORS, resolve_run
from orbitwatch.metrics import intervals


class TelemetryService:
    def __init__(self, root: Path, run: str | None = None):
        self.root = root
        self.run_dir = resolve_run(root, run)
        self.manifest = json.loads((self.run_dir / "manifest.json").read_text())

    def list_channels(self) -> list[dict]:
        return [self.channel_metadata(channel) for channel in self.manifest["channels"]]

    def channel_metadata(self, channel: str) -> dict:
        validate_channel(channel)
        if channel not in self.manifest["channels"]:
            raise ValueError(f"Channel {channel} was not evaluated in this run.")
        return json.loads((self.run_dir / channel / "metadata.json").read_text())

    def arrays(self, channel: str) -> dict[str, np.ndarray]:
        self.channel_metadata(channel)
        with np.load(self.run_dir / channel / "scores.npz", allow_pickle=False) as loaded:
            return {name: loaded[name] for name in loaded.files}

    @staticmethod
    def validate_detector(detector: str) -> None:
        if detector not in DETECTORS:
            raise ValueError(f"Detector must be one of {DETECTORS}.")

    def telemetry_window(self, channel: str, start: int, end: int, detector: str) -> dict:
        self.validate_detector(detector)
        arrays = self.arrays(channel)
        if not 0 <= start <= end < len(arrays["observed"]):
            raise ValueError("Window indices must be within this channel's test sequence.")
        if end - start + 1 > 1024:
            raise ValueError(
                "MCP windows are limited to 1024 observations; request smaller windows."
            )
        prediction = arrays[f"{detector}_prediction"][start : end + 1]
        scores = arrays[f"{detector}_scores"][start : end + 1]
        return {
            "channel": channel,
            "detector": detector,
            "start_sample": start,
            "end_sample": end,
            "observed": arrays["observed"][start : end + 1].tolist(),
            "predicted": [float(x) if np.isfinite(x) else None for x in prediction],
            "scores": [float(x) if np.isfinite(x) else None for x in scores],
            "threshold": self.channel_metadata(channel)["thresholds"][detector],
            "time_unit": "sample index; absolute mission timestamps unavailable",
        }

    def events(self, channel: str, detector: str = "gru", until: int | None = None) -> list[dict]:
        self.validate_detector(detector)
        arrays = self.arrays(channel)
        metadata = self.channel_metadata(channel)
        until = len(arrays["observed"]) - 1 if until is None else until
        if not 0 <= until < len(arrays["observed"]):
            raise ValueError("Replay cursor is outside the test sequence.")
        threshold = metadata["thresholds"][detector]
        scores = arrays[f"{detector}_scores"][: until + 1]
        result = []
        for start, end in intervals(scores > threshold):
            peak = start + int(np.argmax(scores[start : end + 1]))
            result.append(
                {
                    "start": start,
                    "end": end,
                    "duration": end - start + 1,
                    "peak_sample": peak,
                    "peak_ratio": float(scores[peak] / threshold),
                    "ongoing_at_cursor": end == until,
                }
            )
        return result

    def evidence(
        self, channel: str, start: int, end: int, detector: str = "gru", until: int | None = None
    ) -> EvidencePacket:
        self.validate_detector(detector)
        arrays = self.arrays(channel)
        metadata = self.channel_metadata(channel)
        until = len(arrays["observed"]) - 1 if until is None else until
        if not metadata["warmup_samples"] <= start <= end <= until < len(arrays["observed"]):
            raise ValueError("Evidence must be inside the visible, post-warmup test sequence.")
        threshold = metadata["thresholds"][detector]
        scores = arrays[f"{detector}_scores"]
        if not np.all(scores[start : end + 1] > threshold):
            raise ValueError("An alert evidence interval must contain only above-threshold scores.")
        peak = start + int(np.argmax(scores[start : end + 1]))
        facts = {
            "channel": channel,
            "start_sample": start,
            "end_sample": end,
            "peak_sample": peak,
            "peak_ratio": round(float(scores[peak] / threshold), 6),
            "peak_observed": round(float(arrays["observed"][peak]), 6),
            "peak_expected": round(float(arrays[f"{detector}_prediction"][peak]), 6),
            "physical_cause": "unknown",
        }
        return EvidencePacket(
            evidence_id=evidence_identifier(self.run_dir.name, channel, detector, start, end),
            run_id=self.run_dir.name,
            spacecraft=metadata["spacecraft"],
            detector=detector,
            facts=facts,
            threshold=threshold,
            peak_score=float(scores[peak]),
            duration_samples=end - start + 1,
            source_sha256=self.manifest["source_sha256"],
            ongoing_at_cursor=end == until,
            limitations=[
                "Dataset-scaled values have no published physical units.",
                "The score/threshold ratio is not a probability or calibrated fault severity.",
                "Physical cause is unknown; correlations and model residuals do not establish causation.",
                "Only observations up to the selected replay cursor are included.",
            ],
        )
