from __future__ import annotations

import ast
import hashlib
import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import httpx
import numpy as np
import pandas as pd

from orbitwatch.config import DATASET_CAVEAT, DATASET_URL, LABELS_URL

CHANNEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def validate_channel(channel: str) -> str:
    if not CHANNEL_PATTERN.fullmatch(channel):
        raise ValueError("Channel ID must contain only letters, digits, underscores, or hyphens.")
    return channel


@dataclass(frozen=True)
class TelemetryChannel:
    channel: str
    spacecraft: str
    train: np.ndarray
    test: np.ndarray
    labels: np.ndarray


def available_channels(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "labeled_anomalies.csv"
    if not path.is_file():
        raise FileNotFoundError("NASA data are not installed. Run `orbitwatch download` first.")
    metadata = pd.read_csv(path)
    required = {"chan_id", "spacecraft", "anomaly_sequences"}
    if not required.issubset(metadata.columns):
        raise ValueError(
            f"Dataset metadata is missing columns: {sorted(required - set(metadata.columns))}"
        )
    for channel in metadata["chan_id"]:
        validate_channel(str(channel))
    canonical = []
    for channel, group in metadata.groupby("chan_id", sort=False):
        if group["spacecraft"].nunique() != 1 or (
            "num_values" in group and group["num_values"].nunique() != 1
        ):
            raise ValueError(f"{channel}: conflicting mission or length in duplicate annotations.")
        ranges = sorted(
            tuple(interval)
            for value in group["anomaly_sequences"]
            for interval in ast.literal_eval(str(value))
        )
        merged: list[list[int]] = []
        for start, end in ranges:
            if start < 0 or end < start:
                raise ValueError(f"{channel}: invalid annotation interval [{start}, {end}].")
            if merged and start <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        row = group.iloc[0].to_dict()
        row["anomaly_sequences"] = str(merged)
        row["annotation_rows"] = len(group)
        canonical.append(row)
    return pd.DataFrame(canonical).sort_values(["spacecraft", "chan_id"]).reset_index(drop=True)


def load_channel(data_dir: Path, channel: str) -> TelemetryChannel:
    validate_channel(channel)
    metadata = available_channels(data_dir)
    rows = metadata.loc[metadata["chan_id"] == channel]
    if len(rows) != 1:
        raise ValueError(f"Unknown channel: {channel}")
    row = rows.iloc[0]
    train = np.load(data_dir / "train" / f"{channel}.npy", allow_pickle=False)
    test = np.load(data_dir / "test" / f"{channel}.npy", allow_pickle=False)
    if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]:
        raise ValueError(f"{channel}: expected matching two-dimensional feature arrays.")
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise ValueError(
            f"{channel}: non-finite telemetry must be handled explicitly, not imputed silently."
        )
    labels = np.zeros(len(test), dtype=np.uint8)
    ranges = ast.literal_eval(str(row["anomaly_sequences"]))
    for start, end in ranges:
        if not (0 <= start <= end < len(test)):
            raise ValueError(f"{channel}: invalid anomaly interval [{start}, {end}].")
        labels[start : end + 1] = 1
    return TelemetryChannel(
        channel, str(row["spacecraft"]), train.astype(np.float32), test.astype(np.float32), labels
    )


def download_dataset(data_dir: Path) -> dict:
    """Install only explicitly recognized files, after verifying the complete archive."""
    manifest_path = data_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        metadata = available_channels(data_dir)
        for channel in metadata["chan_id"]:
            for split in ("train", "test"):
                if not (data_dir / split / f"{channel}.npy").is_file():
                    raise FileNotFoundError(f"Incomplete dataset: {split}/{channel}.npy")
        return manifest
    if data_dir.exists() and any(data_dir.iterdir()):
        raise FileExistsError(
            f"{data_dir} is nonempty but has no manifest; refusing to overwrite it."
        )
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="orbitwatch-download-", dir=data_dir.parent) as temp:
        staging = Path(temp)
        archive = staging / "dataset.zip"
        digest = hashlib.sha256()
        prepared = staging / "prepared"
        prepared.mkdir()
        with httpx.Client(follow_redirects=True, timeout=180) as client:
            labels_response = client.get(LABELS_URL)
            labels_response.raise_for_status()
            (prepared / "labeled_anomalies.csv").write_text(labels_response.text)
            metadata = available_channels(prepared)
            with client.stream("GET", DATASET_URL) as response:
                response.raise_for_status()
                total = 0
                with archive.open("wb") as handle:
                    for chunk in response.iter_bytes(1024 * 1024):
                        total += len(chunk)
                        if total > 512 * 1024 * 1024:
                            raise ValueError("Dataset download exceeded the 512 MiB safety limit.")
                        digest.update(chunk)
                        handle.write(chunk)
        expected = {
            f"{split}/{channel}.npy"
            for split in ("train", "test")
            for channel in metadata["chan_id"]
        }
        extracted: set[str] = set()
        extracted_bytes = 0
        try:
            with ZipFile(archive) as zipped:
                for entry in zipped.infolist():
                    parts = Path(entry.filename).parts
                    if ".." in parts or Path(entry.filename).is_absolute():
                        raise ValueError("Unsafe archive path.")
                    if len(parts) < 2:
                        continue
                    relative = "/".join(parts[-2:])
                    if relative not in expected:
                        continue
                    if relative in extracted or entry.file_size > 128 * 1024 * 1024:
                        raise ValueError(f"Duplicate or oversized dataset member: {relative}")
                    extracted_bytes += entry.file_size
                    if extracted_bytes > 2 * 1024 * 1024 * 1024:
                        raise ValueError("Extracted dataset exceeded the 2 GiB safety limit.")
                    target = prepared / relative
                    target.parent.mkdir(exist_ok=True)
                    with zipped.open(entry) as source, target.open("wb") as destination:
                        shutil.copyfileobj(source, destination)
                    extracted.add(relative)
        except BadZipFile as error:
            raise ValueError("Dataset server did not return a valid ZIP archive.") from error
        if extracted != expected:
            raise ValueError(f"Archive is missing {len(expected - extracted)} required data files.")
        for channel in metadata["chan_id"]:
            load_channel(prepared, str(channel))
        manifest = {
            "source_url": DATASET_URL,
            "labels_url": LABELS_URL,
            "archive_sha256": digest.hexdigest(),
            "labels_sha256": hashlib.sha256(labels_response.content).hexdigest(),
            "downloaded_at": datetime.now(UTC).isoformat(),
            "channels": len(metadata),
            "source_annotation_rows": int(metadata["annotation_rows"].sum()),
            "duplicate_annotations": {
                str(row["chan_id"]): int(row["annotation_rows"])
                for _, row in metadata.iterrows()
                if row["annotation_rows"] > 1
            },
            "annotation_policy": (
                "Merge duplicate annotations for the same channel by interval union, with one "
                "evaluation per unique file. The original 82-row metadata repeats SMAP P-2 "
                "with overlapping intervals; this yields 81 unique channels."
            ),
            "caveat": DATASET_CAVEAT,
        }
        (prepared / "manifest.json").write_text(json.dumps(manifest, indent=2))
        data_dir.mkdir(exist_ok=True)
        for item in prepared.iterdir():
            shutil.move(str(item), str(data_dir / item.name))
    return manifest
