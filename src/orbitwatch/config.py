from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


def project_root() -> Path:
    return Path(os.environ.get("ORBITWATCH_HOME", Path(__file__).resolve().parents[2])).resolve()


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    window: int = Field(default=32, ge=4, le=256)
    hidden: int = Field(default=24, ge=4, le=128)
    epochs: int = Field(default=4, ge=1, le=100)
    batch_size: int = Field(default=128, ge=1, le=1024)
    learning_rate: float = Field(default=0.003, gt=0, le=0.1)
    calibration_fraction: float = Field(default=0.2, gt=0.05, lt=0.5)
    threshold_quantile: float = Field(default=0.995, gt=0.5, lt=1)
    max_train_windows: int = Field(default=6000, ge=128)
    seed: int = Field(default=17, ge=0)
    threads: int = Field(default=4, ge=1, le=16)


DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "patrickfleith/nasa-anomaly-detection-dataset-smap-msl"
)
LABELS_URL = "https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv"
DATASET_CAVEAT = (
    "NASA SMAP/MSL streams are anonymized, contain one target telemetry value plus encoded "
    "command features, and are not assumed to be synchronized across files. Published data "
    "were already scaled using test-set extrema; this upstream benchmark limitation cannot "
    "be undone. OrbitWatch fits its own scaler on training data only. Sample indices are not "
    "wall-clock timestamps. Physical root-cause labels are unavailable."
)
