from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from orbitwatch.config import ExperimentConfig
from orbitwatch.experiment import run_experiment


@pytest.fixture(scope="session")
def completed_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("orbitwatch")
    data = root / "data"
    (data / "train").mkdir(parents=True)
    (data / "test").mkdir()
    records = []
    rng = np.random.default_rng(71)
    for channel, spacecraft in (("A-1", "SMAP"), ("P-1", "MSL")):
        train = np.c_[np.sin(np.arange(240) / 12) + rng.normal(0, 0.01, 240), np.zeros(240)].astype(
            np.float32
        )
        test = np.c_[np.sin(np.arange(160) / 12), np.zeros(160)].astype(np.float32)
        test[70:85, 0] += 6
        np.save(data / "train" / f"{channel}.npy", train)
        np.save(data / "test" / f"{channel}.npy", test)
        records.append(
            {"chan_id": channel, "spacecraft": spacecraft, "anomaly_sequences": "[[70,84]]"}
        )
    pd.DataFrame(records).to_csv(data / "labeled_anomalies.csv", index=False)
    (data / "manifest.json").write_text(
        json.dumps({"source": "synthetic unit-test fixture, not submission data"})
    )
    config = ExperimentConfig(window=8, hidden=8, epochs=1, max_train_windows=128, threads=1)
    run_experiment(root, config, run_id="fixture", progress=lambda _: None)
    return root
