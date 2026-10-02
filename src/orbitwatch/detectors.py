from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from orbitwatch.config import ExperimentConfig


class WindowDataset(Dataset):
    def __init__(self, data: np.ndarray, window: int, limit: int | None = None):
        self.data = torch.from_numpy(np.asarray(data, dtype=np.float32))
        self.window = window
        if len(data) <= window:
            raise ValueError("A sequence must contain more observations than the context window.")
        count = len(data) - window
        self.indices = (
            np.unique(np.linspace(window, len(data) - 1, limit, dtype=int))
            if limit is not None and count > limit
            else np.arange(window, len(data))
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        end = self.indices[index]
        return self.data[end - self.window : end], self.data[end, 0]


class GRUForecaster(nn.Module):
    def __init__(self, features: int, hidden: int):
        super().__init__()
        self.gru = nn.GRU(features, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        _, state = self.gru(values)
        return self.head(state[-1]).squeeze(-1)


@dataclass
class DetectorOutput:
    baseline_prediction: np.ndarray
    gru_prediction: np.ndarray
    baseline_scores: np.ndarray
    gru_scores: np.ndarray
    thresholds: dict[str, float]
    training_losses: list[float]
    training_seconds: float
    gru_p95_latency_ms: float
    baseline_p95_latency_ms: float
    checkpoint: dict
    fit_samples: int
    calibration_samples: int


def rolling_median_prediction(values: np.ndarray, window: int) -> np.ndarray:
    target = np.asarray(values)[:, 0]
    if len(target) <= window:
        raise ValueError("Not enough observations for the context window.")
    prediction = np.full(len(target), np.nan, dtype=np.float32)
    prediction[window:] = np.median(
        np.lib.stride_tricks.sliding_window_view(target[:-1], window), axis=1
    )
    return prediction


def predict_gru(
    model: GRUForecaster, values: np.ndarray, window: int, batch_size: int
) -> np.ndarray:
    loader = DataLoader(WindowDataset(values, window), batch_size=batch_size, shuffle=False)
    output = np.full(len(values), np.nan, dtype=np.float32)
    position = window
    model.eval()
    with torch.inference_mode():
        for context, _ in loader:
            predictions = model(context).numpy()
            output[position : position + len(predictions)] = predictions
            position += len(predictions)
    return output


def fit_and_score(train: np.ndarray, test: np.ndarray, config: ExperimentConfig) -> DetectorOutput:
    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    split = int(len(train) * (1 - config.calibration_fraction))
    if min(split, len(train) - split, len(test)) <= config.window:
        raise ValueError("Training, calibration, and test segments must each exceed the window.")
    fit = train[:split]
    mean = fit.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = fit.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    normalized_train = (train - mean) / scale
    normalized_test = (test - mean) / scale
    # Context is taken from the past; calibration targets never occur in the fitting segment.
    calibration = normalized_train[split - config.window :]
    model = GRUForecaster(train.shape[1], config.hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss()
    loader = DataLoader(
        WindowDataset(normalized_train[:split], config.window, config.max_train_windows),
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    losses: list[float] = []
    started = time.perf_counter()
    for _ in range(config.epochs):
        model.train()
        total = 0.0
        count = 0
        for context, target in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(context), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(context)
            count += len(context)
        losses.append(total / count)
    training_seconds = time.perf_counter() - started
    model.eval()
    baseline_cal = rolling_median_prediction(calibration, config.window)
    gru_cal = predict_gru(model, calibration, config.window, config.batch_size)
    thresholds = {
        name: max(
            float(
                np.quantile(
                    np.abs(calibration[config.window :, 0] - prediction[config.window :]),
                    config.threshold_quantile,
                )
            ),
            1e-6,
        )
        for name, prediction in (("rolling_median", baseline_cal), ("gru", gru_cal))
    }
    baseline = rolling_median_prediction(normalized_test, config.window)
    gru = predict_gru(model, normalized_test, config.window, config.batch_size)
    latencies: dict[str, list[float]] = {"gru": [], "rolling_median": []}
    sample_points = np.linspace(
        config.window, len(test) - 1, min(40, len(test) - config.window), dtype=int
    )
    with torch.inference_mode():
        for index in sample_points:
            context = torch.from_numpy(normalized_test[index - config.window : index]).unsqueeze(0)
            start = time.perf_counter()
            model(context)
            latencies["gru"].append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            np.median(normalized_test[index - config.window : index, 0])
            latencies["rolling_median"].append((time.perf_counter() - start) * 1000)
    return DetectorOutput(
        baseline_prediction=baseline * scale[0] + mean[0],
        gru_prediction=gru * scale[0] + mean[0],
        baseline_scores=np.abs(normalized_test[:, 0] - baseline),
        gru_scores=np.abs(normalized_test[:, 0] - gru),
        thresholds=thresholds,
        training_losses=losses,
        training_seconds=training_seconds,
        gru_p95_latency_ms=float(np.percentile(latencies["gru"], 95)),
        baseline_p95_latency_ms=float(np.percentile(latencies["rolling_median"], 95)),
        checkpoint={
            "state_dict": model.state_dict(),
            "features": train.shape[1],
            "hidden": config.hidden,
            "window": config.window,
            "mean": torch.from_numpy(mean),
            "scale": torch.from_numpy(scale),
            "threshold": thresholds["gru"],
            "config": config.model_dump(),
        },
        fit_samples=split,
        calibration_samples=len(train) - split,
    )
