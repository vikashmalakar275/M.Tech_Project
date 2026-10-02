from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch
from pydantic import ValidationError

from orbitwatch.config import ExperimentConfig
from orbitwatch.data import available_channels, load_channel, validate_channel
from orbitwatch.detectors import (
    GRUForecaster,
    WindowDataset,
    fit_and_score,
    predict_gru,
    rolling_median_prediction,
)
from orbitwatch.evidence import Claim, check_claims, make_report, report_markdown, template_report
from orbitwatch.experiment import resolve_run
from orbitwatch.metrics import detection_metrics, intervals
from orbitwatch.service import TelemetryService
from orbitwatch.store import InvestigationStore


@pytest.mark.parametrize("value", ["../A-1", "/tmp", "a/b", "", "A';DROP TABLE x;--"])
def test_channel_identifier_rejects_paths_and_sql(value):
    with pytest.raises(ValueError):
        validate_channel(value)


def test_original_data_and_inclusive_labels(completed_root):
    channel = load_channel(completed_root / "data", "A-1")
    assert channel.labels[70:85].sum() == 15
    assert channel.labels[69] == channel.labels[85] == 0
    assert channel.train.dtype == np.float32


def test_duplicate_annotations_are_unioned_not_double_counted(tmp_path):
    pd.DataFrame(
        [
            {"chan_id": "P-2", "spacecraft": "SMAP", "anomaly_sequences": "[[5350,6575]]"},
            {"chan_id": "P-2", "spacecraft": "SMAP", "anomaly_sequences": "[[5300,6420]]"},
        ]
    ).to_csv(tmp_path / "labeled_anomalies.csv", index=False)
    metadata = available_channels(tmp_path)
    assert len(metadata) == 1
    assert metadata.iloc[0]["annotation_rows"] == 2
    assert metadata.iloc[0]["anomaly_sequences"] == "[[5300, 6575]]"


def test_duplicate_mission_conflicts_fail(tmp_path):
    pd.DataFrame(
        [
            {"chan_id": "P-2", "spacecraft": "SMAP", "anomaly_sequences": "[[1,2]]"},
            {"chan_id": "P-2", "spacecraft": "MSL", "anomaly_sequences": "[[1,2]]"},
        ]
    ).to_csv(tmp_path / "labeled_anomalies.csv", index=False)
    with pytest.raises(ValueError, match="conflicting mission"):
        available_channels(tmp_path)


def test_windows_are_past_only_and_include_last_target():
    values = np.arange(20, dtype=np.float32).reshape(10, 2)
    dataset = WindowDataset(values, 4)
    assert len(dataset) == 6
    context, target = dataset[-1]
    np.testing.assert_equal(context.numpy(), values[5:9])
    assert target.item() == values[9, 0]


@pytest.mark.parametrize("predictor", ["median", "gru"])
def test_future_perturbation_does_not_change_past_predictions(predictor):
    rng = np.random.default_rng(5)
    values = rng.normal(size=(60, 2)).astype(np.float32)
    changed = values.copy()
    changed[40:] += 200
    if predictor == "median":
        a, b = (rolling_median_prediction(x, 8) for x in (values, changed))
    else:
        model = GRUForecaster(2, 8)
        a, b = (predict_gru(model, x, 8, 16) for x in (values, changed))
    np.testing.assert_allclose(a[:41], b[:41], equal_nan=True, atol=1e-6)


def test_threshold_does_not_depend_on_test_sequence(completed_root):
    data = load_channel(completed_root / "data", "A-1")
    config = ExperimentConfig(window=8, hidden=8, epochs=1, max_train_windows=128, threads=1)
    normal = fit_and_score(data.train, data.test, config)
    extreme = fit_and_score(data.train, data.test + 100, config)
    assert normal.thresholds == extreme.thresholds
    assert normal.training_losses == extreme.training_losses


def test_model_checkpoint_roundtrip(completed_root):
    channel = load_channel(completed_root / "data", "A-1")
    checkpoint = torch.load(completed_root / "runs/fixture/A-1/gru.pt", weights_only=True)
    model = GRUForecaster(checkpoint["features"], checkpoint["hidden"])
    model.load_state_dict(checkpoint["state_dict"])
    normalized = (channel.test - checkpoint["mean"].numpy()) / checkpoint["scale"].numpy()
    predictions = predict_gru(model, normalized, checkpoint["window"], 16)
    restored = predictions * checkpoint["scale"][0].item() + checkpoint["mean"][0].item()
    arrays = TelemetryService(completed_root).arrays("A-1")
    np.testing.assert_allclose(restored, arrays["gru_prediction"], atol=1e-5, equal_nan=True)


@pytest.mark.parametrize(
    ("mask", "expected"),
    [([], []), ([0, 0], []), ([1], [(0, 0)]), ([0, 1, 1, 0, 1], [(1, 2), (4, 4)])],
)
def test_event_boundaries(mask, expected):
    assert intervals(np.array(mask)) == expected


def test_event_matching_is_one_to_one_without_point_adjustment():
    result = detection_metrics(np.array([0, 1, 1, 1, 0]), np.array([0, 2, 0, 2, 0]), 1)
    assert result["matched_events"] == 1
    assert result["predicted_events"] == 2
    assert result["event_precision"] == 0.5
    assert result["point_recall"] == pytest.approx(2 / 3)
    assert result["mean_delay_samples"] == 0


def test_no_positive_labels_is_explicit():
    result = detection_metrics(np.zeros(4), np.array([0, 2, 0, 0]), 1)
    assert result["average_precision"] is None
    assert result["mean_delay_samples"] is None
    assert result["false_alarms_per_1000_normal_samples"] == 250


@pytest.mark.parametrize(
    ("labels", "scores", "threshold"),
    [([1], [np.nan], 1), ([1], [2], 0), ([0, 2], [0, 1], 1), ([0, 1], [0], 1)],
)
def test_invalid_metrics_are_not_silently_coerced(labels, scores, threshold):
    with pytest.raises(ValueError):
        detection_metrics(np.array(labels), np.array(scores), threshold)


@pytest.fixture
def evidence(completed_root):
    service = TelemetryService(completed_root)
    event = max(service.events("A-1"), key=lambda item: item["peak_ratio"])
    return service.evidence("A-1", event["start"], event["end"])


def test_template_is_complete_and_cited(evidence):
    report = template_report(evidence)
    assert report.coverage == 1
    assert report.status == "complete"
    assert not report.rejected
    assert evidence.evidence_id in report_markdown(report)
    assert "not a verified spacecraft fault" in report_markdown(report)


@pytest.mark.parametrize(
    ("field", "value", "citation"),
    [
        ("physical_cause", "solar panel degradation", "correct"),
        ("bus_voltage", 28.0, "correct"),
        ("channel", "fabricated-channel", "correct"),
        ("peak_ratio", -200.0, "correct"),
        ("channel", "A-1", "invented-citation"),
    ],
)
def test_unsupported_claims_are_rejected(evidence, field, value, citation):
    claim = Claim(
        field=field,
        value=value,
        evidence_id=evidence.evidence_id if citation == "correct" else citation,
    )
    accepted, rejected = check_claims(evidence, [claim])
    assert not accepted
    assert len(rejected) == 1


def test_duplicate_claim_does_not_inflate_coverage(evidence):
    claim = Claim(field="channel", value="A-1", evidence_id=evidence.evidence_id)
    report = make_report(evidence, [claim, claim], "test")
    assert report.coverage == 1 / 8
    assert len(report.rejected) == 1


def test_missing_fact_requires_abstention(evidence):
    redacted = evidence.model_copy(update={"facts": {"channel": "A-1"}})
    claim = Claim(field="physical_cause", value="unknown", evidence_id=evidence.evidence_id)
    report = make_report(redacted, [claim], "test")
    assert report.status == "insufficient_evidence"
    assert report.coverage == 0


def test_sample_indices_cannot_be_rounded(evidence):
    value = evidence.facts["start_sample"]
    claim = Claim(field="start_sample", value=value + 0.00001, evidence_id=evidence.evidence_id)
    assert check_claims(evidence, [claim])[1]


def test_nonfinite_claims_fail_schema():
    with pytest.raises(ValidationError):
        Claim(field="peak_ratio", value=float("nan"), evidence_id="evidence")


def test_replay_never_exposes_future_events(completed_root):
    service = TelemetryService(completed_root)
    for event in service.events("A-1", until=74):
        assert event["end"] <= 74
        assert event["peak_sample"] <= 74
    with pytest.raises(ValueError):
        service.evidence("A-1", 70, 84, until=74)


def test_unknown_run_and_invalid_windows_fail(completed_root):
    with pytest.raises(ValueError):
        resolve_run(completed_root, "../escape")
    service = TelemetryService(completed_root)
    with pytest.raises(ValueError):
        service.telemetry_window("A-1", -1, 3, "gru")
    with pytest.raises(ValueError):
        service.telemetry_window("A-1", 0, 3, "invented")
    window = service.telemetry_window("A-1", 0, 10, "gru")
    assert window["predicted"][:8] == [None] * 8
    json.dumps(window, allow_nan=False)


def test_history_persists(tmp_path, evidence):
    path = tmp_path / "history.sqlite"
    store = InvestigationStore(path)
    report_id = store.save(template_report(evidence))
    restored = InvestigationStore(path).recent()
    assert restored[0]["id"] == report_id
    assert restored[0]["evidence_id"] == evidence.evidence_id
