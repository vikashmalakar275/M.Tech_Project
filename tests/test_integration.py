from __future__ import annotations

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from orbitwatch.mcp_client import investigate_via_mcp
from orbitwatch.service import TelemetryService


@pytest.mark.asyncio
async def test_actual_mcp_subprocess_exchange(completed_root):
    service = TelemetryService(completed_root)
    event = max(service.events("A-1"), key=lambda item: item["peak_ratio"])
    evidence, trace = await investigate_via_mcp(
        completed_root, "A-1", event["start"], event["end"], "gru", 159, "fixture"
    )
    assert evidence.facts["channel"] == "A-1"
    assert "get_event_evidence" in trace[0]["tools"]
    assert trace[1]["evidence_id"] == evidence.evidence_id


def test_streamlit_renders_and_changes_detector(completed_root, monkeypatch):
    monkeypatch.setenv("ORBITWATCH_HOME", str(completed_root))
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert len(app.metric) == 4
    app.selectbox(key="detector").select("rolling_median").run()
    assert not app.exception


def test_streamlit_mcp_investigation_and_replay(completed_root, monkeypatch):
    monkeypatch.setenv("ORBITWATCH_HOME", str(completed_root))
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=30).run()
    app.button(key="investigate").click().run(timeout=60)
    assert not app.exception
    assert not app.error
    assert any("OrbitWatch investigation" in element.value for element in app.markdown)
    app.button(key="restart").click().run()
    assert not app.exception
    assert app.metric[0].value == "9"
    app.button(key="finish").click().run()
    assert app.metric[0].value == "160"


def test_streamlit_empty_install_is_actionable(tmp_path, monkeypatch):
    monkeypatch.setenv("ORBITWATCH_HOME", str(tmp_path))
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert "Download the benchmark" in app.info[0].value


def test_results_have_real_provenance(completed_root):
    manifest = json.loads((completed_root / "runs/fixture/manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert len(manifest["source_sha256"]) == 64
    assert manifest["config"]["seed"] == 17
    assert manifest["duration_seconds"] > 0
