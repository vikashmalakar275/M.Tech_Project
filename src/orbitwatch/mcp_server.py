from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from mcp.server.fastmcp import FastMCP

from orbitwatch.config import project_root
from orbitwatch.evidence import template_report
from orbitwatch.service import TelemetryService
from orbitwatch.store import InvestigationStore

mcp = FastMCP(
    "OrbitWatch",
    instructions=(
        "Research-only NASA telemetry investigation. Values are anonymized and dataset-scaled. "
        "Never infer physical causes, mission timestamps, or operational recommendations. "
        "Use sample indices and cite the evidence_id returned by get_event_evidence."
    ),
)


def execute(tool: str, arguments: dict, operation: Callable[[TelemetryService], Any]) -> Any:
    root = project_root()
    store = InvestigationStore(root / "local" / "investigations.sqlite3")
    try:
        result = operation(TelemetryService(root, os.environ.get("ORBITWATCH_RUN")))
    except (ValueError, FileNotFoundError, OSError) as error:
        store.audit(tool, arguments, f"error: {error}")
        raise
    store.audit(tool, arguments, "success")
    return result


@mcp.tool()
def list_channels() -> dict:
    """List evaluated channels, missions, sequence lengths, and input dimensions."""
    return execute(
        "list_channels",
        {},
        lambda service: {
            "channels": [
                {key: item[key] for key in ("channel", "spacecraft", "test_samples", "features")}
                for item in service.list_channels()
            ]
        },
    )


@mcp.tool()
def get_telemetry_window(channel: str, start: int, end: int, detector: str = "gru") -> dict:
    """Read up to 1024 observations. Bounds are inclusive sample indices, not timestamps."""
    arguments = {"channel": channel, "start": start, "end": end, "detector": detector}
    return execute(
        "get_telemetry_window", arguments, lambda service: service.telemetry_window(**arguments)
    )


@mcp.tool()
def detect_anomalies(
    channel: str, detector: str = "gru", until: int | None = None, offset: int = 0, limit: int = 50
) -> dict:
    """Retrieve causally computed alerts up to a cursor; never reveals alerts after until."""
    if not 1 <= limit <= 100 or offset < 0:
        raise ValueError("Use limit 1..100 and a nonnegative offset.")
    arguments = {"channel": channel, "detector": detector, "until": until}

    def operation(service: TelemetryService) -> dict:
        events = service.events(**arguments)
        return {"total": len(events), "offset": offset, "events": events[offset : offset + limit]}

    return execute("detect_anomalies", {**arguments, "offset": offset, "limit": limit}, operation)


@mcp.tool()
def get_event_evidence(
    channel: str, start: int, end: int, detector: str = "gru", until: int | None = None
) -> dict:
    """Produce traceable, numeric alert evidence. Physical cause is always unknown."""
    arguments = {
        "channel": channel,
        "start": start,
        "end": end,
        "detector": detector,
        "until": until,
    }
    return execute(
        "get_event_evidence", arguments, lambda service: service.evidence(**arguments).model_dump()
    )


@mcp.tool()
def generate_evidence_report(
    channel: str, start: int, end: int, detector: str = "gru", until: int | None = None
) -> dict:
    """Generate a deterministic evidence-backed report without any external model service."""
    arguments = {
        "channel": channel,
        "start": start,
        "end": end,
        "detector": detector,
        "until": until,
    }
    return execute(
        "generate_evidence_report",
        arguments,
        lambda service: template_report(service.evidence(**arguments)).model_dump(),
    )


@mcp.resource("orbitwatch://experiment")
def experiment_manifest() -> str:
    """Read the completed experiment's provenance and declared limitations."""
    service = TelemetryService(project_root(), os.environ.get("ORBITWATCH_RUN"))
    return (service.run_dir / "manifest.json").read_text()


@mcp.prompt()
def investigate_alert(channel: str) -> str:
    return (
        f"Investigate an alert in channel {channel}. List available channels, call detect_anomalies, "
        "then get_event_evidence for one visible event. Cite its evidence_id. Do not interpret "
        "anonymous channels as named hardware. Do not claim that correlation proves causation."
    )


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
