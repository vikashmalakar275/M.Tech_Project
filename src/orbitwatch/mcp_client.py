from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from orbitwatch.evidence import EvidencePacket


async def investigate_via_mcp(
    root: Path, channel: str, start: int, end: int, detector: str, until: int, run: str
) -> tuple[EvidencePacket, list[dict]]:
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "orbitwatch.mcp_server"],
        env={"ORBITWATCH_HOME": str(root), "ORBITWATCH_RUN": run},
    )
    trace: list[dict] = []
    async with stdio_client(server) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            discovery = await session.list_tools()
            trace.append(
                {"operation": "tools/list", "tools": [tool.name for tool in discovery.tools]}
            )
            arguments = {
                "channel": channel,
                "start": start,
                "end": end,
                "detector": detector,
                "until": until,
            }
            started = time.perf_counter()
            result = await session.call_tool("get_event_evidence", arguments)
            if result.isError:
                raise ValueError(f"MCP evidence request failed: {result.content}")
            content = result.structuredContent
            if content is None:
                texts = [item.text for item in result.content if item.type == "text"]
                if not texts:
                    raise ValueError("MCP server returned no structured evidence.")
                content = json.loads(texts[0])
            evidence = EvidencePacket.model_validate(content)
            trace.append(
                {
                    "operation": "tools/call",
                    "tool": "get_event_evidence",
                    "arguments": arguments,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                    "evidence_id": evidence.evidence_id,
                }
            )
    return evidence, trace
