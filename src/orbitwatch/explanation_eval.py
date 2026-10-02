from __future__ import annotations

import json
import time
from itertools import zip_longest
from pathlib import Path

import pandas as pd

from orbitwatch.evidence import (
    CORE_FIELDS,
    Claim,
    check_claims,
    generate_draft,
    template_report,
)
from orbitwatch.service import TelemetryService

GENERATION_LATENCY_SCOPE = (
    "Wall-clock LLM draft generation only; excludes validation and rendering. "
    "The paired validator arm reuses the ordinary draft's timing. "
    "Template generation was not timed and is reported as null, not zero."
)


def evaluate_explanations(root: Path, run: str | None, model: str, cases: int = 12) -> Path:
    if not 2 <= cases <= 100:
        raise ValueError("Explanation evaluation needs between 2 and 100 cases.")
    service = TelemetryService(root, run)
    by_mission: dict[str, list] = {}
    for channel in service.manifest["channels"]:
        events = service.events(channel)
        if events:
            event = events[len(events) // 2]
            evidence = service.evidence(channel, event["start"], event["end"])
            by_mission.setdefault(evidence.spacecraft, []).append(evidence)
    selected = [
        evidence
        for group in zip_longest(*[by_mission[name] for name in sorted(by_mission)])
        for evidence in group
        if evidence is not None
    ]
    if not selected:
        raise ValueError("This experiment produced no GRU alert events to investigate.")
    rows = []
    records = []
    output_dir = service.run_dir / "explanations"
    output_dir.mkdir(exist_ok=True)
    for index in range(min(cases, len(selected) * 2)):
        original = selected[index // 2]
        condition = "complete" if index % 2 == 0 else "missing_prediction"
        facts = dict(original.facts)
        if index % 2:
            facts.pop("peak_expected")
        evidence = original.model_copy(update={"facts": facts})
        print(
            f"Explanation case {index + 1}/{min(cases, len(selected) * 2)}: {facts['channel']} / {condition}",
            flush=True,
        )
        template = template_report(evidence)
        proposals: list[tuple[str, list[Claim], float | None]] = [
            ("template", template.accepted, None)
        ]
        for name, grounded in (("ordinary_local_llm", False), ("grounded_local_llm", True)):
            started = time.perf_counter()
            draft = generate_draft(evidence, model, grounded=grounded)
            latency = time.perf_counter() - started
            proposals.append((name, draft.claims, latency))
            if not grounded:
                proposals.append(("ordinary_llm_plus_validator", draft.claims, latency))
        for mode, claims, latency in proposals:
            supported, rejected = check_claims(evidence, claims)
            emitted = (
                supported
                if mode in ("grounded_local_llm", "ordinary_llm_plus_validator")
                else claims
            )
            emitted_supported, emitted_rejected = check_claims(evidence, emitted)
            row = {
                "case": index + 1,
                "channel": facts["channel"],
                "condition": condition,
                "mode": mode,
                "proposed_claims": len(claims),
                "unsupported_proposed_claims": len(rejected),
                "emitted_claims": len(emitted),
                "unsupported_emitted_claims": len(emitted_rejected),
                "coverage": len(emitted_supported) / len(CORE_FIELDS & facts.keys()),
                "latency_seconds": latency,
            }
            rows.append(row)
            records.append(
                {
                    **row,
                    "evidence": evidence.model_dump(),
                    "draft": [claim.model_dump() for claim in claims],
                    "rejected": [item.model_dump() for item in rejected],
                }
            )
        pd.DataFrame(rows).to_csv(output_dir / "cases.csv", index=False)
        (output_dir / "records.json").write_text(json.dumps(records, indent=2))
    frame = pd.DataFrame(rows)
    summary = []
    for mode, group in frame.groupby("mode"):
        emitted = int(group["emitted_claims"].sum())
        summary.append(
            {
                "mode": mode,
                "cases": len(group),
                "unsupported_emitted_rate": int(group["unsupported_emitted_claims"].sum()) / emitted
                if emitted
                else None,
                "mean_coverage": float(group["coverage"].mean()),
                "mean_latency_seconds": float(group["latency_seconds"].mean())
                if group["latency_seconds"].notna().any()
                else None,
                "unsupported_proposed_claims": int(group["unsupported_proposed_claims"].sum()),
            }
        )
    metadata = {
        "model": model,
        "latency_scope": GENERATION_LATENCY_SCOPE,
        "selection": "Interleaved missions, one midpoint-in-time predicted event per sorted channel; paired complete/redacted evidence. No test labels used.",
        "limitations": [
            "Pilot experiment, not an expert-rated semantic explanation benchmark.",
            "The ordinary-LLM-plus-validator ablation reuses the identical ordinary-model response to isolate filtering. The grounded variant also changes the prompt.",
            "Zero unsupported emitted facts is enforced by a restricted claim schema and trusted rendering; it is not a proof of unrestricted LLM truthfulness.",
            "The template baseline has the same numerical correctness guarantee without an LLM.",
            "Only numerical/identifier claims and the unknown-cause boundary are checked.",
        ],
        "summary": summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
    return output_dir
