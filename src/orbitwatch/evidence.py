from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, StrictStr

FactValue = StrictStr | StrictInt | StrictFloat
CORE_FIELDS = {
    "channel",
    "start_sample",
    "end_sample",
    "peak_sample",
    "peak_ratio",
    "peak_observed",
    "peak_expected",
    "physical_cause",
}
FIELD_LABELS = {
    "channel": "Telemetry channel",
    "start_sample": "Alert begins at sample",
    "end_sample": "Alert ends at sample",
    "peak_sample": "Largest residual at sample",
    "peak_ratio": "Peak residual / calibrated threshold",
    "peak_observed": "Observed value at the peak (dataset-scaled units)",
    "peak_expected": "Predicted value at the peak (dataset-scaled units)",
    "physical_cause": "Physical cause",
}


class EvidencePacket(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    evidence_id: str
    run_id: str
    spacecraft: str
    detector: Literal["rolling_median", "gru"]
    facts: dict[str, FactValue]
    threshold: float = Field(gt=0)
    peak_score: float = Field(ge=0)
    duration_samples: int = Field(ge=1)
    source_sha256: str
    ongoing_at_cursor: bool = False
    limitations: list[str]


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    field: str = Field(min_length=1, max_length=80)
    value: FactValue
    evidence_id: str = Field(min_length=1, max_length=100)


class ClaimDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claims: list[Claim] = Field(min_length=1, max_length=12)


class RejectedClaim(BaseModel):
    claim: Claim
    reason: str


class InvestigationReport(BaseModel):
    evidence: EvidencePacket
    engine: str
    status: Literal["complete", "partial", "insufficient_evidence"]
    accepted: list[Claim]
    rejected: list[RejectedClaim]
    coverage: float
    latency_seconds: float
    generated_at: str


def evidence_identifier(run: str, channel: str, detector: str, start: int, end: int) -> str:
    return hashlib.sha256(f"{run}:{channel}:{detector}:{start}:{end}".encode()).hexdigest()[:20]


def check_claims(
    evidence: EvidencePacket, claims: list[Claim]
) -> tuple[list[Claim], list[RejectedClaim]]:
    accepted: list[Claim] = []
    rejected: list[RejectedClaim] = []
    seen: set[str] = set()
    for claim in claims:
        reason = None
        if claim.evidence_id != evidence.evidence_id:
            reason = "Citation does not reference this evidence record."
        elif claim.field not in CORE_FIELDS or claim.field not in evidence.facts:
            reason = "The evidence does not contain this permitted fact."
        elif claim.field in seen:
            reason = "Duplicate claim does not add factual coverage."
        else:
            actual = evidence.facts[claim.field]
            if isinstance(actual, str):
                if claim.value != actual:
                    reason = "Text claim is not supported by the evidence."
            elif isinstance(claim.value, str):
                reason = "A numeric fact requires a numeric value."
            elif isinstance(actual, int) and claim.value != actual:
                reason = "Sample indices must match exactly."
            elif not math.isclose(float(actual), float(claim.value), rel_tol=0, abs_tol=1e-6):
                reason = "Numeric claim does not match the recorded measurement."
        if reason:
            rejected.append(RejectedClaim(claim=claim, reason=reason))
        else:
            seen.add(claim.field)
            accepted.append(claim)
    return accepted, rejected


def make_report(
    evidence: EvidencePacket, claims: list[Claim], engine: str, latency: float = 0.0
) -> InvestigationReport:
    accepted, rejected = check_claims(evidence, claims)
    denominator = len(CORE_FIELDS & evidence.facts.keys())
    coverage = len(accepted) / denominator if denominator else 0.0
    return InvestigationReport(
        evidence=evidence,
        engine=engine,
        status="complete"
        if coverage == 1
        else ("partial" if accepted else "insufficient_evidence"),
        accepted=accepted,
        rejected=rejected,
        coverage=coverage,
        latency_seconds=latency,
        generated_at=datetime.now(UTC).isoformat(),
    )


def template_report(evidence: EvidencePacket) -> InvestigationReport:
    claims = [
        Claim(field=field, value=value, evidence_id=evidence.evidence_id)
        for field, value in evidence.facts.items()
        if field in CORE_FIELDS
    ]
    return make_report(evidence, claims, "deterministic-template")


def report_markdown(report: InvestigationReport) -> str:
    evidence = report.evidence
    lines = [
        "# OrbitWatch investigation",
        "",
        f"**Evidence:** `{evidence.evidence_id}`  |  **Run:** `{evidence.run_id}`",
        f"**Detector:** {evidence.detector}  |  **Mode:** {report.engine}",
        f"**Report status:** {report.status}  |  **Factual coverage:** {report.coverage:.0%}",
        "",
        "## Evidence-backed observations",
        "",
    ]
    if not report.accepted:
        lines.append("Insufficient supported claims. Inspect the numerical evidence directly.")
    for claim in report.accepted:
        value = f"{claim.value:.6g}" if isinstance(claim.value, float) else str(claim.value)
        lines.append(f"- **{FIELD_LABELS[claim.field]}:** {value}. [{evidence.evidence_id}]")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This is a statistical alert, not a verified spacecraft fault. The anonymized data "
            "cannot establish a physical root cause. No spacecraft commands or operational "
            "actions are issued.",
            "",
            f"Threshold: {evidence.threshold:.6g}; peak score: {evidence.peak_score:.6g}. "
            "These are residual magnitudes, not fault probabilities.",
        ]
    )
    if evidence.ongoing_at_cursor:
        lines.extend(
            ["", "The event is ongoing at the replay cursor; its eventual end is unknown."]
        )
    if report.rejected:
        lines.extend(["", "## Rejected draft claims", ""])
        for item in report.rejected:
            lines.append(f"- `{item.claim.field}`: {item.reason}")
    lines.extend(["", "## Limitations", "", *[f"- {item}" for item in evidence.limitations]])
    return "\n".join(lines)


def ollama_models(base_url: str = "http://127.0.0.1:11434") -> list[str]:
    with httpx.Client(timeout=5) as client:
        response = client.get(f"{base_url}/api/tags")
        response.raise_for_status()
    return [item["name"] for item in response.json()["models"]]


def generate_draft(
    evidence: EvidencePacket,
    model: str = "qwen2.5:3b",
    *,
    grounded: bool = True,
    base_url: str = "http://127.0.0.1:11434",
) -> ClaimDraft:
    if grounded:
        instruction = (
            "Select useful facts from the supplied evidence. Return claims only about fields "
            "actually present in facts. Copy their exact values and evidence_id. Do not infer "
            "physical causes. If a fact is missing, omit it. Include all available permitted facts."
        )
    else:
        instruction = (
            "Write a spacecraft anomaly investigation as structured factual claims. Include the "
            "channel, event interval, peak observation, prediction, score ratio, and your best "
            "physical-cause interpretation when possible. Reference the supplied evidence_id."
        )
    prompt = (
        f"{instruction}\nPermitted fields: {', '.join(sorted(CORE_FIELDS))}.\n"
        f"Evidence JSON:\n{evidence.model_dump_json()}"
    )
    with httpx.Client(timeout=180) as client:
        response = client.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "system": "You are a research telemetry assistant. Return only the requested JSON object.",
                "format": ClaimDraft.model_json_schema(),
                "stream": False,
                "options": {"temperature": 0, "seed": 17, "num_ctx": 4096, "num_predict": 1200},
            },
        )
        response.raise_for_status()
    payload = response.json()
    if payload.get("done") is not True:
        raise ValueError("Local model returned an incomplete response.")
    return ClaimDraft.model_validate_json(payload["response"])


def canonical_json(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
