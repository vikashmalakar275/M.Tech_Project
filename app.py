from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from orbitwatch.config import DATASET_CAVEAT, project_root
from orbitwatch.evidence import (
    generate_draft,
    make_report,
    ollama_models,
    report_markdown,
    template_report,
)
from orbitwatch.mcp_client import investigate_via_mcp
from orbitwatch.metrics import intervals
from orbitwatch.service import TelemetryService
from orbitwatch.store import InvestigationStore

st.set_page_config(page_title="OrbitWatch | Telemetry Intelligence", page_icon="🛰️", layout="wide")
st.markdown(
    """
    <style>
    .stApp {background: #08111f; color: #dce7f3;}
    [data-testid="stSidebar"] {background: #0d192b; border-right: 1px solid #21344d;}
    .block-container {padding-top: 2rem; max-width: 1600px;}
    h1, h2, h3 {letter-spacing: -0.025em;}
    [data-testid="stMetric"] {background:#102038; border:1px solid #233b57;
        padding:16px; border-radius:12px;}
    [data-testid="stMetricLabel"] {color:#98b1ca;}
    .eyebrow {color:#48ddbd; font-size:12px; letter-spacing:0.18em; font-weight:700;}
    .hero {padding: 8px 0 24px 0;}
    .hero h1 {font-size:46px; margin:0; color:#f0f6fc;}
    .hero p {color:#9db3ca; max-width:950px; font-size:16px;}
    .status {display:inline-block; background:#153d35; color:#74edcb; border:1px solid #256455;
        padding:6px 12px; border-radius:30px; font-size:11px; letter-spacing:0.06em;}
    </style>
    """,
    unsafe_allow_html=True,
)
ROOT = project_root()


def completed_runs(root: Path) -> list[str]:
    runs = []
    for path in sorted((root / "runs").glob("*/manifest.json"), reverse=True):
        manifest = json.loads(path.read_text())
        if manifest.get("status") == "complete":
            runs.append(path.parent.name)
    return runs


@st.cache_data
def read_scores(run_path: str, channel: str) -> dict[str, np.ndarray]:
    with np.load(Path(run_path) / channel / "scores.npz", allow_pickle=False) as source:
        return {name: source[name] for name in source.files}


st.markdown(
    '<div class="hero"><div class="eyebrow">SPACECRAFT HEALTH / EVIDENCE-FIRST AI</div>'
    "<h1>OrbitWatch</h1><p>From telemetry to traceable insight. Detect unusual behaviour, "
    "inspect the evidence, and understand exactly what the system can—and cannot—conclude.</p>"
    '<span class="status">RECORDED NASA TELEMETRY · LOCAL RESEARCH PROTOTYPE</span></div>',
    unsafe_allow_html=True,
)
runs = completed_runs(ROOT)
if not runs:
    st.info(
        "The application is installed. Download the benchmark and complete an experiment to begin."
    )
    st.code(
        "orbitwatch download\norbitwatch train --channels A-1 C-1 --run-id quickstart",
        language="bash",
    )
    st.stop()

with st.sidebar:
    st.markdown("### Mission workspace")
    latest_path = ROOT / "runs" / "latest.json"
    latest = json.loads(latest_path.read_text())["run_id"] if latest_path.exists() else runs[0]
    run = st.selectbox(
        "Completed experiment", runs, index=runs.index(latest) if latest in runs else 0, key="run"
    )
    service = TelemetryService(ROOT, run)
    catalog = service.list_channels()
    mission = st.selectbox(
        "Mission", sorted({item["spacecraft"] for item in catalog}), key="mission"
    )
    channel = st.selectbox(
        "Telemetry channel",
        [item["channel"] for item in catalog if item["spacecraft"] == mission],
        key="channel",
    )
    detector = st.selectbox(
        "Detector",
        ["gru", "rolling_median"],
        format_func=lambda x: {
            "gru": "GRU forecaster",
            "rolling_median": "Rolling-median baseline",
        }[x],
        key="detector",
    )
    st.divider()
    st.markdown("**Execution boundary**")
    st.caption(
        "Local CPU detectors. Optional local LLM. No live spacecraft connection or commanding."
    )
    st.caption("Thresholds are calibrated on held-out training data, never on test labels.")
    st.caption(f"Experiment seed: {service.manifest['config']['seed']}")

metadata = service.channel_metadata(channel)
arrays = read_scores(str(service.run_dir), channel)
metrics = pd.read_csv(service.run_dir / "metrics.csv")
store = InvestigationStore(ROOT / "local" / "investigations.sqlite3")
telemetry_tab, benchmark_tab, methods_tab, history_tab = st.tabs(
    ["MISSION CONSOLE", "EXPERIMENT RESULTS", "METHOD & PROVENANCE", "INVESTIGATION HISTORY"]
)

with telemetry_tab:
    left, right = st.columns([3, 1])
    with left:
        st.subheader(f"{mission} / {channel}")
        st.caption(
            "Replay of precomputed causal forecasts. Playback never reveals future samples or future alerts."
        )
    with right:
        auto = st.toggle("Auto-advance replay", value=False)
        step = st.select_slider("Samples per step", options=[25, 100, 250, 500], value=100)
    cursor_key = f"cursor_{run}_{channel}"
    if cursor_key not in st.session_state:
        st.session_state[cursor_key] = len(arrays["observed"]) - 1

    def reset_cursor() -> None:
        st.session_state[cursor_key] = metadata["warmup_samples"]

    def advance_cursor() -> None:
        st.session_state[cursor_key] = min(
            len(arrays["observed"]) - 1, st.session_state[cursor_key] + step
        )

    def finish_cursor() -> None:
        st.session_state[cursor_key] = len(arrays["observed"]) - 1

    buttons = st.columns(3)
    buttons[0].button(
        "Restart replay", on_click=reset_cursor, use_container_width=True, key="restart"
    )
    buttons[1].button(
        "Advance one step", on_click=advance_cursor, use_container_width=True, key="advance"
    )
    buttons[2].button(
        "Show complete sequence", on_click=finish_cursor, use_container_width=True, key="finish"
    )

    @st.fragment(run_every=1.0 if auto else None)
    def mission_console() -> None:
        if auto:
            advance_cursor()
        cursor = st.slider(
            "Replay cursor (inclusive sample index)",
            min_value=metadata["warmup_samples"],
            max_value=len(arrays["observed"]) - 1,
            key=cursor_key,
        )
        threshold = metadata["thresholds"][detector]
        scores = arrays[f"{detector}_scores"]
        events = service.events(channel, detector, cursor)
        current_ratio = scores[cursor] / threshold
        stats = st.columns(4)
        stats[0].metric("Visible observations", f"{cursor + 1:,}")
        stats[1].metric("Alert intervals", f"{len(events):,}")
        stats[2].metric("Current residual / threshold", f"{current_ratio:.2f}x")
        stats[3].metric("Current status", "ALERT" if current_ratio > 1 else "NO ALERT")
        options = st.columns([2, 1])
        context = options[0].select_slider(
            "Visible plot history", options=[256, 512, 1024, 2048, 4096], value=1024
        )
        show_labels = options[1].checkbox("Show evaluation labels", value=False)
        start = max(0, cursor - context + 1)
        x = np.arange(start, cursor + 1)
        figure = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09, row_heights=[0.65, 0.35]
        )
        figure.add_trace(
            go.Scattergl(
                x=x,
                y=arrays["observed"][start : cursor + 1],
                name="Observed",
                line={"color": "#58c6ff", "width": 1.5},
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scattergl(
                x=x,
                y=arrays[f"{detector}_prediction"][start : cursor + 1],
                name="Predicted",
                line={"color": "#f6be62", "width": 1.2},
            ),
            row=1,
            col=1,
        )
        local_scores = scores[start : cursor + 1]
        alert = local_scores > threshold
        figure.add_trace(
            go.Scattergl(
                x=x[alert],
                y=arrays["observed"][start : cursor + 1][alert],
                name="Alert",
                mode="markers",
                marker={"color": "#ff7189", "size": 5},
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scattergl(
                x=x,
                y=local_scores / threshold,
                name="Residual / threshold",
                line={"color": "#45dbb3"},
                fill="tozeroy",
                fillcolor="rgba(69,219,179,0.08)",
            ),
            row=2,
            col=1,
        )
        figure.add_hline(y=1, line_dash="dash", line_color="#ff7189", row=2, col=1)
        if show_labels:
            for begin, end in intervals(arrays["labels"][start : cursor + 1], start):
                figure.add_vrect(
                    x0=begin - 0.5,
                    x1=end + 0.5,
                    fillcolor="#ab85ed",
                    opacity=0.13,
                    line_width=0,
                    row=1,
                    col=1,
                )
        figure.update_layout(
            height=490,
            template="plotly_dark",
            paper_bgcolor="#08111f",
            plot_bgcolor="#0d192b",
            margin={"l": 12, "r": 12, "t": 25, "b": 10},
            legend={"orientation": "h", "y": 1.12},
            hovermode="x unified",
        )
        figure.update_yaxes(title_text="Dataset-scaled value", row=1, col=1)
        figure.update_yaxes(title_text="Threshold ratio", row=2, col=1)
        figure.update_xaxes(title_text="Sample index (not mission time)", row=2, col=1)
        st.plotly_chart(figure, use_container_width=True)
        if show_labels:
            st.caption(
                "Purple shading is the released evaluation label, not an input to the detector or explanation."
            )
        st.subheader("Evidence desk")
        st.caption(
            "Choose an observed alert. The investigation request travels through a real MCP stdio client/server session."
        )
        if not events:
            st.info(
                "No above-threshold alert has occurred at this cursor. Advance playback or select another channel."
            )
            return
        ranked = sorted(events, key=lambda event: event["peak_ratio"], reverse=True)
        selection = st.selectbox(
            "Visible alert to investigate",
            range(len(ranked)),
            format_func=lambda i: f"Samples {ranked[i]['start']}–{ranked[i]['end']} | peak {ranked[i]['peak_ratio']:.2f}x | {ranked[i]['duration']} samples",
        )
        event = ranked[selection]
        engine = st.radio(
            "Report engine",
            ["Deterministic evidence report", "Evidence-validated local LLM"],
            horizontal=True,
        )
        model = st.text_input(
            "Ollama model", value="qwen2.5:3b", disabled=engine.startswith("Deterministic")
        )
        key = f"{run}:{channel}:{detector}:{event['start']}:{event['end']}:{cursor}:{engine}"
        if st.button("Investigate through MCP", type="primary", disabled=auto, key="investigate"):
            with st.spinner("Retrieving traceable evidence through MCP..."):
                try:
                    evidence, trace = asyncio.run(
                        investigate_via_mcp(
                            ROOT,
                            channel,
                            event["start"],
                            event["end"],
                            detector,
                            cursor,
                            run,
                        )
                    )
                    if engine.startswith("Deterministic"):
                        report = template_report(evidence)
                    else:
                        installed = ollama_models()
                        if model not in installed:
                            raise ValueError(
                                f"Model {model!r} is not installed. Run `ollama pull {model}`."
                            )
                        started = time.perf_counter()
                        draft = generate_draft(evidence, model)
                        report = make_report(
                            evidence, draft.claims, f"ollama/{model}", time.perf_counter() - started
                        )
                    store.save(report)
                    st.session_state["investigation"] = (key, report, trace)
                except (httpx.HTTPError, ValueError, OSError, ExceptionGroup) as error:
                    st.error(f"Investigation failed; no fallback report was substituted. {error}")
        saved = st.session_state.get("investigation")
        if saved and saved[0] == key:
            _, report, trace = saved
            report_text = report_markdown(report)
            st.markdown(report_text)
            download = st.columns(2)
            download[0].download_button(
                "Download investigation (.md)",
                report_text,
                file_name=f"{report.evidence.evidence_id}.md",
                mime="text/markdown",
            )
            download[1].download_button(
                "Download evidence (.json)",
                report.model_dump_json(indent=2),
                file_name=f"{report.evidence.evidence_id}.json",
                mime="application/json",
            )
            with st.expander("MCP request trace"):
                st.json(trace)
        with st.expander(f"All {len(events)} visible alert intervals"):
            st.dataframe(pd.DataFrame(events), use_container_width=True, hide_index=True)

    mission_console()

with benchmark_tab:
    st.subheader("Measured results, not illustrative scores")
    st.caption(
        "Fixed chronological calibration, channel-safe windows, untouched test labels, no point adjustment."
    )
    summary = pd.DataFrame(json.loads((service.run_dir / "summary.json").read_text()))
    aggregate = summary[summary["spacecraft"] == "ALL"]
    chart = go.Figure()
    for column, label, color in (
        ("macro_point_f1", "Macro point F1", "#58c6ff"),
        ("macro_average_precision", "Macro average precision", "#45dbb3"),
        ("event_recall", "Pooled event recall", "#f6be62"),
    ):
        chart.add_bar(name=label, x=aggregate["detector"], y=aggregate[column], marker_color=color)
    chart.update_layout(
        template="plotly_dark",
        paper_bgcolor="#08111f",
        plot_bgcolor="#0d192b",
        height=340,
        yaxis={"range": [0, 1]},
        barmode="group",
    )
    st.plotly_chart(chart, use_container_width=True)
    st.dataframe(summary, hide_index=True, use_container_width=True)
    st.info(
        "A small GRU is not assumed to outperform the statistical baseline. False alarms and missed events remain part of the result."
    )
    st.subheader("Channel-level inspection")
    st.dataframe(metrics, hide_index=True, use_container_width=True)
    st.download_button(
        "Download measured metrics CSV",
        metrics.to_csv(index=False),
        file_name="orbitwatch_metrics.csv",
        mime="text/csv",
    )
    explanations_path = service.run_dir / "explanations" / "summary.json"
    if explanations_path.exists():
        explanations = json.loads(explanations_path.read_text())
        st.subheader("Local explanation pilot")
        st.caption(f"Actual local model: {explanations['model']}")
        st.dataframe(
            pd.DataFrame(explanations["summary"]), hide_index=True, use_container_width=True
        )
        st.warning(
            "Validated output is restricted to known fact fields. Zero unsupported emitted claims is a structural guarantee within that schema, not general LLM truthfulness."
        )
        with st.expander("Pilot limitations"):
            for limitation in explanations["limitations"]:
                st.write(limitation)
    else:
        st.info(
            "No local-LLM comparison has been run for this experiment. No explanation scores are invented."
        )

with methods_tab:
    st.subheader("A transparent research pipeline")
    st.code(
        "NASA channel files -> chronological fit/calibration -> causal forecasts\n"
        "  -> fixed residual threshold -> alert intervals -> evidence packet\n"
        "  -> MCP tools -> template or local LLM -> claim checks -> cited report",
        language=None,
    )
    st.markdown(
        "**Research question:** Can evidence-constrained investigation suppress unsupported numerical and causal claims while retaining useful factual coverage?"
    )
    st.markdown(
        "**What is not claimed:** a new GRU architecture, state-of-the-art detection, real mission integration, causal fault diagnosis, or certified onboard deployment."
    )
    st.warning(DATASET_CAVEAT)
    st.markdown(
        "**Published-model boundary:** the historical root-level `Model.py` is preserved but is not used as a faithful USAD, TranAD, or GDN implementation."
    )
    st.json(service.manifest)
    st.markdown(
        "[Original NASA benchmark](https://github.com/khundman/telemanom) · "
        "[Rigorous TSAD evaluation](https://arxiv.org/abs/2109.05257) · "
        "[SigLLM related work](https://github.com/sintel-dev/sigllm)"
    )

with history_tab:
    st.subheader("Local investigation history")
    st.caption(
        "Reports and MCP audit events are stored locally in SQLite. Nothing is sent to an external LLM API."
    )
    history = store.recent()
    if history:
        st.dataframe(pd.DataFrame(history), hide_index=True, use_container_width=True)
    else:
        st.info("Investigate an alert from the mission console to create the first report.")
