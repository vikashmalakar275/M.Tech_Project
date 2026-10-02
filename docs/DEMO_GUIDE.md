# Five-minute demonstration

## Before the meeting

Open `Launch.command`. If using the LLM engine, also start `Start_Local_LLM.command`.
The deterministic demonstration does not need Ollama. Use the submitted measured run,
not a new untrained configuration. `submission/demo_events.json` lists actual predicted
events to make navigation easy; it is not a list of guaranteed true faults.

## Demonstration

1. **Explain the boundary.** This is replayed NASA telemetry, not a live spacecraft.
   Values are anonymized and scaled; indices are samples, not wall-clock mission times.
2. **Show the console.** Select a mission, channel, and detector. Restart replay, advance
   the cursor, and enable auto-advance. Show that only already-visible alerts appear.
3. **Inspect an alert.** Pause replay. Show observed/predicted values, residual ratio,
   and the frozen threshold. Evaluation labels can be overlaid but are not detector inputs.
4. **Investigate through MCP.** Choose an event, generate a deterministic report, and
   expand the actual discovery/tool-call trace. Explain the evidence ID and export JSON.
5. **Switch to the local model.** Show accepted/rejected claims and factual coverage.
   Explain that the model is not allowed to turn anonymous residuals into a hardware diagnosis.
6. **Show measured results.** Compare baseline/GRU tradeoffs and false alarms. Open the
   explanation comparison, including the no-LLM baseline and paired validator ablation.

## What not to say

- Do not call the replay a real-time spacecraft integration.
- Do not call a statistical alert a confirmed hardware failure.
- Do not claim the original project's 96.9% F1 or RAM-reduction table was reproduced.
- Do not claim zero unrestricted hallucinations; only specified factual fields are checked.
- Do not imply the workflow autonomously commands a spacecraft or uses multiple agents.

## Deliverables to open

- `submission/Technical_Report.pdf`
- `submission/Final_Presentation.pptx`
- `submission/detection_metrics.csv`
- `submission/explanation_records.json`, if the local-model pilot has completed
- The code and tests on GitHub
