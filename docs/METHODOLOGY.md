# Methodology and reproducibility

## Scope

OrbitWatch investigates the boundary between numerical anomaly evidence and generated
explanations. It is an experimental systems project, not a proposal for a new GRU cell,
a new communication protocol, or verified spacecraft root-cause diagnosis.

## Data contract

Use the NASA SMAP/MSL benchmark mirrored by `patrickfleith` on Kaggle and the original
Telemanom `labeled_anomalies.csv`. The downloader records both source URLs, the archive
SHA-256, the labels SHA-256, retrieval time, and the channel count.

Each `.npy` file contains a target telemetry series in column zero and additional encoded
command features. Files have different lengths and must not be treated as synchronized
physical sensors or joined end-to-end. Anomaly ranges are inclusive.

The original metadata has 82 rows but **81 unique files**: SMAP `P-2` appears twice with
overlapping annotations `[5350,6575]` and `[5300,6420]`. Before evaluation, OrbitWatch
uses their union `[5300,6575]`, counts that file once, and records this decision in the
manifest. This means 54 unique SMAP files and 27 MSL files, not 82 independent streams.
Conflicting missions or sequence lengths in duplicate records fail explicitly.

The archive installer only writes expected channel filenames and validates shapes, finite
values, metadata, and label ranges. Existing nonempty unverified data are not overwritten.

The released data were already scaled using test-set extrema, according to the original
README. This inherited preprocessing cannot be undone. The new scaler is fitted only on
the fitting segment, but the benchmark is not described as completely free of upstream
test-distribution information.

## Fitting and calibration

For each channel independently:

1. Split the supplied training file chronologically: first 80% for weight/scaler fitting;
   final 20% for calibration.
2. Standardize using mean and standard deviation from the fitting segment. Constant
   features use unit scale instead of division by zero.
3. A target at index `t` uses exactly `[t-window, t)`; it never includes the target or future.
4. Fit a one-layer GRU and linear readout using Smooth L1 loss, AdamW, clipped gradients,
   a fixed seed, and a bounded evenly spaced subset of fitting windows.
5. Forecast calibration targets. Their context may include preceding training observations,
   but their target values do not contribute to optimization.
6. Set each detector's residual threshold to the specified calibration quantile.
7. Evaluate the supplied test file without joining it to the training timeline. The first
   `window` test samples are explicit warmup.

The baseline predicts the median of the previous target window. The GRU consumes all
available features but forecasts only the target telemetry. Neither model is called USAD,
TranAD, or GDN.

`score(t) = abs(standardized_observed(t) - standardized_predicted(t))`

An alert is `score(t) > threshold`. Contiguous alerts form an inclusive event interval.
Threshold ratios are residual magnitudes, **not** calibrated probabilities or fault severity.

## Metrics

- Point precision, recall, F1: no label-aware postprocessing or point adjustment.
- Average precision: a threshold-independent ranking metric, distinct from trapezoidal
  integration of a precision-recall curve.
- Event precision/recall/F1: chronological one-to-one interval-overlap matching. Each
  prediction and truth interval can match once. Additional fragments count as extra events.
- Delay: `max(0, predicted_start - true_start)` for the matched event, in samples.
  Missed events are represented in recall, not assigned an arbitrary finite delay.
- False-positive samples per 1,000 normal samples: complements event metrics and exposes
  models that stay in the alarm state.
- Latency: CPU single-window forward-pass p95 within each channel; not end-to-end mission
  latency or power consumption.

Aggregate point F1 and AP are macro averages across channels. Aggregate event counts
are pooled before computing precision/recall. Mean channel-p95 latency is not a pooled
p95 and is labelled accordingly. Raw per-channel data remain available.

No-positive cases have undefined AP (`null`), not invented perfect scores. Zero predicted
events use zero precision/recall where denominators vanish, as documented by the tests.

## Evidence contract

Evidence carries the run ID, channel, detector, event indices, peak observation and
prediction, calibrated threshold, peak score and ratio, source fingerprint, and explicit
limitations. It excludes evaluation labels. IDs are stable traceability hashes, **not**
cryptographic signatures or proof of authenticity.

At a replay cursor, event extraction and evidence stop at that cursor. If an event reaches
the cursor, its eventual end is unknown and the report marks it as ongoing.

Permitted claims are channel, start/end/peak sample, peak ratio, peak observation,
peak prediction, and `physical_cause = unknown`. Citation IDs and text must match;
sample indices match exactly; floating measurements tolerate at most `1e-6`.
Missing facts, unsupported fields, invented causes, incorrect numbers, and duplicates
are rejected and retained for inspection. Empty accepted output is explicitly
`insufficient_evidence`, not a successful generic summary.

## Explanation experiment

Interleave missions, select one midpoint-in-time predicted event per sorted channel, and
create paired complete/redacted cases without consulting labels. Redaction withholds the
peak predicted value. Both prompts receive the same evidence under each condition.

Compare:

1. Deterministic template.
2. Ordinary local-model structured draft.
3. The exact same ordinary draft passed through validation.
4. Evidence-copying prompt plus validation.

The paired third arm isolates validator filtering without additional model sampling.
The fourth changes both prompting and filtering. Record proposals, rejections, emitted
facts, factual coverage, and generation latency, including the complete raw case records.

Generation latency is wall-clock LLM draft generation only, excluding validation and
rendering. The paired validator arm reuses the ordinary draft's measured generation
time; it is not a separate end-to-end timing measurement. Template generation was not
timed and is reported as `null` / `n/a`, not an exactly measured zero execution cost.

This evaluates a restricted fact-selection/reporting workflow, not unrestricted prose
truthfulness. Template correctness is a strong baseline. A zero error rate for accepted
claims is structurally enforced, not independent proof that an LLM understood spacecraft
engineering. Cases from the same event are not independent samples.

## Reproducibility artifacts

Each completed run includes configuration, runtime/package versions, source fingerprint,
dataset provenance, channel metadata, training losses, scaler/model checkpoint,
per-sample predictions/scores, full metric CSV, and aggregate JSON.

The source fingerprint covers package Python files at experiment start. An experiment's
manifest remains the record of that run even if unrelated application/report code later
changes. The Git history identifies the submitted implementation.

The computational snapshot for `nasa-full-seed17` is preserved in
[commit 2a5b5cc](https://github.com/vikashmalakar275/M.Tech_Project/commit/2a5b5cc25f132f0a9cd2b9a58119a3e0c7301a8f).
Its package fingerprint matches the recorded manifest. Subsequent reporting corrections
replace unmeasured template timing placeholders with null and clarify timing scope;
the detector results, generated claims, and measured LLM timings are unchanged.

Negative findings remain part of the report. The generator does not insert literature
accuracy tables or promise that a deep model beats a simple baseline.
