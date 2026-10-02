# Viva preparation and limitations

**What is the main contribution?**

A locally reproducible telemetry-investigation system that keeps numerical evidence,
model-generated proposals, validation, and user-facing reports separate. The empirical
comparison measures the cost and factual coverage of that boundary.

**Why MCP rather than direct Python calls?**

MCP provides a standard discoverable interface usable by different hosts. The project
demonstrates a real protocol exchange and audit trail. MCP itself does not improve
detector accuracy, establish truth, or create security automatically.

**Why use a GRU rather than another transformer?**

The final scope prioritizes a small, reproducible causal forecaster and rigorous evaluation
on a local Mac. A more complex model is not automatically better. The baseline comparison
determines what happens under the recorded configuration.

**Is this a new model?**

No. The GRU and rolling median are established. The proposed contribution is the evaluated
system and evidence-validation workflow. Its academic novelty and credit suitability need
supervisor review and a fuller related-work assessment.

**Why not use the existing GDN/TranAD/USAD code?**

The earlier script contains simplified namesakes, not the complete defining architectures
and training procedures. It remains as a historical artifact but is not used to claim a
published-model reproduction.

**Does the system find root causes?**

No verified physical causes are available in these anonymized datasets. It characterizes
where and how forecasts deviate. Correlation, attention, and large residuals do not
establish causation.

**What does the threshold ratio mean?**

A score divided by a calibration threshold. A ratio above one triggers an alarm.
It is not the probability of spacecraft failure or a mission-defined severity.

**Why not compare with the old high F1 table?**

Different implementations, thresholds, splits, aggregation, and point-adjustment rules
can make such comparisons invalid. This project reports its own unadjusted metrics
with complete provenance.

**Can the explanation checker miss errors?**

It cannot verify the physical truth of an anonymous measurement or causal explanation.
It checks a restricted set of stored facts and references. Bugs in data preparation or
evidence construction remain possible and are addressed through tests and transparency,
not claims of formal verification.

**Why include a deterministic template?**

It is an essential baseline. If the local LLM adds latency without improving useful
coverage, that is a valid negative finding. It prevents presenting a chatbot wrapper
as an automatically superior solution.

**What further experiments would strengthen the thesis?**

Repeated seeds; confidence intervals with the channel/event as the unit of analysis;
predeclared threshold sensitivity; additional datasets; independent expert ratings;
normal/ambiguous investigation scenarios; and measured operator task outcomes.

**Is this ready for operational spacecraft use?**

No. It lacks mission integration, validated physical semantics, deployment qualification,
operational safety assessment, and flight-hardware measurements. Keep it a research tool.

**How was AI used?**

AI assistance was used in implementation and documentation. Understand and verify the
system, disclose assistance under institutional policy, and do not sign statements
claiming otherwise.
