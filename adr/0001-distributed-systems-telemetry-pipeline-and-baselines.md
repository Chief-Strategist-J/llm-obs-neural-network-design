# ADR 0001: Distributed Systems Telemetry Modeling Foundation, Leak-Free Evaluation, and Trivial Baseline Engine

## Status
Accepted

## Context
When designing neural network architectures for complex distributed systems (e.g. distributed microservice traces, temporal workflows, Kafka consensus, and cascade failure prediction), practitioners routinely fall into the trap of starting with neural layer architecture and hyperparameter tuning rather than data integrity, leak-free temporal splitting, and baseline verification.

Distributed system telemetry records (spans, events, RPC metrics) have critical structural characteristics:
1. High cross-record correlation within a single distributed trace (`trace_id`) or service session (`session_id`).
2. Strict temporal causality where future events cannot inform past states.
3. Severe label noise and rarity in catastrophic cascade events.

Random train/test splits cause severe data leakage across distributed traces and temporal horizons, leading to over-optimistic validation metrics that collapse in production. Furthermore, without evaluating against trivial baselines (majority-class predictors and linear moving baselines), there is no evidence that a complex neural network provides positive signal over minimal heuristics.

## Decision
We establish the `system_telemetry_pipeline` feature inside `neural-network-design` implementing:
1. **Contract-First Task & Modality Modeling**: Ingress contracts accepting structured telemetry record shapes, target labels, and evaluation metrics for distributed systems.
2. **Deterministic Anti-Leakage Rules (`TelemetrySplitRules`)**:
   - Trace and Group Disjointness: Absolute separation of `trace_id` and service nodes between training and evaluation splits ($\text{TrainTraces} \cap \text{ValTraces} = \emptyset$).
   - Temporal Monotonicity: Strict causal split verifying $\max(\text{TrainTimestamp}) \le \min(\text{ValTimestamp})$ to prevent lookahead bias.
3. **Automated Trivial Baseline Engine (`TrivialBaselineWorkflow`)**:
   - Evaluates a zero-weight trivial predictor (empirical majority class frequency and historical prior mean) across the split before any neural architecture is initialized.
   - Computes baseline benchmark scores ($F_1$, MAE, Accuracy) that serve as mandatory gates for neural network training.
4. **Finite State Machine Protocol (`TelemetryPipelineMachine`)**:
   - Strict lifecycle transitions: `SPECIFIED` -> `DATA_AUDITED` -> `SPLIT_VALIDATED` -> `BASELINE_ESTABLISHED`.

## Consequences
- Guarantees zero train/test leakage across distributed trace sessions.
- Forces all future neural architectures to prove superiority against baseline heuristics prior to training.
- Completely adheres to Hexagonal Architecture, SRP, and zero-comment coding contracts.
