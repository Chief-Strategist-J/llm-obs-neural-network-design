# Neural Network Design Service & Hybrid Telemetry Engine

Neural Network Design architecture specification, topology validation, parameter estimation, and hybrid statistical/deep telemetry analytics for LLM Observability.

---

## 1. Executive Overview: Classical Baselines vs. Deep Neural Networks

A core architectural tenet of this service is that **deep neural networks should never operate in isolation for distributed systems telemetry**.

[Certain] For streaming distributed telemetry (such as trace cascade failures, temporal worker lag, latency spikes, and microservice anomaly detection), classical statistical baselines (streaming quantile estimation, Exponentially Weighted Moving Average [EWMA], Cumulative Sum [CUSUM], and Gradient Boosted Decision Trees like XGBoost):
1. **Outperform or match custom deep neural networks** on tabular and metric time-series data.
2. **Require 1/100th of the operational compute budget**, running with microsecond-level CPU latency and zero GPU cluster infrastructure.
3. **Offer 100% deterministic explainability**, eliminating black-box uncertainty during mission-critical incident triage.

### The Hybrid Architecture Strategy
This package implements a **two-tier hybrid inference pipeline**:
- **Tier 1 (Fast-Path Filter & Invariant Gate)**: Lightweight streaming statistical baselines (EWMA, streaming quantiles, CUSUM) evaluate 100% of telemetry spans in real time, detecting known distribution shifts and abrupt breaches at wire speed.
- **Tier 2 (Deep Contextual Reasoner)**: The PyTorch neural network is triggered selectively on ambiguous, multi-span topological traces where high-dimensional cross-service interactions cannot be captured by scalar heuristics.

No neural checkpoint is promoted to production unless its validation loss and F1 score decisively exceed the trivial heuristic baseline gate.

---

## 2. Directory Architecture & Boundaries

Strict adherence to hexagonal architectural separation, shared utilities reuse, and zero inline comments:

- `contracts/`: OpenAPI schemas and telemetry event definitions.
- `config/`: Declarative YAML/JSON runtime configurations and schema constraints.
- `messaging/`: Streaming topic definitions (Kafka `telemetry.spans.raw`) and DLQ policies.
- `deploy/`: Container and Kubernetes manifests with verified dependency layers.
- `adr/`: Architecture Decision Records documenting mathematical foundations and evaluation rules.
- `src/shared/`: Canonical application errors, envelope models, trace constants, and deterministic utilities.
- `src/infra/`: OpenTelemetry tracing infrastructure and Confluent Kafka adapters.
- `src/features/network_spec/`: Topology validation rules, lifecycle state machines, and PyTorch execution engine.
- `src/features/telemetry_pipeline/`: Physical causality verification and leak-free split rules.
- `src/api/`: REST routing and API handlers.
- `tests/`: Automated unit and integration test suites.
