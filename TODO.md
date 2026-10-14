# Neural Network Design Service: Roadmap & Technical Debt Ledger

This document tracks critical invariants, pending architectural deliverables, and future roadmap items for the `neural-network-design` package.

---

## 1. What Is Actually Critical (Production Gate Invariants)

These items directly safeguard system correctness, physical causality, and prevent production model regressions:

- [x] **Leak-Free Trace-Boundary Splitting**: Guarantee zero `trace_id` overlap between training and validation partitions.
- [x] **Temporal Causality Preservation**: Enforce strict chronological separation ($T_{\text{train}} < T_{\text{val}}$) to prevent future telemetry states from leaking into past inference windows.
- [x] **Trivial Baseline Gate Enforcement**: No neural network specification or checkpoint can transition to `ACTIVE` without beating running heuristic baselines (majority class and rolling mean).
- [ ] **Adaptive Loss Weighting for Catastrophic Imbalance**: Telemetry failures constitute < 0.1% of records. Focal Loss or effective-sample class-balanced weights must be integrated to prevent gradient collapse to zero-prediction.
- [ ] **Streaming Quantile (t-digest / DDSketch) Baselines**: Implement wire-speed CPU baseline evaluators to sit immediately before PyTorch inference.

---

## 2. What Is Currently Pending (Active Backlog)

Items scoped for stabilization and local engine refinement:

- [ ] **Streaming Baseline Adapters**:
  - Add EWMA (Exponentially Weighted Moving Average) and CUSUM (Cumulative Sum) statistical state filters in `src/features/telemetry_pipeline/`.
  - Add tabular decision tree baseline (XGBoost / LightGBM) integration port for structured tabular metric frames.
- [ ] **Checkpoint Serialization & Storage Port**:
  - Implement durable state-checkpoint adapter (`S3` / `GCS` / local artifacts directory) for PyTorch `.pt` models managed by the lifecycle state machine.
- [ ] **Real-time Kafka Consumer Loop**:
  - Implement a persistent background worker thread that drains `telemetry.spans.raw` and pipes batches directly into `TelemetryPipelineService`.

---

## 3. What Needs to Be Done in the Future (Long-Term Scalability Roadmap)

Items required when scaling from single-node training to distributed enterprise scale:

- [ ] **Distributed Data Parallel (DDP) & Ray Integration**:
  - Decouple `PyTorchOverfitEngine` into a distributed worker orchestrator using Ray Train or Torchrun for multi-GPU training.
- [ ] **Graph Neural Network (GNN) for Service Topologies**:
  - Microservice traces form Directed Acyclic Graphs (DAGs). Migrate complex trace inference from simple MLPs to Graph Convolutional Networks (GCN / GAT) where nodes represent services and edges represent RPC spans.
- [ ] **Dynamic Hybrid Routing Engine**:
  - Implement an uncertainty-gated routing policy: route 98% of low-entropy telemetry queries to streaming quantiles/EWMA, and trigger the PyTorch model only when entropy exceeds the threshold.
- [ ] **Automated Concept Drift & Continuous Retraining Daemon**:
  - Detect covariate and concept drift via Kolmogorov-Smirnov / Population Stability Index (PSI) tests on incoming Kafka streams, automatically triggering candidate retraining workflows.
