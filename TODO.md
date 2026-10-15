# Neural Network Design Service: Architectural Roadmap and Critical Debt Ledger

This ledger defines the strict milestone sequence, critical system invariants, and technical deliverables required to evolve `neural-network-design` into a high-throughput hybrid telemetry engine.

---

## 1. Executive Summary and Critical Invariants

Deep neural networks are computationally wasteful and operational liabilities when applied blindly to time-series telemetry. This package exists strictly as a **two-tier hybrid engine**:
1. **Tier 1 (Fast-Path Filter and Statistical Guardrail)**: Microsecond-latency CPU heuristics (EWMA, CUSUM, streaming quantiles, GBDTs) inspect 100% of telemetry spans.
2. **Tier 2 (Deep Contextual Reasoner)**: PyTorch networks are invoked only when multi-service trace topologies exhibit ambiguity that scalar rules cannot resolve.

### Non-Negotiable Gate Invariants
- **Zero Trace Leakage**: No `trace_id` may ever co-exist across training and validation partitions.
- **Strict Physical Causality**: Telemetry partitions must maintain strict chronological ordering ($T_{\text{train}} < T_{\text{val}}$).
- **Trivial Baseline Rule**: No deep neural network checkpoint may be promoted to `ACTIVE` status unless it demonstrates a statistically significant improvement over running majority-class and rolling-mean baselines.
- **Zero Comment Invariant**: All implementation code must strictly maintain zero inline comments, relying on explicit typing and hexagonal module boundaries.

---

## 2. Phased Architectural Progression Table

| Phase | Epic / Focus Area | Critical Deliverables and Technical Tasks | Primary Architectural Risk | Acceptance Criteria and Gate Metric |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 1: Foundation and Statistical Fast-Path** | Wire-Speed Classical Baselines | • Implement streaming EWMA and CUSUM stateful filters in `src/features/telemetry_pipeline/`.<br>• Implement DDSketch / t-digest streaming quantile estimators.<br>• Implement XGBoost tabular baseline port with zero GPU dependency. | Over-allocating memory in long-running streaming quantile windows. | Heuristic baselines evaluate spans in $< 15\,\mu\text{s}$ per record on CPU with $O(1)$ memory. |
| **Phase 2: Streaming Ingestion and Dedicated Worker** | Persistent Streaming Consumer Loop | • Implement dedicated background Kafka worker loop draining `telemetry.spans.raw`.<br>• Enforce distributed tracing W3C context propagation across span micro-batches.<br>• Implement Dead-Letter Queue (DLQ) routing for corrupted telemetry payloads. | Consumer lag accumulation under sudden telemetry burst traffic ($> 50\text{k}$ spans/sec). | End-to-end Kafka-to-pipeline latency $< 100\,\text{ms}$; zero span drops; DLQ captures all unparseable payloads. |
| **Phase 3: Loss Formulations and Checkpoint Storage** | Severe Imbalance and Durability Port | • Implement Focal Loss and class-balanced effective-sample cross-entropy to handle catastrophic failure rates ($< 0.1\%$).<br>• Implement durable `CheckpointStoragePort` (S3/MinIO/GCS and local disk adapter) with SHA-256 integrity verification.<br>• Add automated model rollbacks if inference latency breaches SLA. | Gradient collapse to zero-prediction due to extreme class imbalance in telemetry. | Model achieves recall $> 85\%$ on rare failure classes ($< 0.1\%$ prevalence); checkpoint serialization succeeds with verified checksums. |
| **Phase 4: Dynamic Hybrid Routing** | Uncertainty-Gated Inference Engine | • Implement predictive entropy scoring on incoming telemetry batches.<br>• Implement fast-path router: direct $> 95\%$ of low-entropy spans to Tier 1 classical baselines.<br>• Route only ambiguous/high-entropy topological spans to PyTorch Tensor Core. | High inference latency on cold-start neural model evaluation. | Total GPU compute reduced by $\ge 90\%$; aggregate pipeline P99 latency $< 20\,\text{ms}$. |
| **Phase 5: Topology Modeling via Graph Neural Networks** | Microservice DAG Representation | • Transform multi-span distributed traces into Directed Acyclic Graphs (DAGs).<br>• Implement Graph Convolutional Network (GCN / GAT) module using PyTorch Geometric.<br>• Benchmark GNN cascade-prediction accuracy against linear MLP and XGBoost baselines. | Graph construction overhead dominating execution time over tabular batching. | GNN out-predicts XGBoost by $\ge 15\%$ on multi-hop cascade failures across $> 10$ services. |
| **Phase 6: Continuous Closed-Loop Retraining** | Autonomous Drift Detection and Promotion | • Implement automated drift detection using Kolmogorov-Smirnov and Population Stability Index (PSI) on streaming metrics.<br>• Trigger candidate model training upon confirmed distribution drift.<br>• Enforce automated promotion gate: candidate must beat production model and baseline gate across continuous evaluation benchmark batches. | Model thrashing or training on corrupted distribution anomalies. | Zero-downtime hot-swapping of neural weights; automated rollback on validation regression. |

---

## 3. Detailed Component Backlog and Risk Matrix

| Component | Priority | Status | Description and What Is Actually Required | Risk Level |
| :--- | :--- | :--- | :--- | :--- |
| `TelemetrySplitRules` | P0 (Critical) | **Done** | Pure functional rules enforcing trace-boundary isolation and temporal lookahead prevention. | Low |
| `TrivialBaselineWorkflow` | P0 (Critical) | **Done** | Mathematical verification that neural architectures outperform majority-class frequency and rolling mean. | Low |
| `AdaptiveLossCore` | P0 (Critical) | **Pending** | Focal Loss and class-balanced cost functions required to stop models from predicting zero failures. | High |
| `StreamingBaselinePort` | P0 (Critical) | **Pending** | Wire-speed EWMA, CUSUM, and DDSketch streaming filters to bypass deep inference on standard telemetry. | Medium |
| `KafkaWorkerDaemon` | P1 (High) | **Pending** | Production background process consuming `telemetry.spans.raw` and feeding continuous pipeline batches. | Medium |
| `CheckpointStorageAdapter` | P1 (High) | **Pending** | S3/MinIO/GCS adapter storing model weights with SHA-256 verification and atomic metadata locking. | Medium |
| `UncertaintyRouter` | P1 (High) | **Pending** | Entropy-based gateway directing low-complexity spans to classical baselines and complex spans to neural layers. | Medium |
| `TraceGraphBuilder` | P2 (Medium) | **Planned** | Pure conversion of multi-span OpenTelemetry traces into PyTorch Geometric graph tensors. | High |
| `DriftMonitorDaemon` | P2 (Medium) | **Planned** | Streaming statistical testing (PSI / KS-test) detecting feature drift and triggering closed-loop retraining. | Medium |
| `DistributedRayTrainer` | P3 (Low) | **Planned** | Multi-GPU distributed training using Ray Train or Torchrun for hyperparameter sweeps and heavy topology models. | Medium |
