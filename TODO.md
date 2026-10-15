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
- **Vectorized Preprocessing Invariant**: Multi-gigabyte offline batch cleaning must use Polars streaming expressions with zero Python GIL lock-in and zero NaN/Inf leaks.
- **Verification Matrix Invariant**: Every feature must maintain passing unit, integration, end-to-end, and performance benchmark suites.
- **Zero Comment Invariant**: All implementation code must strictly maintain zero inline comments, relying on explicit typing and hexagonal module boundaries.

---

## 2. Phased Architectural Progression Table

| Phase | Epic / Focus Area | Critical Deliverables and Technical Tasks | Status | Acceptance Criteria and Gate Metric |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 1: Foundation and Statistical Fast-Path** | Wire-Speed Classical Baselines and Preprocessing | • Causal telemetry split with zero trace leakage.<br>• Trivial majority-class and rolling-mean baseline gate.<br>• Polars streaming batch data sanitization with outlier Winsorization and Z-score normalization.<br>• PyTorch architecture specification and tiny-batch overfit convergence gate. | **Completed** | Full cross-feature execution, leak-free mathematical split, Polars throughput $\ge 100\text{k}$ records/s, PyTorch loss convergence $< 0.08$. |
| **Phase 2: Comprehensive Test and Ingress Hardening** | Multi-Layer Test Hierarchy and Centralized Ingress | • Unit test suites across all three feature modules.<br>• Integration tests verifying cross-feature pipeline handoff, repository persistence, and Kafka records.<br>• End-to-end REST API lifecycle tests and standard error response envelopes.<br>• Performance benchmark tests for Polars scaling and PyTorch CPU inference latency.<br>• Centralized `config/` isolation and Docker Compose configuration. | **Completed** | 29/29 passing tests across unit, integration, e2e, and performance suites; live curl verification on port 8080. |
| **Phase 3: Streaming Ingestion and Dedicated Worker** | Persistent Streaming Consumer Loop | • Implement dedicated background Kafka worker loop draining `telemetry.spans.raw`.<br>• Enforce distributed tracing W3C context propagation across span micro-batches.<br>• Implement Dead-Letter Queue (DLQ) routing for corrupted telemetry payloads. | **Pending** | End-to-end Kafka-to-pipeline latency $< 100\,\text{ms}$; zero span drops; DLQ captures all unparseable payloads. |
| **Phase 4: Loss Formulations and Checkpoint Storage** | Severe Imbalance and Durability Port | • Implement Focal Loss and class-balanced effective-sample cross-entropy to handle catastrophic failure rates ($< 0.1\%$).<br>• Implement durable `CheckpointStoragePort` (S3/MinIO/GCS and local disk adapter) with SHA-256 integrity verification.<br>• Add automated model rollbacks if inference latency breaches SLA. | **Pending** | Model achieves recall $> 85\%$ on rare failure classes ($< 0.1\%$ prevalence); checkpoint serialization succeeds with verified checksums. |
| **Phase 5: Dynamic Hybrid Routing** | Uncertainty-Gated Inference Engine | • Implement predictive entropy scoring on incoming telemetry batches.<br>• Implement fast-path router: direct $> 95\%$ of low-entropy spans to Tier 1 classical baselines.<br>• Route only ambiguous/high-entropy topological spans to PyTorch Tensor Core. | **Planned** | Total GPU compute reduced by $\ge 90\%$; aggregate pipeline P99 latency $< 20\,\text{ms}$. |
| **Phase 6: Topology Modeling via Graph Neural Networks** | Microservice DAG Representation | • Transform multi-span distributed traces into Directed Acyclic Graphs (DAGs).<br>• Implement Graph Convolutional Network (GCN / GAT) module using PyTorch Geometric.<br>• Benchmark GNN cascade-prediction accuracy against linear MLP and XGBoost baselines. | **Planned** | GNN out-predicts XGBoost by $\ge 15\%$ on multi-hop cascade failures across $> 10$ services. |
| **Phase 7: Continuous Closed-Loop Retraining** | Autonomous Drift Detection and Promotion | • Implement automated drift detection using Kolmogorov-Smirnov and Population Stability Index (PSI) on streaming metrics.<br>• Trigger candidate model training upon confirmed distribution drift.<br>• Enforce automated promotion gate: candidate must beat production model and baseline gate across continuous evaluation benchmark batches. | **Planned** | Zero-downtime hot-swapping of neural weights; automated rollback on validation regression. |

---

## 3. Detailed Component Backlog and Risk Matrix

| Component | Priority | Status | Description and What Is Actually Required | Risk Level |
| :--- | :--- | :--- | :--- | :--- |
| `TelemetrySplitRules` | P0 (Critical) | **Done** | Pure functional rules enforcing trace-boundary isolation and temporal lookahead prevention. | Low |
| `TrivialBaselineWorkflow` | P0 (Critical) | **Done** | Mathematical verification that neural architectures outperform majority-class frequency and rolling mean. | Low |
| `DataSanitizationRules` | P0 (Critical) | **Done** | Vectorized Polars and NumPy batch cleaning with NaN/Inf imputation, outlier winsorization, and Z-score normalization. | Low |
| `DataSanitizationService` | P0 (Critical) | **Done** | Hexagonal domain service managing sanitization profiles, sample transformations, and trace context propagation. | Low |
| `NetworkSpecService` | P0 (Critical) | **Done** | Architecture specification, constraint validation, tiny-batch overfit verification, and dataset training loop. | Low |
| `PyTorchNeuralArchitecture` | P0 (Critical) | **Done** | Residual feed-forward backbone with GELU activations and LayerNorm for stable gradient propagation. | Low |
| `MultiLayerTestHierarchy` | P0 (Critical) | **Done** | 29 tests spanning unit, cross-feature integration, end-to-end API lifecycle, and performance latency benchmarks. | Low |
| `CentralizedDockerConfig` | P0 (Critical) | **Done** | Unified `.env.example`, `config/env.schema`, `config/default.yaml`, and `deploy/docker/docker-compose.yaml`. | Low |
| `AdaptiveLossCore` | P0 (Critical) | **Done** | Focal Loss and class-balanced cost functions required to stop models from predicting zero failures on imbalanced data. | Low |
| `StreamingBaselinePort` | P0 (Critical) | **Pending** | Wire-speed EWMA, CUSUM, and DDSketch streaming filters to bypass deep inference on standard telemetry. | Medium |

| `KafkaWorkerDaemon` | P1 (High) | **Pending** | Production background process consuming `telemetry.spans.raw` and feeding continuous pipeline batches. | Medium |
| `CheckpointStorageAdapter` | P1 (High) | **Pending** | S3/MinIO/GCS adapter storing model weights with SHA-256 verification and atomic metadata locking. | Medium |
| `UncertaintyRouter` | P1 (High) | **Pending** | Entropy-based gateway directing low-complexity spans to classical baselines and complex spans to neural layers. | Medium |
| `TraceGraphBuilder` | P2 (Medium) | **Planned** | Pure conversion of multi-span OpenTelemetry traces into PyTorch Geometric graph tensors. | High |
| `DriftMonitorDaemon` | P2 (Medium) | **Planned** | Streaming statistical testing (PSI / KS-test) detecting feature drift and triggering closed-loop retraining. | Medium |
| `DistributedRayTrainer` | P3 (Low) | **Planned** | Multi-GPU distributed training using Ray Train or Torchrun for hyperparameter sweeps and heavy topology models. | Medium |
