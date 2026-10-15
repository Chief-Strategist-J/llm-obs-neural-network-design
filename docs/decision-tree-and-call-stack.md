# End-to-End System Execution: Decision Tree and Deep Function Call Stack

This document details the complete deterministic decision tree and low-level runtime function call stack for the `neural-network-design` distributed systems telemetry and neural training engine.

---

## 1. End-to-End Execution ASCII Decision Tree

```text
[ START: Ingress Telemetry Micro-Batch Received ]
  |
  +--> [ Gate 1: Network Spec Budget Validation ]
  |      |
  |      +-- (layer_count > 500 AND parameter_budget < 1,000,000) ?
  |      |     |
  |      |     +-- [YES] --> [ ACTION: Raise ValidationError ]
  |      |                   Status: REJECTED
  |      |                   Log: Parameter density too high for requested budget
  |      |                   TERMINATE
  |      |
  |      +-- [NO]  --> [ Proceed to Step 2 ]
  |
  +--> [ Step 2: Streaming Ingestion & Cryptographic Hashing ]
  |      |
  |      +--> Consume batch from Kafka topic: "telemetry.spans.raw"
  |      +--> Extract W3C TraceContext ('traceparent', 'x-request-id')
  |      +--> Parse into immutable Sequence[TelemetryRecordDTO]
  |
  +--> [ Gate 3: Causal Integrity & Trace-Boundary Isolation ]
  |      |
  |      +-- TrainTraces INTERSECT ValTraces != EMPTY ?
  |      |     |
  |      |     +-- [YES] --> [ ACTION: Raise LeakageDetectedError ]
  |      |                   Log: Contamination detected. Same trace_id in Train and Val.
  |      |                   HALT PIPELINE
  |      |
  |      +-- max(TrainTimestamp) > min(ValTimestamp) ?
  |      |     |
  |      |     +-- [YES] --> [ ACTION: Raise LeakageDetectedError ]
  |      |                   Log: Physical causality violation. Temporal lookahead detected.
  |      |                   HALT PIPELINE
  |      |
  |      +-- [NO]  --> [ Proceed to Step 4: Zero Leakage Confirmed ]
  |
  +--> [ Step 4: Trivial Empirical Baseline Benchmark ]
  |      |
  |      +--> Compute empirical prior on training split:
  |      |      Majority Class = argmax(count(y_train == c))
  |      |      Rolling Mean Latency = sum(latency) / N
  |      +--> Evaluate dummy heuristic predictor on validation split
  |      +--> Record Baseline Metrics: { Baseline_Accuracy, Baseline_F1, Baseline_MAE }
  |      +--> Persist baseline benchmark record to database
  |
  +--> [ Gate 5: Tiny-Batch Autograd & Gradient Flow Gate ]
  |      |
  |      +--> Extract 10 to 100 calibration samples
  |      +--> Initialize PyTorch Neural Core (Kaiming uniform weights, LayerNorm, GELU)
  |      +--> Execute 50 epochs of backpropagation on calibration batch
  |      |
  |      +-- final_loss <= 0.08 ?
  |      |     |
  |      |     +-- [NO]  --> [ ACTION: Flag Broken Graph ]
  |      |                   Log: Gradient flow failure, dead activations, or bad LR.
  |      |                   HALT TRAINING (Do not scale compute)
  |      |
  |      +-- [YES] --> [ STATUS: TINY_BATCH_OVERFIT_VERIFIED ]
  |                    Proceed to Step 6
  |
  +--> [ Step 6: Multi-Epoch Production Dataset Training ]
  |      |
  |      +--> Construct TensorDataset and batched DataLoader (shuffle=True)
  |      +--> Loop epochs (1 .. E):
  |      |      +--> Forward pass: encoder -> residual blocks -> linear classification head
  |      |      +--> Compute loss: CrossEntropyLoss or FocalLoss
  |      |      +--> Backward pass: compute adjoint gradients (loss.backward())
  |      |      +--> Apply gradient clipping: clip_grad_norm_(max_norm=1.0)
  |      |      +--> Optimizer update: AdamW step with weight decay (1e-4)
  |      +--> Evaluate validation split under torch.no_grad()
  |      +--> Compute Val_Loss, Val_Accuracy, Val_F1
  |
  +--> [ Gate 7: Trivial Baseline Promotion Gate ]
  |      |
  |      +-- (Candidate_Val_F1 <= Baseline_F1) OR (Candidate_Val_Loss >= Baseline_Loss) ?
  |      |     |
  |      |     +-- [YES] --> [ ACTION: Discard Neural Candidate ]
  |      |                   Log: Neural network failed to outperform trivial heuristic.
  |      |                   Route 100% production traffic to classical baseline.
  |      |                   TERMINATE
  |      |
  |      +-- [NO]  --> [ ACTION: Promote Candidate Checkpoint ]
  |                    Atomically write weights to disk: f"{spec_id}.pt"
  |                    Compute cryptographic SHA-256 integrity digest
  |                    Transition State Machine: STATUS_PENDING/OVERFIT -> STATUS_TRAINED -> STATUS_ACTIVE
  |
  +--> [ Step 8: Two-Tier Hybrid Production Inference ]
  |      |
  |      +--> Inbound streaming telemetry span arrived at wire speed
  |      +--> Tier 1 (CPU Fast-Path): Evaluate streaming EWMA / CUSUM / quantiles
  |      |
  |      +-- Is span ambiguous or multi-service root cause candidate?
  |      |     |
  |      |     +-- [NO]  --> Emit Fast-Path prediction (Latency < 15 microseconds)
  |      |     +-- [YES] --> Invoke Tier 2 Active PyTorch Tensor Checkpoint
  |      |                   Emit Deep Contextual Anomaly Probability
  |
  +--> [ Gate 9: Streaming Concept Drift & Incident Monitoring ]
         |
         +--> Compute Kolmogorov-Smirnov & PSI tests on rolling telemetry windows
         |
         +-- Drift Detected (p < 0.01) OR High Incident False-Positive Rate ?
               |
               +-- [NO]  --> Continue Serving
               +-- [YES] --> [ ACTION: Autonomous Trigger Closed-Loop ]
                             Append edge-case incidents to dataset pool
                             Loop back to Step 2 (Autonomous Retrain Cycle)
```

---

## 2. In-Depth Runtime Function Call Stack

The following trace details the execution stack across all hexagonal layers when invoking dataset training:

```text
[POST /v1/network-spec/train]
 │
 ├── 1. Ingress Delivery Layer (FastAPI Router)
 │    │  File: src/api/rest/v1/routers/network_spec_router.py
 │    │  Function: train_dataset(request: Request, response: Response)
 │    │  Context: Extracts W3C header 'traceparent' -> constructs TraceContext(trace_id=...)
 │    │
 │    └──> 2. API Handler Layer
 │          │  File: src/api/rest/v1/handlers/network_spec_handler.py
 │          │  Function: NetworkSpecHandler.handle_train_dataset(ctx, payload)
 │          │
 │          ├──> 3. Schema Parsing & Bounds Validation
 │          │     │  File: src/features/network_spec/schema/network_spec_schema.py
 │          │     │  Function: NetworkSpecSchema.parse_training_command(payload)
 │          │     │  Assertions:
 │          │     │    - spec_id is valid non-empty string
 │          │     │    - len(train_samples) > 0 and len(val_samples) > 0
 │          │     │    - epochs > 0, batch_size > 0, learning_rate > 0
 │          │     │  Returns: DatasetTrainingCommand (immutable frozen dataclass)
 │          │     │
 │          └──> 4. Domain Service Orchestration (With Tracing Boundary)
 │                │  File: src/features/network_spec/service/network_spec_service.py
 │                │  Function: NetworkSpecService.train_dataset(ctx, command)
 │                │  Tracing: Tracer.start_span("network_spec.train_dataset", ctx) as child_ctx
 │                │  Error Boundary: Structured try / catch(ApplicationError) / finally(span.end)
 │                │
 │                └──> 5. Domain Workflow Layer
 │                      │  File: src/features/network_spec/workflows/network_spec_workflow.py
 │                      │  Function: NetworkSpecWorkflow.execute_dataset_training_flow(ctx, command)
 │                      │
 │                      ├──> 6. Repository Entity Retrieval
 │                      │     │  File: src/features/network_spec/repository/network_spec_repository.py
 │                      │     │  Port Call: NetworkSpecRepositoryPort.find_by_id(ctx, command.spec_id)
 │                      │     │  Returns: NetworkSpecModel (status="PENDING" or "TINY_BATCH_OVERFIT_VERIFIED")
 │                      │     │
 │                      ├──> 7. Compute Driven Port Invocation
 │                      │     │  Port Interface: NeuralComputePort.fit_dataset(...)
 │                      │     │  Adapter Impl: PyTorchNeuralComputeAdapter.fit_dataset(...)
 │                      │     │  File: src/features/network_spec/repository/neural_compute_adapter.py
 │                      │     │
 │                      │     ├──> 8. PyTorch Module Instantiation
 │                      │     │     │  Class: PyTorchNeuralArchitecture(input_dim, hidden_dim, num_classes, layer_count)
 │                      │     │     │  Submodules:
 │                      │     │     │    - Linear(input_dim, hidden_dim) -> LayerNorm -> GELU
 │                      │     │     │    - ResidualBlock(hidden_dim) x layer_count
 │                      │     │     │    - Linear(hidden_dim, num_classes)
 │                      │     │     │  Weight Init: Kaiming uniform (He) on Linear weights, 0.0 on biases
 │                      │     │     │
 │                      │     ├──> 9. Dataset Batching & Loader Setup
 │                      │     │     │  Tensor Conversion: x_train, y_train, x_val, y_val -> torch.Tensor(dtype=float32/long)
 │                      │     │     │  Dataset: TensorDataset(x_train, y_train)
 │                      │     │     │  Loader: DataLoader(train_ds, batch_size=batch_size, shuffle=True)
 │                      │     │     │
 │                      │     ├──> 10. Multi-Epoch Training Loop
 │                      │     │     │  For epoch in range(epochs):
 │                      │     │     │    For batch_x, batch_y in loader:
 │                      │     │     │      ├── optimizer.zero_grad()
 │                      │     │     │      ├── logits = net(batch_x)
 │                      │     │     │      ├── loss = criterion(logits, batch_y)
 │                      │     │     │      ├── loss.backward()
 │                      │     │     │      ├── torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
 │                      │     │     │      └── optimizer.step()
 │                      │     │     │
 │                      │     ├──> 11. Generalization Validation Loop
 │                      │     │     │  net.eval()
 │                      │     │     │  with torch.no_grad():
 │                      │     │     │    val_logits = net(x_val)
 │                      │     │     │    val_loss = criterion(val_logits, y_val).item()
 │                      │     │     │    val_accuracy = (argmax(val_logits) == y_val).sum() / N
 │                      │     │     │
 │                      │     └──> 12. Atomic Weight Serialization & Cryptographic Hashing
 │                      │           │  torch.save({"model_state_dict": net.state_dict(), ...}, ckpt_path)
 │                      │           │  hasher = hashlib.sha256() -> hasher.update(file_bytes)
 │                      │           │  Returns: (epochs, train_loss, val_loss, val_accuracy, ckpt_path, sha256_hex)
 │                      │           │
 │                      ├──> 13. State Machine State Promotion
 │                      │     │  File: src/features/network_spec/machines/network_spec_machine.py
 │                      │     │  Function: NetworkSpecMachine.transition(existing.status, "TRAINED")
 │                      │     │  Validation: Asserts transition is permitted in _VALID_TRANSITIONS table
 │                      │     │  Returns: "TRAINED"
 │                      │     │
 │                      ├──> 14. Repository Entity Update
 │                      │     │  File: src/features/network_spec/repository/network_spec_repository.py
 │                      │     │  Port Call: NetworkSpecRepositoryPort.save(ctx, updated_model)
 │                      │     │  Returns: NetworkSpecModel(status="TRAINED")
 │                      │     │
 │                      └──> 15. DTO Assembly
 │                            │  Returns: tuple[NetworkSpecModel, DatasetTrainingResult]
 │
 └── 16. Response Serialization (Back to Delivery Layer)
      │  File: src/features/network_spec/schema/network_spec_schema.py
      │  Function: NetworkSpecSchema.training_result_to_api(result)
      │  Payload:
      │    {
      │      "success": true,
      │      "data": {
      │        "spec_id": "...",
      │        "trained_epochs": 10,
      │        "train_loss": 0.042,
      │        "val_loss": 0.051,
      │        "val_accuracy": 0.985,
      │        "checkpoint_path": "checkpoints/uuid.pt",
      │        "checkpoint_sha256": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12..."
      │      },
      │      "meta": { "trace_id": "..." }
      │    }
      │
      [HTTP 200 OK Response Emitted to Ingress Client]
```
