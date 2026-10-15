# 0002: Telemetry Data Sanitization, Robust Imputation, and Classification Feature Preprocessing : An Architecture Decision Record (ADR)

- **Status**: Accepted
- **Date**: 2026-10-15
- **Authors**: Platform Architecture and Core Neural Systems Team
- **RFC / Standards**: IEEE 754 (Floating-Point Arithmetic), OpenTelemetry W3C TraceContext RFC, ISO/IEC 25010 (System and Software Quality Models)
- **Deciders**: Lead Systems Architect, Distributed Systems Lead, Machine Learning Infrastructure Engineer

---

## 1. Context and Problem Statement

In distributed systems telemetry, raw metrics and trace observations emitted across thousands of microservices, serverless tasks, and LLM gateways suffer from severe data quality anomalies:
1. **NaN and Infinity Propagation**: Gateway timeouts, zero-division in rate calculators, socket resets, and uninitialized floating-point counters emit `NaN` or `+/-Inf`. Feeding a single un-sanitized `NaN` into a PyTorch neural network corrupts the entire forward graph and causes backward gradient collapse to `NaN` across all weight tensors.
2. **Extreme Outliers and Cold-Start Variance**: Telemetry features (e.g. latency in milliseconds, queue depths, garbage collection pauses) exhibit heavy-tailed distributions with extreme spikes (e.g. $10^5\text{ ms}$ on cold boots vs. $15\text{ ms}$ baseline). Un-clamped outliers dominate mean squared error and cross-entropy gradient vectors, starving standard features of optimization signal.
3. **Distribution Leakage Across Partitions**: Standard ML pipelines fit global mean and variance statistics across the combined dataset. In distributed streaming systems, fitting normalization parameters on validation or future spans leaks distribution statistics back in time, destroying empirical generalization.
4. **Dead and Singular Features**: Dormant microservice flags or un-triggered fault-injection fields remain constant ($\sigma = 0$). Dividing by zero standard deviation produces division errors or singular covariance matrices in downstream models.
5. **Coupling and Policy Drift**: Preprocessing logic hardcoded inside training loops or database ETL scripts cannot adapt as new services or metrics emerge. Sanitization policies must be decoupled into an isolated, versioned hexagonal feature.

---

## 2. Decision Drivers and Core Principles

1. **Strict Upstream Sanitization Invariant**: Zero `NaN`, `+/-Inf`, or un-normalized scalar values may enter neural compute cores. All sanitization must occur before tensor construction.
2. **Causal Fitting Invariant**: Normalization statistics (feature mean $\mu$, standard deviation $\sigma$, and quartile bounds) must be fitted strictly on the causal training partition ($T_{\text{train}}$) and applied as fixed constants to validation and inference streams.
3. **Winsorization and Outlier Clamping**: Extreme tail values are clamped at $k \times \sigma$ standard deviations from the feature mean, preventing catastrophic gradient spikes while preserving ordinal magnitude.
4. **Dead-Feature Masking Invariant**: Features with zero standard deviation ($\sigma < 10^{-8}$) are deterministically clamped to $0.0$, preventing floating-point division by zero.
5. **Hexagonal Boundary Isolation**: Sanitization rules, profiling state machines, and statistics repositories live in a dedicated feature module (`src/features/data_sanitization/`), keeping ingress and training pipelines decoupled.
6. **Zero Comments Doctrine**: All implementation code strictly enforces zero inline comments, relying on self-documenting types, contracts, and pure functions.

---

## 3. High-Level Architecture (HLD)

The Data Sanitization feature sits as an invariant gatekeeper between the Causal Split Engine (`system_telemetry_pipeline`) and the Neural Core (`network_spec`).

### 3.1 High-Level Sanitization Architecture Diagram

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': { 'primaryColor': '#1F2937', 'primaryBorderColor': '#60A5FA', 'primaryTextColor': '#F3F4F6', 'lineColor': '#60A5FA', 'secondaryColor': '#064E3B', 'tertiaryColor': '#78350F', 'mainBkg': '#111827', 'nodeBorder': '#60A5FA'}}}%%
flowchart TD
    RAW["Raw Partitioned Telemetry (Train & Val)"] --> DISPATCHER["Data Sanitization Service"]
    
    subgraph FIT_PHASE["Fit Phase (Training Split Only)"]
        STATS_CALC["DataSanitizationRules.compute_feature_statistics"]
        PROFILE["SanitizationProfileModel (Status: FITTED)"]
        STATS_CALC --> PROFILE
    end
    
    subgraph TRANSFORM_PHASE["Transform Phase (Applied Uniformly)"]
        NAN_IMPUTE["NaN / Inf Imputation Gate"]
        OUTLIER_CLAMP["Outlier Winsorization Gate (k * sigma)"]
        ZSCORE_NORM["Z-Score Normalization ((x - mu) / sigma)"]
        DEAD_MASK["Dead Feature Masking (sigma = 0 -> 0.0)"]
        
        NAN_IMPUTE --> OUTLIER_CLAMP
        OUTLIER_CLAMP --> ZSCORE_NORM
        ZSCORE_NORM --> DEAD_MASK
    end
    
    DISPATCHER -->|1. Fit on Train Split| FIT_PHASE
    PROFILE -->|2. Supply Locked (mu, sigma) Parameters| TRANSFORM_PHASE
    DISPATCHER -->|3. Transform Train & Val Splits| TRANSFORM_PHASE
    TRANSFORM_PHASE --> CLEANED["Cleaned Finite Tensors (CleanedSampleDTO)"]
    CLEANED --> TENSOR_ENGINE["PyTorch Neural Tensor Core"]
```

---

## 4. End-to-End Execution Decision Tree and Function Call Stack

### 4.1 End-to-End Sanitization ASCII Decision Tree

```text
[ START: Telemetry Feature Vector (x_1, ..., x_D) Arrives ]
  |
  +--> [ Gate 1: Dimension Bounds Validation ]
  |      |
  |      +-- len(features) != profile.feature_count ?
  |      |     |
  |      |     +-- [YES] --> [ ACTION: Raise ValidationError ]
  |      |                   Log: Feature dimension mismatch against fitted profile
  |      |                   TERMINATE
  |      |
  |      +-- [NO]  --> [ Proceed to Step 2 ]
  |
  +--> [ Gate 2: Classification Target Label Validation ]
  |      |
  |      +-- target_label < 0 ?
  |      |     |
  |      |     +-- [YES] --> [ ACTION: Raise ValidationError ]
  |      |                   Log: Discrete target class index must be non-negative
  |      |                   TERMINATE
  |      |
  |      +-- [NO]  --> [ Proceed to Step 3 ]
  |
  +--> [ Step 3: Per-Feature Iterative Sanitization Loop (col = 1 .. D) ]
  |      |
  |      +--> Read raw value v = features[col]
  |      +--> Retrieve locked stats: mean = mu[col], std = sigma[col], is_constant
  |      |
  |      +--> [ Sub-Gate 3.1: NaN and Inf Imputation ]
  |      |      |
  |      |      +-- math.isnan(v) ?
  |      |      |     +-- [YES] --> v = (is_constant ? config.impute_nan_value : mu[col])
  |      |      |                   Increment imputed_count
  |      |      |
  |      |      +-- math.isinf(v) ?
  |      |            +-- [YES] --> v = copysign(config.impute_inf_value, v)
  |      |                          Increment imputed_count
  |      |
  |      +--> [ Sub-Gate 3.2: Outlier Winsorization and Clamping ]
  |      |      |
  |      |      +-- config.clip_outliers AND NOT is_constant ?
  |      |            |
  |      |            +-- v > mu[col] + k * sigma[col] ? --> v = mu[col] + k * sigma[col]
  |      |            +-- v < mu[col] - k * sigma[col] ? --> v = mu[col] - k * sigma[col]
  |      |
  |      +--> [ Sub-Gate 3.3: Normalization & Zero-Variance Masking ]
  |             |
  |             +-- is_constant == True ?
  |             |     +-- [YES] --> v = 0.0 (Mask dead feature, prevent zero division)
  |             |
  |             +-- config.normalize_zscore == True ?
  |                   +-- [YES] --> v = (v - mu[col]) / sigma[col]
  |
  +--> [ Step 4: Emit Sanitized Tensor Representation ]
         |
         +--> Construct CleanedSampleDTO(sample_id, tuple(cleaned_features), label, imputed_count)
         +--> Assert every scalar is finite (math.isfinite(x) == True)
         +--> Stream into DataLoader and PyTorch Tensor Core
```

### 4.2 Runtime Function Call Stack Trace

```text
[DataSanitizationService.fit_profile / transform_samples]
 │
 ├── 1. Service Layer Entry (With Distributed W3C Tracing)
 │    │  File: src/features/data_sanitization/service/data_sanitization_service.py
 │    │  Function: DataSanitizationService.fit_profile(ctx, train_features, config)
 │    │  Tracing: Tracer.start_span("data_sanitization.fit", ctx) as child_ctx
 │    │  Error Boundary: Structured try / catch(ApplicationError) / finally(span.end)
 │    │
 │    └──> 2. Domain Workflow Layer
 │          │  File: src/features/data_sanitization/workflows/data_sanitization_workflow.py
 │          │  Function: DataSanitizationWorkflow.execute_fit_flow(ctx, train_features, config)
 │          │
 │          ├──> 3. Pure Mathematical Distribution Profiling
 │          │     │  File: src/features/data_sanitization/rules/data_sanitization_rules.py
 │          │     │  Function: DataSanitizationRules.compute_feature_statistics(raw_samples)
 │          │     │  Invariants Checked:
 │          │     │    - Skips non-finite values (NaN / Inf) during mean/std calculation
 │          │     │    - Computes sample variance: s^2 = sum((x - mu)^2) / (N - 1)
 │          │     │    - Detects dead features: is_constant = (std < 1e-8 or max - min < 1e-8)
 │          │     │  Returns: tuple[FeatureStatisticsDTO, ...]
 │          │     │
 │          ├──> 4. State Machine Transition
 │          │     │  File: src/features/data_sanitization/machines/data_sanitization_machine.py
 │          │     │  Function: DataSanitizationMachine.transition(STATUS_RAW, STATUS_FITTED)
 │          │     │  Returns: "FITTED"
 │          │     │
 │          └──> 5. Repository Profile Persistence
 │                │  File: src/features/data_sanitization/repository/data_sanitization_repository.py
 │                │  Port Call: DataSanitizationRepositoryPort.save_profile(ctx, profile)
 │                │  Returns: SanitizationProfileModel (status="FITTED")
 │
 ├── 6. Service Transformation Phase
 │    │  File: src/features/data_sanitization/service/data_sanitization_service.py
 │    │  Function: DataSanitizationService.transform_samples(ctx, profile_id, samples)
 │    │  Tracing: Tracer.start_span("data_sanitization.transform", ctx) as child_ctx
 │    │
 │    └──> 7. Domain Workflow Transformation Execution
 │          │  File: src/features/data_sanitization/workflows/data_sanitization_workflow.py
 │          │  Function: DataSanitizationWorkflow.execute_transform_flow(ctx, profile_id, samples)
 │          │
 │          ├──> 8. Profile Retrieval
 │          │     │  Port Call: DataSanitizationRepositoryPort.find_profile_by_id(ctx, profile_id)
 │          │     │  Returns: SanitizationProfileModel with locked statistics
 │          │     │
 │          └──> 9. Pure Sample Sanitization
 │                │  File: src/features/data_sanitization/rules/data_sanitization_rules.py
 │                │  Function: DataSanitizationRules.sanitize_sample(s_id, feats, label, stats, config)
 │                │  Execution:
 │                │    - Impute NaN -> mu[col]
 │                │    - Impute Inf -> copysign(impute_inf_value, val)
 │                │    - Winsorize -> clamp to [mu - k*std, mu + k*std]
 │                │    - Scale -> (val - mu) / std
 │                │    - Mask dead features -> 0.0
 │                │  Returns: CleanedSampleDTO
 │
 └── 10. Cleaned Tensors Emitted to Downstream PyTorch DataLoader
```

---

## 5. Critical Mathematical Formulations

### 5.1 Robust Mean and Sample Variance Calculation
For each telemetry metric column $j \in \{1, \dots, D\}$ with valid finite observations $S_j = \{x_{i,j} \mid x_{i,j} \in \mathbb{R}_{\text{finite}}\}$:
$$\mu_j = \frac{1}{|S_j|} \sum_{x \in S_j} x$$
$$\sigma_j = \sqrt{\frac{1}{|S_j| - 1} \sum_{x \in S_j} (x - \mu_j)^2}$$

If $|S_j| < 2$ or $\sigma_j < 10^{-8}$, the feature is marked constant: $\text{is\_constant}_j = \text{True}, \sigma_j = 1.0$.

### 5.2 Winsorization Boundary Formulation
To neutralize extreme metric spikes without dropping records:
$$x_{\text{clamped}} = \max\left(\mu_j - k \cdot \sigma_j, \min\left(x, \mu_j + k \cdot \sigma_j\right)\right)$$
Where default parameter $k = 5.0$ covers $99.9999\%$ of values under normal conditions, intercepting only pathological infinite outliers.

### 5.3 Z-Score Standardization
Standardizes heterogeneous telemetry dimensions into unit scale:
$$z_{i,j} = \begin{cases} 0.0 & \text{if } \text{is\_constant}_j \\ \frac{x_{i,j} - \mu_j}{\sigma_j} & \text{otherwise} \end{cases}$$

---

## 6. Consequences and Trade-Offs

### Positive
- **Guaranteed Finite Gradients**: Completely eliminates `NaN` gradient crashes during PyTorch backpropagation.
- **Leak-Free Transformation**: Normalization parameters are locked to training distributions, ensuring valid empirical testing.
- **Extreme Spike Resilience**: Outlier winsorization prevents multi-second cold-start network latency from skewing gradient steps.
- **Architectural Decoupling**: Cleaning rules can be adjusted, audited, and tested independently of model architecture.

### Negative and Mitigations
- **Computation Overhead**: Feature statistics must be computed across training sets before transformations start.
  - *Mitigation*: The pure algorithm operates in single-pass linear time $O(N \cdot D)$ on CPU with $O(D)$ memory.
