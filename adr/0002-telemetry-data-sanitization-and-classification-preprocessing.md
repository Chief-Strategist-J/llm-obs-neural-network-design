# 0002: Telemetry Data Sanitization, Robust Imputation, and Classification Feature Preprocessing : An Architecture Decision Record (ADR)

- **Status**: Accepted
- **Date**: 2026-10-15
- **Authors**: Platform Architecture and Core Neural Systems Team
- **RFC / Standards**: IEEE 754 (Floating-Point Arithmetic), OpenTelemetry W3C TraceContext RFC, ISO/IEC 25010 (System and Software Quality Models)
- **Deciders**: Lead Systems Architect, Distributed Systems Lead, Machine Learning Infrastructure Engineer

---

## 1. Context and Problem Statement

In distributed systems telemetry, raw metrics and trace observations emitted across thousands of microservices, serverless tasks, and LLM gateways suffer from severe data quality pathologies:
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
    RAW["Raw Partitioned Telemetry (Train and Val)"] --> DISPATCHER["Data Sanitization Service"]
    
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
    DISPATCHER -->|3. Transform Train and Val Splits| TRANSFORM_PHASE
    TRANSFORM_PHASE --> CLEANED["Cleaned Finite Tensors (CleanedSampleDTO)"]
    CLEANED --> TENSOR_ENGINE["PyTorch Neural Tensor Core"]
```

---

## 3.3 Low-Level Architecture (LLD) and Subsystem Topology

The Low-Level Architecture isolates ports, adapters, pure rules, and domain workflows into strict hexagonal boundaries:

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': { 'primaryColor': '#1E293B', 'primaryBorderColor': '#38BDF8', 'primaryTextColor': '#F1F5F9', 'lineColor': '#38BDF8', 'secondaryColor': '#064E3B', 'tertiaryColor': '#78350F', 'mainBkg': '#0F172A', 'nodeBorder': '#38BDF8'}}}%%
classDiagram
    direction TB

    class DataSanitizationRepositoryPort {
        <<interface>>
        +save_profile(ctx: TraceContext, profile: SanitizationProfileModel)~SanitizationProfileModel~
        +find_profile_by_id(ctx: TraceContext, profile_id: str)~SanitizationProfileModel~
    }

    class InMemoryDataSanitizationRepository {
        -dict _profiles
        +save_profile(ctx: TraceContext, profile: SanitizationProfileModel)~SanitizationProfileModel~
        +find_profile_by_id(ctx: TraceContext, profile_id: str)~SanitizationProfileModel~
    }

    class DataSanitizationRules {
        <<pure-function>>
        +compute_feature_statistics(raw_samples: Sequence)~tuple[FeatureStatisticsDTO]~
        +sanitize_sample(sample_id: str, features: Sequence, label: int, stats: Sequence, config: SanitizationConfigDTO)~CleanedSampleDTO~
    }

    class DataSanitizationMachine {
        <<state-machine>>
        +_VALID_TRANSITIONS: dict
        +transition(current_state: str, next_state: str)~str~
    }

    class DataSanitizationWorkflow {
        -DataSanitizationRepositoryPort _repo
        +execute_fit_flow(ctx: TraceContext, train_feature_matrix: Sequence, config: SanitizationConfigDTO)~SanitizationProfileModel~
        +execute_transform_flow(ctx: TraceContext, profile_id: str, samples: Sequence)~tuple[CleanedSampleDTO]~
    }

    class DataSanitizationService {
        -DataSanitizationRepositoryPort _repository
        -DataSanitizationWorkflow _workflow
        +fit_profile(ctx: TraceContext, train_features: Sequence, config: SanitizationConfigDTO)~Result~
        +transform_samples(ctx: TraceContext, profile_id: str, samples: Sequence)~Result~
    }

    DataSanitizationRepositoryPort <|.. InMemoryDataSanitizationRepository : implements
    DataSanitizationWorkflow --> DataSanitizationRepositoryPort : calls
    DataSanitizationWorkflow --> DataSanitizationRules : invokes pure
    DataSanitizationWorkflow --> DataSanitizationMachine : transitions
    DataSanitizationService --> DataSanitizationWorkflow : orchestrates
```

---

## 3.4 Ingress Execution Flow and End-to-End Sequence

```mermaid
sequenceDiagram
    autonumber
    actor Pipeline as "System Telemetry Pipeline"
    participant Service as "DataSanitizationService"
    participant OTEL as "OpenTelemetry Collector"
    participant Workflow as "DataSanitizationWorkflow"
    participant Rules as "DataSanitizationRules (Pure Logic)"
    participant Machine as "DataSanitizationMachine"
    participant Repo as "SanitizationRepository (Port/Adapter)"
    actor TensorCore as "PyTorch Tensor Core"

    Note over Pipeline,TensorCore: Phase 1: Fit Sanitization Profile Strictly on Training Partition
    Pipeline->>Service: fit_profile(ctx, train_features, config)
    Service->>OTEL: Start Child Span (data_sanitization.fit)
    Service->>Workflow: execute_fit_flow(ctx, train_features, config)
    Workflow->>Rules: compute_feature_statistics(train_features)
    Note over Rules: Compute mean and sample variance per column. Filter NaN/Inf. Detect zero-variance features.
    Rules-->>Workflow: tuple[FeatureStatisticsDTO, ...]
    Workflow->>Machine: transition(RAW, FITTED)
    Machine-->>Workflow: STATUS_FITTED
    Workflow->>Repo: save_profile(ctx, profile)
    Repo-->>Workflow: SanitizationProfileModel (status="FITTED")
    Workflow-->>Service: SanitizationProfileModel
    Service->>OTEL: End Span (StatusCode.OK)
    Service-->>Pipeline: Result.ok(SanitizationProfileModel)

    Note over Pipeline,TensorCore: Phase 2: Transform Training and Validation Telemetry Uniformly
    Pipeline->>Service: transform_samples(ctx, profile_id, samples)
    Service->>OTEL: Start Child Span (data_sanitization.transform)
    Service->>Workflow: execute_transform_flow(ctx, profile_id, samples)
    Workflow->>Repo: find_profile_by_id(ctx, profile_id)
    Repo-->>Workflow: SanitizationProfileModel (locked mu, sigma)
    loop For each telemetry sample
        Workflow->>Rules: sanitize_sample(s_id, feats, label, stats, config)
        Note over Rules: 1. Impute NaN -> mu<br/>2. Impute Inf -> boundary<br/>3. Winsorize outlier (k * sigma)<br/>4. Scale Z-Score ((x - mu) / sigma)<br/>5. Mask dead features -> 0.0
        Rules-->>Workflow: CleanedSampleDTO
    end
    Workflow-->>Service: tuple[CleanedSampleDTO, ...]
    Service->>OTEL: End Span (StatusCode.OK)
    Service-->>Pipeline: Result.ok(tuple[CleanedSampleDTO, ...])
    Pipeline->>TensorCore: Feed Cleaned Finite Tensors (No NaNs, No Infs)
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
  |      +--> [ Sub-Gate 3.3: Normalization and Zero-Variance Masking ]
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

### 4.2 Runtime Function Call Stack Matrix

| Level | Subsystem / Layer | Function / Method Signature | Exact Execution Responsibility |
| :--- | :--- | :--- | :--- |
| **1. Service Entry** | Domain Service | `DataSanitizationService.fit_profile(ctx, train_features, config)` | Starts OpenTelemetry child span (`data_sanitization.fit`), wraps workflow in structured try/catch/finally. |
| **2. Workflow Orchestration** | Domain Workflow | `DataSanitizationWorkflow.execute_fit_flow(ctx, train_feature_matrix, config)` | Computes feature statistics on training split, builds profile model, transitions state machine, and persists profile. |
| **3. Pure Statistical Engine** | Pure Domain Rules | `DataSanitizationRules.compute_feature_statistics(raw_samples)` | Iterates across sample columns, skips NaN/Inf, calculates mean and sample variance ($s^2$), detects constant features. |
| **4. State Machine** | Pure State Machine | `DataSanitizationMachine.transition(STATUS_RAW, STATUS_FITTED)` | Enforces state progression, validating that transition from `RAW` to `FITTED` is permitted. |
| **5. Repository Persistence** | Driven Port Contract | `DataSanitizationRepositoryPort.save_profile(ctx, profile)` | Abstract contract boundary saving `SanitizationProfileModel` to storage adapter. |
| **6. Service Transform** | Domain Service | `DataSanitizationService.transform_samples(ctx, profile_id, samples)` | Starts OpenTelemetry child span (`data_sanitization.transform`), coordinates sample-level cleaning. |
| **7. Workflow Transformation** | Domain Workflow | `DataSanitizationWorkflow.execute_transform_flow(ctx, profile_id, samples)` | Retrieves locked profile from repository, iterates samples, invokes pure sanitization rules. |
| **8. Pure Sample Cleaner** | Pure Domain Rules | `DataSanitizationRules.sanitize_sample(sample_id, features, label, statistics, config)` | Performs NaN imputation, Inf boundary clamping, outlier winsorization, Z-score scaling, and dead-feature masking. |
| **9. Model Assembly** | Types & DTOs | `CleanedSampleDTO(sample_id, tuple(cleaned_features), target_label, imputed_count)` | Assembles immutable clean sample with guaranteed finite floating-point properties. |

#### Detailed Runtime Function Call Stack Trace

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

## 5. Critical Mathematical Algorithms Implemented in the Engine

### 5.1 Robust Mean and Sample Variance Algorithm

#### 1. Non-Technical Definition
Imagine taking the temperature of a datacenter every minute. Occasionally, a sensor loses Wi-Fi and records "Error" or "999,999 degrees". Before calculating the datacenter's true average temperature, you must ignore the sensor errors and only average the numbers that make sense. This algorithm filters out corrupted measurements and calculates the true center and typical variation of the data.

#### 2. Technical Definition
Given raw observation matrix $\mathbf{X} \in \mathbb{R}^{N \times D}$, let $S_j = \{x_{i,j} \mid x_{i,j} \in \mathbb{R}_{\text{finite}}\}$ denote the set of finite scalar observations for feature column $j \in \{1, \dots, D\}$.
The robust mean $\mu_j$ and unbiased sample variance $s_j^2$ are formulated as:
$$\mu_j = \frac{1}{|S_j|} \sum_{x \in S_j} x$$
$$s_j^2 = \frac{1}{|S_j| - 1} \sum_{x \in S_j} (x - \mu_j)^2, \quad \sigma_j = \sqrt{s_j^2}$$

If $|S_j| < 2$ or $\sigma_j < 10^{-8}$, the column is marked dead: $\text{is\_constant}_j = \text{True}, \sigma_j = 1.0$.

#### 3. Algorithm Flow Diagram

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': { 'primaryColor': '#1F2937', 'primaryBorderColor': '#60A5FA', 'primaryTextColor': '#F3F4F6', 'lineColor': '#60A5FA', 'secondaryColor': '#064E3B', 'tertiaryColor': '#78350F', 'mainBkg': '#111827', 'nodeBorder': '#60A5FA'}}}%%
flowchart LR
    INPUT["Raw Column Vector x_j"] --> FILTER{"Filter isfinite(x)?"}
    FILTER -- "NaN / Inf" --> DROP["Ignore from Mean Calc"]
    FILTER -- "Finite" --> ACCUM["Accumulate sum(x) & sum(x^2)"]
    ACCUM --> MEAN["Compute mu_j = sum / N"]
    MEAN --> VAR["Compute s_j^2 = sum((x - mu)^2) / (N - 1)"]
    VAR --> ZERO_CHK{"sigma_j < 1e-8?"}
    ZERO_CHK -- "Yes" --> CONST["Set is_constant = True, sigma = 1.0"]
    ZERO_CHK -- "No" --> ACTIVE["Set is_constant = False, sigma = sqrt(s^2)"]
```

#### 4. Language-Agnostic Pseudocode

```text
ALGORITHM ComputeFeatureStatistics(matrix: Array of Array of Float) -> Array of Statistics:
    dim := length(matrix[0])
    stats := empty Array of Statistics
    
    FOR j FROM 0 TO dim - 1:
        finite_vals := empty List of Float
        FOR i FROM 0 TO length(matrix) - 1:
            val := matrix[i][j]
            IF is_finite(val) THEN:
                finite_vals.append(val)
            END IF
        END FOR
        
        IF length(finite_vals) < 2 THEN:
            stats.append(Statistics(mean=0.0, std=1.0, is_constant=TRUE))
            CONTINUE
        END IF
        
        mean := sum(finite_vals) / length(finite_vals)
        variance := sum((x - mean)^2 FOR x IN finite_vals) / (length(finite_vals) - 1)
        std := sqrt(variance)
        is_const := (std < 1e-8)
        
        stats.append(Statistics(mean=mean, std=(is_const ? 1.0 : std), is_constant=is_const))
    END FOR
    
    RETURN stats
END ALGORITHM
```

---

### 5.2 Outlier Winsorization and Clamping Algorithm

#### 1. Non-Technical Definition
If everyone in a room makes $\$50,000$ a year and a multi-billionaire walks in, the mathematical "average salary" jumps to $\$50,000,000$. A neural network trying to learn from this will spend all its time obsessed with the billionaire and forget how ordinary people earn. Winsorization replaces extreme extremes with the highest reasonable normal value so the network stays focused on normal patterns.

#### 2. Technical Definition
Winsorization replaces extreme tail values beyond $k$ standard deviations ($\pm k \cdot \sigma_j$) with the respective boundary thresholds, preserving directional magnitude while clamping unbounded variance:
$$x_{\text{clamped}} = \begin{cases} 
\mu_j + k \cdot \sigma_j & \text{if } x > \mu_j + k \cdot \sigma_j \\ 
\mu_j - k \cdot \sigma_j & \text{if } x < \mu_j - k \cdot \sigma_j \\ 
x & \text{otherwise} 
\end{cases}$$
Where $k = 5.0$ guarantees that less than $1$ in $1,000,000$ valid Gaussian telemetry spans is altered, intercepting strictly pathological extremes.

#### 3. Algorithm Flow Diagram

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': { 'primaryColor': '#1F2937', 'primaryBorderColor': '#60A5FA', 'primaryTextColor': '#F3F4F6', 'lineColor': '#60A5FA', 'secondaryColor': '#064E3B', 'tertiaryColor': '#78350F', 'mainBkg': '#111827', 'nodeBorder': '#60A5FA'}}}%%
flowchart TD
    VAL["Scalar Value v"] --> CHECK_CONST{"is_constant?"}
    CHECK_CONST -- "Yes" --> PASS["Return v"]
    CHECK_CONST -- "No" --> BOUNDS["upper = mu + k*std\nlower = mu - k*std"]
    BOUNDS --> CHECK_UPPER{"v > upper?"}
    CHECK_UPPER -- "Yes" --> CLAMP_UP["v = upper"]
    CHECK_UPPER -- "No" --> CHECK_LOWER{"v < lower?"}
    CHECK_LOWER -- "Yes" --> CLAMP_DOWN["v = lower"]
    CHECK_LOWER -- "No" --> EMIT["Return v unchanged"]
    CLAMP_UP --> EMIT
    CLAMP_DOWN --> EMIT
```

#### 4. Language-Agnostic Pseudocode

```text
ALGORITHM WinsorizeOutlier(v: Float, mu: Float, sigma: Float, k: Float, is_constant: Boolean) -> Float:
    IF is_constant THEN:
        RETURN v
    END IF
    
    upper_bound := mu + (k * sigma)
    lower_bound := mu - (k * sigma)
    
    IF v > upper_bound THEN:
        RETURN upper_bound
    ELSE IF v < lower_bound THEN:
        RETURN lower_bound
    ELSE:
        RETURN v
    END IF
END ALGORITHM
```

---

### 5.3 Z-Score Standardization with Dead-Feature Masking

#### 1. Non-Technical Definition
If one feature is measured in nanoseconds ($10^9$) and another is measured in percentage ($0.5$), the neural network will pay billions of times more attention to the nanoseconds simply because the numbers are bigger. Standardization translates all measurements onto the same universal ruler: zero means average, positive means above average, and negative means below average.

#### 2. Technical Definition
Standardizes non-constant feature coordinates to zero empirical mean and unit variance. Features with zero variance are safely mapped to zero to eliminate numerical singularity:
$$z_{i,j} = \begin{cases} 
0.0 & \text{if } \text{is\_constant}_j \\ 
\frac{x_{i,j} - \mu_j}{\sigma_j} & \text{otherwise} 
\end{cases}$$

#### 3. Algorithm Flow Diagram

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': { 'primaryColor': '#1F2937', 'primaryBorderColor': '#60A5FA', 'primaryTextColor': '#F3F4F6', 'lineColor': '#60A5FA', 'secondaryColor': '#064E3B', 'tertiaryColor': '#78350F', 'mainBkg': '#111827', 'nodeBorder': '#60A5FA'}}}%%
flowchart LR
    IN_V["Clamped Value v"] --> CHK_DEAD{"is_constant?"}
    CHK_DEAD -- "Yes" --> ZERO["Output 0.0\n(Dead Feature Neutralized)"]
    CHK_DEAD -- "No" --> SUB["Subtract Mean (v - mu)"]
    SUB --> DIV["Divide by Standard Deviation / sigma"]
    DIV --> OUT_Z["Output Standardized z (Mean=0, Var=1)"]
```

#### 4. Language-Agnostic Pseudocode

```text
ALGORITHM StandardizeZScore(v: Float, mu: Float, sigma: Float, is_constant: Boolean) -> Float:
    IF is_constant THEN:
        RETURN 0.0
    END IF
    
    RETURN (v - mu) / sigma
END ALGORITHM
```

---

## 6. Database Tables and Migration Specifications

```sql
CREATE TABLE IF NOT EXISTS sanitization_profiles (
    profile_id VARCHAR(64) PRIMARY KEY,
    feature_count INTEGER NOT NULL,
    statistics_json JSONB NOT NULL,
    config_json JSONB NOT NULL,
    status VARCHAR(32) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_sanitization_profiles_status ON sanitization_profiles (status);
```

---

## 7. Consequences and Trade-Offs

### Positive
- **Guaranteed Finite Gradients**: Completely eliminates `NaN` gradient crashes during PyTorch backpropagation.
- **Leak-Free Transformation**: Normalization parameters are locked to training distributions, ensuring valid empirical testing.
- **Extreme Spike Resilience**: Outlier winsorization prevents multi-second cold-start network latency from skewing gradient steps.
- **Architectural Decoupling**: Cleaning rules can be adjusted, audited, and tested independently of model architecture.

### Negative and Mitigations
- **Computation Overhead**: Feature statistics must be computed across training sets before transformations start.
  - *Mitigation*: The pure algorithm operates in single-pass linear time $O(N \cdot D)$ on CPU with $O(D)$ memory.
