# ADR 0002: PyTorch Tensor Foundations, Layer Normalization, and Tiny-Batch Overfit Validation

## Status
Accepted

## Context
Following ADR-0001 (which established leak-free data splitting and trivial baseline benchmark gates), neural network models must be built on PyTorch and tensor primitives with mathematical stability.

A frequent pathology in deep learning projects is premature architectural complexity: configuring thousands of parameters and deep backbones without verifying optimization stability and gradient propagation. 

Key principles established:
1. **Optimization Stability First**: Stable initialization (He / Kaiming / Xavier), layer normalization (`LayerNorm` / `RMSNorm`), and residual connections are preconditions for effective gradient flow.
2. **Tiny-Batch Overfitting Gate (10–100 samples)**: Before any distributed training or hyperparameter search, a model must achieve near-zero loss on a small batch. Failure to overfit a small batch proves a structural bug (e.g. broken autograd graph, label mismatch, dead activations), not a tuning challenge.
3. **Hexagonal Encapsulation**: PyTorch execution, loss calculation (CrossEntropy, MSE, Focal), and autograd updates must reside behind clean port/adapter interfaces and workflow DAGs, keeping the domain stateless.

## Decision
We create the `neural_learning_core` feature in `neural-network-design`:
1. **PyTorch Tensor Core Architecture (`TelemetryResidualClassifier`)**:
   - Modular residual MLP architecture with PyTorch `nn.Module`.
   - Incorporates `LayerNorm`, `ReLU`/`GELU`, and residual skip connections.
   - Initialized via Kaiming uniform distributions.
2. **Tiny-Batch Overfitting Workflow (`TinyBatchOverfitWorkflow`)**:
   - Accepts a tiny batch of 10–100 samples.
   - Runs gradient descent (AdamW with gradient clipping) for 30–50 iterations.
   - Verifies empirical training loss drops below strict threshold ($< 0.05$).
3. **State Machine (`NeuralTrainingMachine`)**:
   - Enforces state progression: `INITIALIZED` -> `STABILITY_VERIFIED` -> `TINY_BATCH_OVERFIT_PASSED` -> `READY_FOR_SCALING`.

## Consequences
- PyTorch and tensor operations are cleanly isolated behind hexagonal ports.
- Prevents wasted compute on un-trainable network configurations.
- All code strictly complies with zero-comment, single responsibility, and immutable type standards.
