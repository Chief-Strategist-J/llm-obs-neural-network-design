# Neural Network Design Service

Neural Network Design architecture specification, topology validation, and parameter estimation engine for LLM Observability.

## Scaffolding & Directory Architecture
Strict adherence to `0-package-folder-writing-rule.md`, `1-feature-folder-writing-rule.md`, and `3.shared.code.writing.rule.md`:
- `contracts/`: OpenAPI & event contracts.
- `config/`: Declarative configuration schemas and runtime profiles.
- `database/`: Schema migrations and lockfiles.
- `messaging/`: Streaming topic definitions and DLQ policies.
- `deploy/`: K8s and Docker deployment manifests.
- `src/shared/`: Pure deterministic utilities, application errors, envelope models, and constants.
- `src/infra/`: Configuration loaders and OpenTelemetry tracer.
- `src/features/network_spec/`: Hexagonal domain feature implementation.
- `src/api/`: REST handlers and routers.
- `scripts/`: Operational scripts.
- `tests/`: Automated unit and integration test suites.
