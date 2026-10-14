"""Centralized system and HTTP constants for neural-network-design package."""

HTTP_HEADER_TRACEPARENT: str = "traceparent"
HTTP_HEADER_REQUEST_ID: str = "x-request-id"
HTTP_HEADER_CORRELATION_ID: str = "x-correlation-id"

STATUS_PENDING: str = "PENDING"
STATUS_ACTIVE: str = "ACTIVE"
STATUS_ARCHIVED: str = "ARCHIVED"

CODE_VALIDATION_FAILED: str = "VALIDATION_FAILED"
CODE_NOT_FOUND: str = "RESOURCE_NOT_FOUND"
CODE_INTERNAL_ERROR: str = "INTERNAL_SERVER_ERROR"

TOPIC_SPEC_CREATED: str = "neural-network.spec-created.v1"
TOPIC_TELEMETRY_SPANS_RAW: str = "llm-obs.telemetry.spans.raw.v1"
TOPIC_TELEMETRY_DRIFT_ALERTS: str = "llm-obs.telemetry.drift-alerts.v1"
TOPIC_MODEL_RETRAIN_COMMANDS: str = "llm-obs.neural.retrain-commands.v1"

