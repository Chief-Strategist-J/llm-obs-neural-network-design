# ADR 0005: High-Throughput Streaming Telemetry Worker and Dead-Letter Queue (DLQ) Architecture

## Status
Accepted

## Context
Production LLM observability generates thousands of spans per second across multi-hop microservices. Ingesting this volume via synchronous batch HTTP requests introduces artificial latency and risk of buffer starvation. 

To bridge continuous Kafka streams into the hexagonal neural telemetry pipeline, the system requires a dedicated streaming consumer daemon capable of:
1. Ingesting raw streaming spans partitioned by `trace_id` from Kafka (`telemetry.spans.raw`).
2. Buffering streaming records into configurable micro-batches gated by both batch size (e.g. 100 spans) and elapsed time (e.g. 1000 ms).
3. Extracting and propagating W3C distributed trace context (`traceparent`, `x-request-id`, `x-correlation-id`).
4. Routing corrupted, malformed, or unparseable payloads directly to a dedicated Dead-Letter Queue (DLQ) topic (`telemetry.spans.dlq`) without crashing or halting consumer offset progress.
5. Emitting deterministic worker observability metrics (`total_consumed`, `total_batched`, `total_dlq_routed`, `total_batches_dispatched`).

## Decision
1. **Hexagonal Feature Structure**:
   - Implemented `src/features/streaming_telemetry_worker/` with:
     - `rules/streaming_telemetry_worker_rules.py`: Pure deterministic functions for span parsing, micro-batch construction, flushing conditions, and DLQ payload generation.
     - `types/streaming_telemetry_worker_types.py`: Immutable DTOs (`TelemetrySpanMessageDTO`, `MicroBatchDTO`, `DeadLetterMessageDTO`, `WorkerMetricsDTO`).
     - `repository/streaming_telemetry_worker_port.py` and `streaming_telemetry_worker_repository.py`: Storage contracts and thread-safe implementations for micro-batches, DLQ records, and telemetry metrics.
     - `service/streaming_telemetry_worker_service.py`: Orchestrator managing consumer polling, buffer flushing, DLQ routing via KafkaProducerPort, and OpenTelemetry child span lifecycle.

2. **Zero Magic Strings and Centralized Configuration**:
   - Registered `TOPIC_TELEMETRY_SPANS_DLQ` in `src/shared/constants/system_constants.py`.
   - Declared `dlq_telemetry_topic`, `batch_size`, and `batch_timeout_ms` under `config/default.yaml` and `config/env.schema`.

3. **Fault Tolerance and Non-Blocking Ingestion**:
   - Malformed payloads do not throw uncaught exceptions; they are routed with original byte streams, failure reasons, and partition/offset markers to the DLQ topic.
   - Consumers commit offsets asynchronously to maintain wire-speed ingestion throughput.

## Consequences
- Continuous streaming ingestion is decoupled from deep PyTorch neural training and batch processing.
- Corrupted payloads are isolated into DLQ for post-mortem analysis without blocking streaming pipelines.
- Unit, integration, and end-to-end tests verify micro-batching, DLQ routing, and downstream baseline gating.
