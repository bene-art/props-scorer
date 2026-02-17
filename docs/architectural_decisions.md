# Architectural Decisions

Brief rationale for key design choices.

## Why FastAPI?

Minimal surface area. Strong typing via Pydantic. Native async support. Auto-generated OpenAPI documentation without additional tooling.

## Why Structured JSON Logging?

JSON logs are machine-parseable by default. Log aggregation systems (ELK, Datadog, CloudWatch) ingest them directly. Request correlation becomes trivial with structured fields.

## Why Request ID Propagation?

Distributed tracing requires correlation. Generating a request ID at ingress and propagating it through logs and response headers enables end-to-end observability without external tracing infrastructure.

## Why Explicit Model Versioning?

Model version appears in every prediction response. This makes debugging straightforward: given a prediction, you know exactly which model produced it. Enables safe rollback and A/B comparison.

## Why Pydantic Schemas?

Runtime validation at API boundaries. Type hints propagate to documentation. Invalid requests fail fast with clear error messages.

## Why Small Surface Area?

Three endpoints. One model. No feature flags. Complexity is the enemy of reliability. This service does one thing and makes that thing observable.
