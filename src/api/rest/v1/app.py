from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.api.rest.v1.routers.network_spec_router import router as network_spec_router
from src.api.rest.v1.routers.telemetry_pipeline_router import router as telemetry_pipeline_router
from src.infra.observability.tracer import Tracer


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Neural Network Design Service",
        description="Hexagonal service for Neural Network Design, telemetry data pipelines, leak-free evaluation, and PyTorch tensor verification.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health_check():
        return {"status": "ok"}

    from src.shared.middleware.error_envelope_middleware import ErrorEnvelopeMiddleware
    app.add_middleware(ErrorEnvelopeMiddleware)

    app.include_router(network_spec_router)
    app.include_router(telemetry_pipeline_router)
    return app


app = create_app()
