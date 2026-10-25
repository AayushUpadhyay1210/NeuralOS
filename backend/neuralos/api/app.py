"""FastAPI application. Run with ``uvicorn neuralos.api.app:app``."""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .. import __version__
from ..rl.loader import ModelUnavailable
from ..sim.engine import SimulationError
from .routes import router

log = logging.getLogger("neuralos.api")


def create_app() -> FastAPI:
    app = FastAPI(
        title="NeuralOS API",
        version=__version__,
        description="CPU scheduling simulator with a reinforcement-learning scheduler. "
        "This is a simulator, not an operating-system kernel.",
    )
    origins = os.environ.get(
        "NEURALOS_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in origins if o.strip()],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.exception_handler(ModelUnavailable)
    async def _model_unavailable(_: Request, exc: ModelUnavailable) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(SimulationError)
    async def _sim_error(_: Request, exc: SimulationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": f"simulation failed: {exc}"})

    @app.exception_handler(ValueError)
    async def _value_error(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error")
        return JSONResponse(status_code=500,
                            content={"detail": f"internal error: {type(exc).__name__}: {exc}"})

    app.include_router(router)
    return app


app = create_app()
