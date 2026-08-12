"""FastAPI application factory.

Startup is FAIL-CLOSED: no configured API key means the app refuses to start
(v1 silently served everything openly when its key env var was missing).
Startup also recovers interrupted work — RUNNING jobs left by a crash are
re-queued before the worker starts, so no job is ever stranded (v1 defect).
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from authorai import __version__
from authorai import db as dbmod
from authorai.api import router as api_router
from authorai.config import Settings
from authorai.jobs import Worker
from authorai.log import setup_logger

logger = setup_logger(__name__)


def create_app(settings: Settings | None = None, worker: Worker | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not settings.api_key:
            raise RuntimeError(
                "AUTHORAI_API_KEY is not set — refusing to start an unauthenticated server"
            )
        conn = dbmod.connect(settings.db_path, settings.embedding_dim)
        try:
            recovered = dbmod.requeue_running_jobs(conn)
            if recovered:
                logger.info("re-queued %d interrupted job(s): %s", len(recovered), recovered)
        finally:
            conn.close()
        active_worker = worker if worker is not None else Worker(settings)
        active_worker.start()
        try:
            yield
        finally:
            active_worker.stop()

    app = FastAPI(title="Author AI", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    app.include_router(api_router)
    return app


app = create_app()
