"""FastAPI application entry point."""

from fastapi import FastAPI

from authorai import __version__

app = FastAPI(title="Author AI", version=__version__)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}
