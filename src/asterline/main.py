"""FastAPI application factory — stub for Milestone 1."""

from fastapi import FastAPI

from asterline.config import Settings


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = Settings()
    app = FastAPI(title=settings.app_name, debug=settings.debug)

    @app.get("/health")
    async def health():
        return {"status": "ok", "app": settings.app_name}

    return app


app = create_app()
