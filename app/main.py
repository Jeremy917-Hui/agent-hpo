"""HTTP entry point. Domain services must not depend on FastAPI requests."""

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.config import SOURCE_ROOT, Settings
from app.dataset_api import router as dataset_router
from app.datasets import DatasetError, DatasetStore
from app.model_api import router as model_router
from app.models.base import ModelError
from app.run_api import router as run_router
from app.runs import RunStore


class JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "time": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname, "message": record.getMessage(),
        }, ensure_ascii=False)


def create_app(settings: Settings | None = None):
    settings = settings or Settings.load()
    logger = logging.getLogger(f"agent_hpo.{settings.log_dir}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    @asynccontextmanager
    async def lifespan(application):
        settings.prepare()
        application.state.datasets = DatasetStore(settings)
        application.state.datasets.initialize()
        application.state.runs = RunStore(settings)
        application.state.runs.initialize()
        handler = logging.FileHandler(settings.log_dir / "backend.jsonl", encoding="utf-8")
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.info("backend_started milestone=3 mode=local")
        try:
            yield
        finally:
            logger.info("backend_stopped")
            logger.removeHandler(handler)
            handler.close()

    app = FastAPI(title="Machine Learning HPO Agent", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.logger = logger
    app.include_router(dataset_router)
    app.include_router(model_router)
    app.include_router(run_router)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])
    templates = Jinja2Templates(directory=str(SOURCE_ROOT / "app" / "templates"))
    app.mount("/static", StaticFiles(directory=SOURCE_ROOT / "app" / "static"), name="static")

    @app.exception_handler(DatasetError)
    async def dataset_error(request, error):
        return JSONResponse({"detail": str(error)}, status_code=error.status_code)

    @app.exception_handler(ModelError)
    async def model_error(request, error):
        return JSONResponse({"detail": str(error), "error": {"code": error.code, "message": str(error), "details": error.details}}, status_code=error.status_code)

    @app.exception_handler(OSError)
    async def storage_error(request, error):
        logger.error("storage_error type=%s", type(error).__name__)
        return JSONResponse({"detail": "本地存储不可用或剩余空间不足，请检查磁盘和日志。"}, status_code=507)

    @app.middleware("http")
    async def local_request(request: Request, call_next):
        if request.method in {"POST", "PATCH", "PUT", "DELETE"}:
            origin = request.headers.get("origin")
            if origin and origin != f"{request.url.scheme}://{request.headers.get('host')}":
                return JSONResponse({"detail": "Cross-origin writes are not allowed"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = "default-src 'self'; style-src 'self'; script-src 'self'; frame-ancestors 'none'"
        logger.info("http method=%s path=%s status=%s", request.method, request.url.path, response.status_code)
        return response

    @app.get("/health")
    def health():
        try:
            free = settings.check_disk()
        except OSError:
            return JSONResponse({"status": "degraded", "reason": "disk_space"}, status_code=503)
        return {"status": "ok", "mode": "local", "milestone": 3, "free_disk_gib": free}

    @app.get("/")
    def dashboard(request: Request):
        return templates.TemplateResponse(request=request, name="dashboard.html", context={"max_upload_mb": settings.max_upload_mb, "max_dataset_rows": settings.max_dataset_rows})

    return app


app = create_app()
