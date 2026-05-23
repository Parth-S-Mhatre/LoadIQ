from __future__ import annotations

import importlib
from threading import Lock
from typing import Any

from fastapi import FastAPI


class LazyFastAPIApp:
    """Import a model app only when its mounted route receives traffic."""

    def __init__(self, module_name: str, app_name: str = "app") -> None:
        self.module_name = module_name
        self.app_name = app_name
        self._app: Any | None = None
        self._lock = Lock()

    @property
    def loaded(self) -> bool:
        return self._app is not None

    def _load(self) -> Any:
        if self._app is None:
            with self._lock:
                if self._app is None:
                    module = importlib.import_module(self.module_name)
                    self._app = getattr(module, self.app_name)
        return self._app

    async def __call__(self, scope, receive, send) -> None:
        app = self._load()
        await app(scope, receive, send)


model1_app = LazyFastAPIApp("Model1")
model2_app = LazyFastAPIApp("Model2")

app = FastAPI(
    title="LoadIQ Combined Backend",
    description="Single Render service that lazily serves Model1 and Model2 APIs.",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "message": "LoadIQ backend is running.",
        "model1": {
            "base_path": "/model1",
            "predict": "POST /model1/predict",
            "features": "GET /model1/features",
            "health": "GET /model1/health",
        },
        "model2": {
            "base_path": "/model2",
            "predict": "POST /model2/predict",
            "predict_batch": "POST /model2/predict_batch",
            "features": "GET /model2/features",
            "health": "GET /model2/health",
        },
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "mode": "combined_lazy_backend",
        "models": {
            "model1_loaded": model1_app.loaded,
            "model2_loaded": model2_app.loaded,
        },
    }


@app.get("/api/health_check")
def health_check():
    return {
        "status": "running",
        "mode": "combined_lazy_backend",
        "model1_loaded": model1_app.loaded,
        "model2_loaded": model2_app.loaded,
    }


app.mount("/model1", model1_app)
app.mount("/model2", model2_app)
