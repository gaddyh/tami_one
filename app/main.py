import asyncio
import logging

from fastapi import FastAPI

from app.config import settings
from app.commitments.commitments_agent import configure_dspy
from app.commitments.processor import drain_and_process
from app.db import init_db, load_cache
from app.routers import business_webhook, personal_webhook

logging.basicConfig(level=getattr(logging, settings.log_level, logging.INFO))

logger = logging.getLogger(__name__)

app = FastAPI(title="360dialog Echo Bot")

DRAIN_INTERVAL_SECONDS = 30 * 60
_drain_task: asyncio.Task | None = None


async def _drain_loop() -> None:
    while True:
        logger.info("Starting drain cycle")
        await asyncio.sleep(DRAIN_INTERVAL_SECONDS)
        try:
            logger.info("Processing commitments")
            results = await drain_and_process()
            total = sum(len(v) for v in results.values())
            logger.info("Drain cycle complete: %d commitment(s)", total)
        except Exception:
            logger.exception("Error in drain cycle")


@app.on_event("startup")
async def _on_startup() -> None:
    logger.info("Configuring DSPy with model=%s", settings.openai_model)
    configure_dspy(settings)
    init_db()
    load_cache()
    global _drain_task
    _drain_task = asyncio.create_task(_drain_loop())


@app.on_event("shutdown")
async def _on_shutdown() -> None:
    if _drain_task:
        _drain_task.cancel()
        try:
            await _drain_task
        except asyncio.CancelledError:
            pass


@app.get("/")
async def root() -> dict[str, str]:
    return {"status": "ok", "service": "360dialog-echo-bot"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(business_webhook.router)
app.include_router(personal_webhook.router)


def serve() -> None:
    """Local development only. Production uses: uvicorn app.main:app --host 0.0.0.0 --port $PORT"""
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)