import logging

from fastapi import FastAPI

from app.config import settings
from app.routers import business_webhook, personal_webhook

logging.basicConfig(level=getattr(logging, settings.log_level, logging.INFO))

app = FastAPI(title="360dialog Echo Bot")


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