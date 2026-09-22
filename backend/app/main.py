import os
import sys
import time

from sqlalchemy.exc import OperationalError

if sys.platform == "win32":
    # EasyOCR's first-run model download prints a progress bar using Unicode
    # block characters; Windows' default console codepage (cp1252) can't
    # encode them, which crashes the OCR call. Force UTF-8 on the standard
    # streams so this - and any other library that prints Unicode - works
    # regardless of the terminal's active codepage.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .database import Base, engine
from .routers import auth, scans, dashboard
from .services import ocr_service, storage_service, supabase_auth_service

# Uvicorn configures only its own loggers, so without this the app's own
# log records have no handler and anything below WARNING is dropped - which
# on a hosted platform means the startup diagnostics below never reach the
# log stream.
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _create_all_with_retry(attempts: int = 3, delay_seconds: float = 2.0) -> None:
    """
    Supabase's connection pooler occasionally drops a brand-new connection
    outright (a transient network blip, not a stale-connection problem
    pool_pre_ping solves) - this is the very first query the app makes, so
    there's no pooled connection to retry with. A couple of short retries
    turns an occasional dropped connection here into a slightly slower
    startup instead of a crashed process.
    """
    for attempt in range(1, attempts + 1):
        try:
            Base.metadata.create_all(bind=engine)
            return
        except OperationalError:
            if attempt == attempts:
                raise
            logger.warning(
                "Database connection attempt %s failed, retrying in %ss...", attempt, delay_seconds
            )
            time.sleep(delay_seconds)


_create_all_with_retry()


def _seed_default_admin() -> None:
    if not settings.auth_enabled:
        logger.warning(
            "SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY / SUPABASE_ANON_KEY "
            "are not fully configured - authentication will not work."
        )
        return
    if supabase_auth_service.find_user_by_email(settings.default_admin_email):
        return
    supabase_auth_service.create_user(
        settings.default_admin_email,
        settings.default_admin_password,
        settings.default_admin_name,
        "admin",
    )
    logger.info("Seeded default admin in Supabase: %s", settings.default_admin_email)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Every step here is best-effort: a hosted platform restarts the
    # instance on any startup exception, so a momentarily unreachable
    # Supabase must not take the whole deployment down - the affected
    # endpoints can report the problem per-request instead.
    try:
        storage_service.ensure_bucket()
    except Exception:
        logger.exception("Could not ensure the Supabase Storage bucket exists")

    try:
        _seed_default_admin()
    except Exception:
        logger.exception("Could not seed the default admin user in Supabase Auth")

    # Loading EasyOCR's models takes ~10s; do it off the startup path so the
    # health check passes immediately, but before the first scan arrives.
    threading.Thread(target=ocr_service.warm_up, name="easyocr-warmup", daemon=True).start()

    yield


app = FastAPI(
    title="Legal Metrology Compliance Checker",
    description="Automated screening of packaged commodity labels against the "
    "Legal Metrology (Packaged Commodities) Rules, 2011.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(scans.router)
app.include_router(dashboard.router)


# HEAD as well as GET: uptime monitors and some platform health checkers
# probe with HEAD, and FastAPI (unlike bare Starlette) does not add HEAD to a
# GET route automatically - so a HEAD probe against a @app.get route answers
# 405 Method Not Allowed and reads as an outage.
@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return {
        "service": "Legal Metrology Compliance Checker API",
        "status": "ok",
        "docs": "/docs",
    }


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {"status": "ok"}
