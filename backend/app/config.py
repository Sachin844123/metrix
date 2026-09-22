import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class Settings:
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./legal_metrology.db")

    default_admin_email: str = os.getenv("DEFAULT_ADMIN_EMAIL", "admin@legalmetrology.gov.in")
    default_admin_password: str = os.getenv("DEFAULT_ADMIN_PASSWORD", "Admin@123")
    default_admin_name: str = os.getenv("DEFAULT_ADMIN_NAME", "Chief Inspector")

    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    # Blank by default, and deliberately separate from GROQ_MODEL: the
    # default text model is text-only, and sending it an image rejects the
    # whole request with "messages[0].content must be a string". Set this
    # only to a genuinely vision-capable model
    # (https://console.groq.com/docs/vision) to turn the photo-review assist
    # on; everything works without it via OCR + the text model.
    groq_vision_model: str = os.getenv("GROQ_VISION_MODEL", "").strip()

    # Upper bound on a single uploaded photo. Two of these are read fully
    # into memory per scan, on an instance whose whole budget may be 512 MB,
    # so an unbounded upload is a straightforward way to OOM the worker.
    # Phone photos are comfortably under this; it only rejects the absurd.
    max_upload_bytes: int = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))

    # Longest edge, in pixels, that any image is scaled to before OCR. This
    # is the single biggest lever on peak memory: the text detector's
    # activations scale with its input, and measured end to end a scan peaks
    # at roughly 410 MB at 960 px but 500 MB+ at 1200 px, against the 512 MB
    # of a free-tier instance. Raise it on a larger instance - more pixels
    # means fine print (ingredients, care addresses) reads more reliably.
    ocr_max_long_side: int = int(os.getenv("OCR_MAX_LONG_SIDE", "960"))

    cors_origins: list[str] = [
        o.strip()
        for o in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
        if o.strip()
    ]

    upload_dir: Path = BASE_DIR / "uploads"
    report_dir: Path = BASE_DIR / "reports"

    # Supabase Storage for label images. If left blank, uploads fall back to
    # local disk (settings.upload_dir) - useful for zero-setup local dev.
    supabase_url: str = os.getenv("SUPABASE_URL", "").rstrip("/")
    supabase_service_role_key: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    supabase_storage_bucket: str = os.getenv("SUPABASE_STORAGE_BUCKET", "label-images")

    # Supabase Auth. The anon/publishable key identifies this app as a client
    # when verifying a user's own session; the service role key is needed for
    # admin actions (creating users, listing users). If either is blank, auth
    # is unavailable and the app will fail to start - Supabase Auth is not
    # optional the way Storage/Groq are, since there's no local fallback.
    supabase_anon_key: str = os.getenv("SUPABASE_ANON_KEY", "")

    @property
    def storage_enabled(self) -> bool:
        return bool(self.supabase_url and self.supabase_service_role_key)

    @property
    def auth_enabled(self) -> bool:
        return bool(self.supabase_url and self.supabase_service_role_key and self.supabase_anon_key)


settings = Settings()
settings.upload_dir.mkdir(exist_ok=True)
settings.report_dir.mkdir(exist_ok=True)
