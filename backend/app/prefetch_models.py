"""
Download EasyOCR's detection/recognition models ahead of time.

Run as a build step (`python -m app.prefetch_models`) so a deployed
instance starts with the ~100 MB of models already on disk. Without this,
the first scan request after every deploy triggers the download inline -
which is slow enough to time out the request and, on a platform with an
ephemeral filesystem, happens again after each restart.

Imports nothing that touches the database or Supabase, so it is safe to run
during a build where those aren't reachable yet.
"""
import sys

from .config import settings
from .services import ocr_service


def main() -> int:
    print(f"Prefetching EasyOCR models into {settings.easyocr_model_dir} ...")
    if not ocr_service.warm_up():
        # Don't fail the build over this: the app still works, it just pays
        # the download cost on its first scan.
        print("WARNING: could not prefetch EasyOCR models; they will download on first use.")
        return 0
    print("EasyOCR models ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
