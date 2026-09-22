"""
OCR extraction service built on RapidOCR (PP-OCRv4 models running under ONNX
Runtime).

Why not EasyOCR, which this used before: EasyOCR runs on PyTorch, and
importing torch alone costs several hundred MB of resident memory before a
single model loads - more than a 512 MB instance has to give. The OOM killer
takes the worker mid-request, which the browser sees as a 502 with no CORS
headers on it. ONNX Runtime with the same family of models reads labels just
as well in a fraction of the footprint, and the models ship inside the wheel
rather than being downloaded on first use.

The engine is loaded lazily and cached, because loading it is slow relative
to a request and there is no reason to pay for it more than once.
"""
import ctypes
import logging
import threading
from typing import TypedDict

from . import image_preprocessing

logger = logging.getLogger(__name__)

_engine = None
_engine_lock = threading.Lock()

# One scan's detection pass allocates a few hundred MB at its peak. Two
# running at once is what actually tips a small instance over, and FastAPI
# runs sync endpoints on a thread pool, so concurrent uploads genuinely do
# overlap. Serialising the OCR itself trades a little latency under load for
# not being OOM-killed - the queued request still completes, where an
# OOM-killed worker takes every in-flight request down with it.
_ocr_gate = threading.Semaphore(1)

# The front-of-pack photo is only read to identify the product, which means
# the largest text on the pack - it does not need the resolution the
# declarations panel does, and shrinking it cuts that pass's cost sharply.
FRONT_OF_PACK_LONG_SIDE = 700


def _load_malloc_trim():
    """
    glibc's allocator keeps freed memory in its own heap rather than
    returning it to the kernel, so the resident size a scan peaks at stays
    charged to the process long after the scan finished - which is what a
    memory-capped host actually measures and kills on. malloc_trim() hands
    the free pages back; measured here it returns a post-scan process from
    ~300 MB to ~110 MB. Absent outside glibc (musl, macOS), hence the probe.
    """
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim.argtypes = [ctypes.c_size_t]
        libc.malloc_trim.restype = ctypes.c_int
        return libc.malloc_trim
    except (OSError, AttributeError):
        return None


_malloc_trim = _load_malloc_trim()


def release_memory() -> None:
    """Return this process's freed heap to the OS. Safe to call anywhere."""
    if _malloc_trim is not None:
        try:
            _malloc_trim(0)
        except Exception:  # pragma: no cover - defensive, never worth failing a scan
            logger.debug("malloc_trim failed", exc_info=True)


class OcrLine(TypedDict):
    text: str
    confidence: float
    bbox: list[list[float]]  # 4 (x, y) points
    height_px: float


def _get_engine():
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                from rapidocr_onnxruntime import RapidOCR

                _engine = RapidOCR(
                    # One thread per model: the deployment target is a
                    # fractional vCPU, where extra ONNX Runtime threads add
                    # their arenas and stacks to the memory bill and then
                    # contend for a core they don't have.
                    intra_op_num_threads=1,
                    inter_op_num_threads=1,
                    # The preprocessing pipeline already bounds and
                    # orientation-corrects the image, so the engine never
                    # needs to rescale and the angle classifier has nothing
                    # left to correct.
                    use_cls=False,
                    max_side_len=image_preprocessing.MAX_WORK_LONG_SIDE,
                )
                logger.info(
                    "OCR engine ready (working images bounded to %d px on the long edge)",
                    image_preprocessing.MAX_WORK_LONG_SIDE,
                )
    return _engine


def warm_up() -> bool:
    """
    Load the OCR models. Called in a background thread at startup so the
    first real scan doesn't pay the load cost. Returns True on success -
    never raises, because a failed warm-up should only make the first scan
    slow, not stop the app from serving.
    """
    try:
        _get_engine()
        return True
    except Exception:
        logger.exception("OCR warm-up failed; models will load on the first scan instead")
        return False


def extract_lines(image_bytes: bytes, max_long_side: int | None = None) -> list[OcrLine]:
    engine = _get_engine()
    processed, scale_factor = image_preprocessing.normalize_for_ocr(image_bytes, max_long_side)
    try:
        with _ocr_gate:
            result, _elapsed = engine(processed)
    finally:
        # The working image is the single largest object this function
        # holds; drop it before building the (tiny) line list rather than
        # keeping both alive until the caller returns, then hand the pages
        # the OCR pass churned through back to the OS.
        del processed
        release_memory()

    lines: list[OcrLine] = []
    for bbox, text, confidence in result or []:
        # bbox/height are in the (rescaled) working image's pixel space;
        # scale back to the original photo's pixel space so a
        # calibration_mm_per_px computed against the original photo still
        # gives correct font-size-in-mm measurements.
        scaled_bbox = [[float(x) / scale_factor, float(y) / scale_factor] for x, y in bbox]
        ys = [pt[1] for pt in scaled_bbox]
        lines.append(
            {
                "text": text,
                "confidence": float(confidence),
                "bbox": scaled_bbox,
                "height_px": float(max(ys) - min(ys)),
            }
        )
    return lines


def full_text(lines: list[OcrLine]) -> str:
    return "\n".join(l["text"] for l in lines)


def guess_product_name(lines: list[OcrLine]) -> str | None:
    """
    Naive fallback for when Groq isn't configured to identify the product
    from its front-of-pack photo: brand/product names are almost always the
    most prominent (tallest) text on the front of a package, so pick the
    tallest OCR line that looks like real text rather than noise.
    """
    candidates = [l for l in lines if len(l["text"].strip()) >= 3]
    if not candidates:
        return None
    return max(candidates, key=lambda l: l["height_px"])["text"].strip()
