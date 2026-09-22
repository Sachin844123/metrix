"""
Image preprocessing to make real-world phone photos of product labels
readable by OCR. Phone photos routinely arrive rotated (EXIF orientation),
undersized for small print (nutrition tables, net-quantity text), and low
contrast (glossy/foil packaging, uneven lighting) - all of which badly
degrade OCR accuracy if the raw bytes are handed to the engine directly.

Everything here is written to hold a small, predictable memory ceiling,
because the deployed instance runs on a 512 MB box. Two rules do most of
that work:

  * Nothing is processed at full phone-camera resolution. A 12 MP photo is
    ~36 MB per RGB array copy, and a pipeline that makes four copies of it
    is already over budget before OCR loads a single model. Every image is
    scaled into a bounded working size first.
  * Work happens in grayscale (a third of the bytes of BGR), which is all
    the OCR models and the quality heuristics actually need.
"""
import io

import cv2
import numpy as np
from PIL import Image, ImageOps

from ..config import settings

# Hard ceiling on the working image, and the main reason this module exists.
# The detector does NOT downscale its input for images in this range (its
# "min" limit only ever scales small images up), so whatever is handed to it
# is what its activations are sized against. See settings.ocr_max_long_side.
MAX_WORK_LONG_SIDE = settings.ocr_max_long_side

# ...and a floor, because a downscaled/thumbnail upload has genuinely too
# little detail for small print and is worth upscaling into range. Bounded
# by the ceiling above when the two conflict.
MIN_WORK_SHORT_SIDE = min(700, MAX_WORK_LONG_SIDE)

# OpenCV spawns a worker thread per core for several of these filters; each
# one's stack and scratch buffers are pure overhead on a fractional-CPU
# instance, where the threads contend rather than parallelise.
cv2.setNumThreads(1)

# Groq accepts a far larger image by URL than inlined as a base64 data URL,
# and a just-uploaded photo has no public URL - so anything sent to the
# vision model has to fit the (much smaller) base64 budget. Base64 inflates
# bytes by 4/3, so a ~3 MB JPEG stays comfortably inside Groq's 4 MB limit
# while still being far more detail than a label needs.
MAX_VISION_BYTES = 3 * 1024 * 1024
MAX_VISION_LONG_SIDE = 1600


def _work_scale(width: int, height: int, max_long_side: int) -> float:
    """
    How much to scale an image of this size to land inside the working
    bounds. Downscaling wins over upscaling when the two rules conflict (a
    long, narrow panorama), since the ceiling is a memory limit and the
    floor is only a quality preference.
    """
    long_side, short_side = max(width, height), min(width, height)
    scale = min(1.0, max_long_side / long_side)
    if short_side * scale < MIN_WORK_SHORT_SIDE:
        scale = min(MIN_WORK_SHORT_SIDE / short_side, max_long_side / long_side)
    return scale


def load_working_gray(image_bytes: bytes, max_long_side: int | None = None) -> tuple[np.ndarray, float]:
    """
    Decode a photo into a bounded-size, EXIF-corrected grayscale array.

    Returns (array, scale_factor), where scale_factor is the array's size
    relative to the EXIF-corrected original - callers measuring pixel
    distances must divide by it to get back to the original photo's pixel
    space, which is what any mm-per-px calibration was computed against.
    """
    max_long_side = max_long_side or MAX_WORK_LONG_SIDE
    img = Image.open(io.BytesIO(image_bytes))
    original_width, original_height = img.size
    original_long_side = max(original_width, original_height)
    target_long_side = max(
        1, round(original_long_side * _work_scale(original_width, original_height, max_long_side))
    )

    # draft() lets the JPEG decoder emit grayscale at a reduced DCT scale
    # directly, so a 12 MP photo never exists in memory at full size or in
    # colour. It only reduces by powers of two, and is a no-op for formats
    # that can't do this at all (e.g. PNG), so an exact resize still follows.
    try:
        img.draft("L", (target_long_side, target_long_side))
    except (AttributeError, ValueError):
        pass

    img = ImageOps.exif_transpose(img)  # phones store rotation in EXIF, not pixels
    img = img.convert("L")

    # Work in long-side terms rather than (width, height): EXIF rotation may
    # have swapped the axes since the scale was computed, and the long side
    # is the one bound both working limits are expressed against.
    current_long_side = max(img.size)
    if current_long_side != target_long_side:
        ratio = target_long_side / current_long_side
        img = img.resize(
            (max(1, round(img.width * ratio)), max(1, round(img.height * ratio))), Image.LANCZOS
        )

    arr = np.array(img, dtype=np.uint8)
    scale_factor = max(img.size) / original_long_side
    img.close()
    return arr, scale_factor


def _enhance(arr: np.ndarray) -> np.ndarray:
    """
    CLAHE + edge-preserving denoise + a mild unsharp mask, in that order.

    Deliberately *not* fastNlMeansDenoising: it is the best-quality denoiser
    OpenCV ships and also, by a wide margin, the most expensive - tens of
    seconds and several full-size scratch buffers on a multi-megapixel
    photo. A bilateral filter keeps the edges that matter for character
    shapes at a small fraction of that cost.
    """
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    arr = clahe.apply(arr)
    arr = cv2.bilateralFilter(arr, d=5, sigmaColor=40, sigmaSpace=40)
    blurred = cv2.GaussianBlur(arr, (0, 0), sigmaX=1.2)
    return cv2.addWeighted(arr, 1.5, blurred, -0.5, 0)


def normalize_for_ocr(image_bytes: bytes, max_long_side: int | None = None) -> tuple[np.ndarray, float]:
    """
    Returns (grayscale array ready for OCR, scale_factor) - see
    load_working_gray for what scale_factor means and why callers need it.

    An array is returned rather than encoded bytes so the OCR engine can
    consume it directly, with no encode/decode round trip and no second
    full-size copy of the image alive at once.

    Falls back to the unenhanced (but correctly scaled) image if any
    enhancement step fails, so a preprocessing bug never breaks scanning
    outright.
    """
    arr, scale = load_working_gray(image_bytes, max_long_side)
    try:
        return _enhance(arr), scale
    except Exception:
        return arr, scale


def blur_score(image_bytes: bytes) -> float:
    """
    Variance of the Laplacian - a low score means the photo is likely too
    blurry for reliable OCR. Used to surface a hint to the user, not to
    block scanning.

    Measured on the same bounded working image the OCR engine sees, which
    also makes the score comparable between a 2 MP and a 12 MP upload -
    Laplacian variance otherwise scales with resolution.
    """
    arr, _ = load_working_gray(image_bytes)
    # CV_32F, not CV_64F: half the bytes for a statistic that never needs
    # double precision.
    return float(cv2.Laplacian(arr, cv2.CV_32F).var())


def short_side_px(image_bytes: bytes) -> int:
    """
    The photo's shorter dimension in pixels, as uploaded. Scaling (done in
    load_working_gray) helps OCR but can't invent detail a low-resolution
    source photo never captured - this is used to warn the user separately
    from the blur check.

    Reads the image header only; min(width, height) is unchanged by any EXIF
    rotation, so there is no need to decode a single pixel.
    """
    with Image.open(io.BytesIO(image_bytes)) as img:
        return min(img.size)


def make_thumbnail(image_bytes: bytes, long_side: int, max_bytes: int | None = None) -> bytes:
    """
    EXIF-corrected JPEG no larger than `long_side` on its longest edge, and
    - if max_bytes is given - recompressed until it fits that budget.
    """
    img = Image.open(io.BytesIO(image_bytes))
    try:
        img.draft("RGB", (long_side, long_side))
    except (AttributeError, ValueError):
        pass
    img = ImageOps.exif_transpose(img).convert("RGB")
    if max(img.size) > long_side:
        img.thumbnail((long_side, long_side), Image.LANCZOS)

    for quality in (85, 70, 55, 40):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        data = buf.getvalue()
        if max_bytes is None or len(data) <= max_bytes:
            return data

    # Still too big (a very large, very noisy photo) - halve the dimensions
    # and take whatever that gives; the alternative is a guaranteed rejection.
    img.thumbnail((img.width // 2, img.height // 2), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=55, optimize=True)
    return buf.getvalue()


def encode_for_vision(image_bytes: bytes) -> tuple[bytes, str]:
    """
    Returns (JPEG bytes, mime) for a photo about to be base64-inlined into a
    Groq vision request. A raw 12-megapixel phone photo is several times
    Groq's base64 limit and would come back as a 400; label text is still
    perfectly legible at this size.
    """
    return make_thumbnail(image_bytes, MAX_VISION_LONG_SIDE, MAX_VISION_BYTES), "image/jpeg"
