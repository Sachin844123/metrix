"""
Optional AI-assist layer using Groq's LLM API.

The deterministic rule engine (rule_engine.py) is the sole authority on
compliant/non-compliant verdicts - that keeps the checker auditable and
reproducible for enforcement use. Groq is used only in a supporting role,
and only ever to *add* information, never to overturn a verdict the rule
engine already reached:

  1. `identify_product_from_text` reads the OCR'd front-of-pack text to
     auto-fill the product name, brand, and category, so nothing has to be
     typed in manually. This is the default path, because the default
     GROQ_MODEL (openai/gpt-oss-120b) is a text-only model.
  2. `generate_summary` turns the rule engine's findings into a
     plain-language report summary.
  3. If - and only if - GROQ_VISION_MODEL is set to a vision-capable model,
     `identify_product` and `analyze_image` additionally send the photos
     themselves to that model as an independent second look: it can read
     declarations OCR garbled or split across lines, and can visually flag
     legibility issues (tiny font, low contrast) that a pixel-height
     measurement might miss.

If GROQ_API_KEY is not configured, every function no-ops and the app keeps
working on rule-engine output alone.

Note on model capability: Groq's catalog rotates, and text-only models
reject the multi-part `content` array a vision request needs with
`400 ... messages[0].content must be a string`. Rather than raising that on
every scan, the first such response marks vision unavailable for the rest
of the process and everything falls back to the text-only path above.
"""
import base64
import json
import logging
import threading

from ..config import settings
from . import image_preprocessing

logger = logging.getLogger(__name__)

_client = None

# Set once a vision request is rejected because the configured model can't
# accept images, so the remaining scans in this process skip vision instead
# of paying for (and logging) the same failure every time.
_vision_unavailable = False
_vision_lock = threading.Lock()

# Substrings Groq uses when a model cannot serve a request containing an
# image. Matched case-insensitively against the error message.
_NO_VISION_MARKERS = (
    "content must be a string",
    "must be a string",
    "does not support image",
    "image input",
    "image_url",
    "multimodal",
    "model_not_found",
    "does not exist",
)


def _get_client():
    global _client
    if not settings.groq_api_key:
        return None
    if _client is None:
        from groq import Groq

        _client = Groq(api_key=settings.groq_api_key)
    return _client


def is_enabled() -> bool:
    """True if Groq can be called at all (text model included)."""
    return bool(settings.groq_api_key)


def vision_enabled() -> bool:
    """
    True only if a separate vision-capable model is configured AND it hasn't
    already rejected an image request in this process. The default
    GROQ_VISION_MODEL is blank, because the default GROQ_MODEL
    (openai/gpt-oss-120b) is text-only - see the module docstring.
    """
    return bool(settings.groq_api_key and settings.groq_vision_model) and not _vision_unavailable


def _mark_vision_unavailable(reason: str) -> None:
    global _vision_unavailable
    with _vision_lock:
        if _vision_unavailable:
            return
        _vision_unavailable = True
    logger.warning(
        "GROQ_VISION_MODEL=%r cannot accept image input (%s). Disabling the Groq "
        "vision assist for this process and falling back to OCR + the text-only "
        "GROQ_MODEL. Either leave GROQ_VISION_MODEL blank or set it to a "
        "vision-capable model from https://console.groq.com/docs/vision.",
        settings.groq_vision_model,
        reason,
    )


def _is_vision_unsupported(exc: Exception) -> bool:
    """
    True if `exc` means "this model can't be given an image" (as opposed to a
    transient failure, a bad key, or an oversized payload - those should be
    retried on the next scan rather than disabling vision permanently).
    """
    status = getattr(exc, "status_code", None)
    if status not in (400, 404):
        return False
    message = str(exc).lower()
    return any(marker in message for marker in _NO_VISION_MARKERS)


def _parse_json_object(content: str | None) -> dict:
    """
    Parse a model's JSON reply. Reads the outermost {...} rather than the
    whole string, which also skips the ```json fences and stray prose some
    models add even in JSON mode. Returns {} if nothing parses.
    """
    if not content:
        return {}
    text = content.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        data = json.loads(text[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _normalize_identity(data: dict) -> dict:
    return {
        "product_name": (data.get("product_name") or "").strip(),
        "brand_name": (data.get("brand_name") or "").strip(),
        "category": (data.get("category") or "").strip(),
    }


def _fallback_summary(product_name: str, declarations: list[dict]) -> str:
    violations = [d for d in declarations if not d["compliant"]]
    if not violations:
        return f"{product_name} appears to comply with all checked Legal Metrology declarations."
    issues = "; ".join(f"{d['label']} ({d['issue']})" for d in violations)
    return f"{product_name} has {len(violations)} potential issue(s): {issues}."


def _vision_json(prompt: str, image_bytes: bytes, mime: str) -> dict:
    """
    Run a JSON-mode completion against GROQ_VISION_MODEL with an image
    attached. Returns the parsed object, or {} if vision is unavailable or
    the call failed - callers must have a text-only fallback.
    """
    client = _get_client()
    if client is None or not vision_enabled():
        return {}

    # Groq caps a base64-inlined image well below what a phone camera
    # produces, so downscale/recompress before encoding rather than letting
    # the request bounce back as a 400.
    try:
        payload_bytes, payload_mime = image_preprocessing.encode_for_vision(image_bytes)
    except Exception:
        logger.exception("Could not re-encode image for the Groq vision call; sending it as-is")
        payload_bytes, payload_mime = image_bytes, mime

    b64 = base64.b64encode(payload_bytes).decode("ascii")
    try:
        resp = client.chat.completions.create(
            model=settings.groq_vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{payload_mime};base64,{b64}"},
                        },
                    ],
                }
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        if _is_vision_unsupported(exc):
            _mark_vision_unavailable(str(exc))
        else:
            logger.warning("Groq vision request failed: %s", exc)
        return {}

    return _parse_json_object(resp.choices[0].message.content)


def _text_json(prompt: str) -> dict:
    """JSON-mode completion against the text-only GROQ_MODEL."""
    client = _get_client()
    if client is None:
        return {}
    try:
        resp = client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        logger.warning("Groq text request failed: %s", exc)
        return {}
    return _parse_json_object(resp.choices[0].message.content)


_IDENTIFY_FIELDS = (
    "1. product_name - the full product name as printed (include "
    "variant/flavour and pack size if shown, e.g. 'Refined Sunflower "
    "Oil 1 L').\n"
    "2. brand_name - the brand/company name.\n"
    "3. category - a short general category (e.g. 'Edible Oil', "
    "'Biscuits', 'Rice', 'Snacks').\n\n"
    "Respond ONLY with a JSON object: "
    '{"product_name": "...", "brand_name": "...", "category": "..."} '
    "- use an empty string for any field you genuinely cannot determine."
)


def identify_product(image_bytes: bytes, mime: str) -> dict:
    """
    Read a front-of-pack photo with a vision model and identify the product.
    Returns {"product_name": str, "brand_name": str, "category": str} with
    empty strings for anything it couldn't determine, or {} if no vision
    model is configured or the call failed - in which case the caller should
    fall back to identify_product_from_text().
    """
    data = _vision_json(
        "Look at this photo of the front of a packaged consumer product "
        "sold in India. Identify:\n" + _IDENTIFY_FIELDS,
        image_bytes,
        mime,
    )
    return _normalize_identity(data) if data else {}


def identify_product_from_text(ocr_text: str) -> dict:
    """
    Identify the product from the OCR'd text of its front-of-pack photo,
    using the text-only GROQ_MODEL. This is the default identification path:
    it needs no vision-capable model, just the text OCR already extracted.
    Returns {} if Groq isn't configured or the text is unusable.
    """
    if not is_enabled() or not ocr_text.strip():
        return {}
    data = _text_json(
        "The following text was read by OCR from the front of a packaged "
        "consumer product sold in India. It may contain OCR noise, split "
        "words, and marketing copy mixed in with the product's real name.\n\n"
        f"OCR text:\n\"\"\"\n{ocr_text[:4000]}\n\"\"\"\n\n"
        "From that text alone, identify:\n" + _IDENTIFY_FIELDS
    )
    return _normalize_identity(data) if data else {}


def analyze_image(image_bytes: bytes, mime: str, product_name: str, declarations: list[dict]) -> dict:
    """
    Send the label photo to a vision model as an independent check.
    Returns {"recovered": {label: snippet}, "visual_notes": str, "summary": str}
    - empty if no vision model is configured or the call failed, in which
    case the caller should fall back to generate_summary() for the summary.
    """
    if not vision_enabled():
        return {}

    missing_labels = [d["label"] for d in declarations if not d["found"]]
    prompt = (
        "You are assisting a Legal Metrology compliance inspector reviewing "
        f"a label photo for the product '{product_name}', sold as a packaged "
        "commodity in India. An automated OCR + rule-based check already "
        "ran; here are its results as JSON (do not contradict a declaration "
        "already marked compliant):\n"
        f"{json.dumps(declarations)}\n\n"
        "Look at the attached label image yourself and:\n"
        "1. For each of these declarations the automated check could NOT "
        f"find: {json.dumps(missing_labels)} - check if it is actually "
        "visible in the photo (OCR sometimes misses stylised or tiny "
        "text). Only report it if you can actually read it in the image.\n"
        "2. Note any visible legibility problems - e.g. a declaration "
        "printed in a font that looks disproportionately small, low "
        "contrast against the background, or partly obscured.\n"
        "3. Write a concise 2-4 sentence plain-language compliance summary "
        "for the inspector, referencing rule numbers where given.\n\n"
        "Respond ONLY with a JSON object of the form "
        '{"recovered": {"<declaration label>": "<exact text you read, '
        'or omit if not found>"}, "visual_notes": "<string, empty if none>", '
        '"summary": "<string>"}'
    )

    data = _vision_json(prompt, image_bytes, mime)
    if not data:
        return {}
    recovered = data.get("recovered")
    return {
        "recovered": {
            k: v
            for k, v in (recovered if isinstance(recovered, dict) else {}).items()
            if isinstance(v, str) and v.strip()
        },
        "visual_notes": data.get("visual_notes") or "",
        "summary": data.get("summary") or "",
    }


def generate_summary(product_name: str, declarations: list[dict]) -> str:
    """Plain-language report summary from the text-only GROQ_MODEL."""
    client = _get_client()
    if client is None:
        return _fallback_summary(product_name, declarations)

    prompt = (
        "Write a concise (2-4 sentence) plain-language compliance summary for "
        "an enforcement inspector, based on this automated Legal Metrology "
        "label check. Be factual and neutral, reference rule numbers where "
        f"given.\n\nProduct: {product_name}\n\n"
        f"Declaration results (JSON): {json.dumps(declarations)}"
    )

    try:
        resp = client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return (resp.choices[0].message.content or "").strip() or _fallback_summary(
            product_name, declarations
        )
    except Exception as exc:
        logger.warning("Groq generate_summary failed, using the built-in summary instead: %s", exc)
        return _fallback_summary(product_name, declarations)
