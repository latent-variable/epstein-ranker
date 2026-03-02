from __future__ import annotations

import base64
import functools
import io
import json
import math
import random
import re
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

from .constants import RETRIABLE_HTTP_STATUS_CODES


class UnsupportedEndpointError(RuntimeError):
    """Raised when a specific API route is unavailable on the target server."""


class ModelRequestError(RuntimeError):
    """Raised for model request failures with retriable vs permanent classification."""

    def __init__(
        self,
        message: str,
        *,
        retriable: bool,
        status_code: Optional[int] = None,
        retry_after_seconds: Optional[float] = None,
    ) -> None:
        super().__init__(message)
        self.retriable = retriable
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds


class ContentFilterError(ModelRequestError):
    """Raised when a provider rejects input due to content filtering.

    This is a permanent, non-retriable failure — retrying will never succeed and
    wastes time + API credits.
    """

    def __init__(self, message: str, *, status_code: Optional[int] = None) -> None:
        super().__init__(message, retriable=False, status_code=status_code)


def _is_content_filter_error(response_text: str) -> bool:
    """Detect provider content-filter rejections from the HTTP response body."""
    lowered = response_text.lower()
    return "data_inspection_failed" in lowered or "inappropriate content" in lowered


def _is_empty_model_message_error(exc: Exception) -> bool:
    return isinstance(exc, ValueError) and "empty message" in str(exc).lower()


def _parse_retry_after_seconds(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    raw = value.strip()
    if not raw:
        return None
    try:
        delay = float(raw)
        if delay > 0:
            return delay
    except ValueError:
        pass
    try:
        dt = datetime.strptime(raw, "%a, %d %b %Y %H:%M:%S GMT").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    delta = (dt - datetime.now(timezone.utc)).total_seconds()
    if delta <= 0:
        return None
    return delta


def build_text_analysis_input(filename: str, text: str) -> str:
    return (
        "Analyze the following document and respond with the JSON schema "
        "described in the system prompt.\n"
        f"Filename: {filename}\n"
        "---------\n"
        f"{text.strip()}\n"
        "---------"
    )


def build_image_analysis_instruction(
    filename: str,
    page_count: int,
    *,
    total_pages: Optional[int] = None,
    start_page: int = 1,
    part_index: Optional[int] = None,
    part_total: Optional[int] = None,
) -> str:
    page_end = max(start_page, start_page + max(page_count, 1) - 1)
    part_label = ""
    if part_index is not None and part_total is not None and part_total > 1:
        part_label = f" (part {part_index}/{part_total})"
    if total_pages is not None and total_pages > page_count:
        pages_label = (
            f"sampled pages {start_page}-{page_end} of {total_pages} page(s){part_label}"
        )
    else:
        pages_label = (
            f"pages {start_page}-{page_end}{part_label}"
            if page_count > 0
            else "the provided page image"
        )
    return (
        "Analyze this source document image and respond with the JSON schema "
        "described in the system prompt.\n"
        f"Filename: {filename}\n"
        f"Input contains {pages_label}."
    )


def infer_api_priority(endpoint: str, *, prefer_openai: bool = False) -> List[str]:
    if prefer_openai:
        return ["openai", "chat"]
    normalized = endpoint.rstrip("/")
    if normalized.endswith("/api/v1"):
        return ["chat", "openai"]
    return ["openai", "chat"]


def derive_localhost_fallback_endpoints(endpoint: str) -> List[str]:
    normalized = endpoint.rstrip("/")
    if not re.match(r"^https?://(?:localhost|127\.0\.0\.1)(?::\d+)?(?:/.*)?$", normalized):
        return []
    scheme = "https" if normalized.startswith("https://") else "http"
    host = "127.0.0.1" if "127.0.0.1" in normalized else "localhost"
    candidates = [
        f"{scheme}://{host}:5555/api/v1",
        f"{scheme}://{host}:5555/v1",
    ]
    return [candidate for candidate in candidates if candidate != normalized]


def normalize_endpoint_base(endpoint: str) -> str:
    normalized = endpoint.strip().rstrip("/")
    lowered = normalized.lower()
    for suffix in ("/chat/completions", "/chat"):
        if lowered.endswith(suffix):
            return normalized[: -len(suffix)]
    return normalized


def build_request_targets(
    endpoint: str,
    api_format: str,
    *,
    prefer_openai: bool = False,
) -> List[Tuple[str, str]]:
    normalized = normalize_endpoint_base(endpoint)
    targets: List[Tuple[str, str]] = []

    if api_format == "openai":
        targets.append(("openai", normalized))
        for fallback_endpoint in derive_localhost_fallback_endpoints(normalized):
            targets.append(("openai", fallback_endpoint))
    elif api_format == "chat":
        targets.append(("chat", normalized))
        for fallback_endpoint in derive_localhost_fallback_endpoints(normalized):
            targets.append(("chat", fallback_endpoint))
    else:
        for mode in infer_api_priority(normalized, prefer_openai=prefer_openai):
            targets.append((mode, normalized))
        for fallback_endpoint in derive_localhost_fallback_endpoints(normalized):
            for mode in infer_api_priority(fallback_endpoint, prefer_openai=prefer_openai):
                targets.append((mode, fallback_endpoint))

    deduped: List[Tuple[str, str]] = []
    seen: Set[Tuple[str, str]] = set()
    for mode, base in targets:
        key = (mode, base)
        if key in seen:
            continue
        deduped.append(key)
        seen.add(key)
    return deduped


def image_path_to_data_url(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
    }
    mime = mime_map.get(suffix)
    if not mime:
        raise RuntimeError(f"Unsupported image suffix for multimodal input: {image_path}")
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def encode_image_bytes_to_data_url(image_bytes: bytes, *, mime: str) -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def encode_png_bytes_to_data_url(png_bytes: bytes) -> str:
    return encode_image_bytes_to_data_url(png_bytes, mime="image/png")


def _sanitize_debug_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "image"


def _get_pillow() -> Any:
    try:
        from PIL import Image  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Image packing/compression requires Pillow. Install via: python -m pip install pillow"
        ) from exc
    return Image


def _encode_raster_bytes(
    image_bytes: bytes,
    *,
    output_format: str,
    jpeg_quality: int,
    image_max_side: int,
) -> Tuple[str, Dict[str, Any]]:
    normalized_format = output_format.lower()
    if normalized_format not in {"png", "jpeg"}:
        raise RuntimeError(f"Unsupported image output format: {output_format}")

    # Fast path: keep existing PNG bytes untouched.
    if normalized_format == "png" and image_max_side <= 0:
        return (
            encode_image_bytes_to_data_url(image_bytes, mime="image/png"),
            {"bytes": len(image_bytes), "width": None, "height": None, "format": "png"},
        )

    Image = _get_pillow()
    with Image.open(io.BytesIO(image_bytes)) as image:
        raster = image.convert("RGB")
        width, height = raster.size
        if image_max_side > 0 and max(width, height) > image_max_side:
            scale = image_max_side / float(max(width, height))
            new_size = (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            )
            raster = raster.resize(new_size, Image.Resampling.LANCZOS)
        out = io.BytesIO()
        if normalized_format == "jpeg":
            raster.save(
                out,
                format="JPEG",
                quality=max(1, min(jpeg_quality, 95)),
                optimize=True,
                progressive=True,
            )
            mime = "image/jpeg"
        else:
            raster.save(out, format="PNG", optimize=True)
            mime = "image/png"
        final_width, final_height = raster.size
        raster.close()
        encoded_bytes = out.getvalue()
    return (
        encode_image_bytes_to_data_url(encoded_bytes, mime=mime),
        {
            "bytes": len(encoded_bytes),
            "width": final_width,
            "height": final_height,
            "format": normalized_format,
            "mime": mime,
        },
    )


def render_pdf_page_to_png_bytes(pdf_path: Path, *, page_number: int, dpi: int) -> bytes:
    with tempfile.TemporaryDirectory(prefix="epstein_pdf_render_") as tmpdir:
        out_prefix = Path(tmpdir) / f"page_{page_number}"
        cmd = [
            "pdftoppm",
            "-png",
            "-r",
            str(dpi),
            "-f",
            str(page_number),
            "-singlefile",
            str(pdf_path),
            str(out_prefix),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "pdftoppm is required for PDF image mode but was not found on PATH."
            ) from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "").strip()
            raise RuntimeError(f"Failed to render PDF page for {pdf_path}: {detail}") from exc
        rendered_path = out_prefix.with_suffix(".png")
        if not rendered_path.exists():
            raise RuntimeError(f"PDF rendering produced no image for {pdf_path}")
        return rendered_path.read_bytes()


def render_pdf_page_to_data_url(pdf_path: Path, *, page_number: int, dpi: int) -> str:
    return encode_png_bytes_to_data_url(
        render_pdf_page_to_png_bytes(pdf_path, page_number=page_number, dpi=dpi)
    )


@functools.lru_cache(maxsize=8192)
def detect_pdf_page_count(pdf_path: Path) -> Optional[int]:
    cmd = ["pdfinfo", str(pdf_path)]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        return None
    except subprocess.CalledProcessError:
        return None
    match = re.search(r"^Pages:\s+(\d+)\s*$", result.stdout, flags=re.MULTILINE)
    if not match:
        return None
    try:
        page_count = int(match.group(1))
    except ValueError:
        return None
    return page_count if page_count > 0 else None


def compose_png_pages(
    page_pngs: List[bytes],
    *,
    pages_per_image: int,
) -> Tuple[List[bytes], List[Dict[str, Any]]]:
    packed_images: List[bytes] = []
    packed_meta: List[Dict[str, Any]] = []
    if pages_per_image <= 1:
        for idx, png in enumerate(page_pngs):
            packed_meta.append(
                {
                    "block_index": idx + 1,
                    "source_pages": [idx + 1],
                    "bytes": len(png),
                    "format": "png",
                    "width": None,
                    "height": None,
                }
            )
            packed_images.append(png)
        return packed_images, packed_meta

    Image = _get_pillow()

    for start in range(0, len(page_pngs), pages_per_image):
        chunk = page_pngs[start : start + pages_per_image]
        if not chunk:
            continue
        images = []
        for png in chunk:
            with Image.open(io.BytesIO(png)) as image:
                images.append(image.convert("RGB"))
        cols = max(1, math.ceil(math.sqrt(len(images))))
        rows = max(1, math.ceil(len(images) / cols))
        cell_w = max(img.width for img in images)
        cell_h = max(img.height for img in images)
        canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
        for idx, img in enumerate(images):
            row = idx // cols
            col = idx % cols
            x = col * cell_w + max((cell_w - img.width) // 2, 0)
            y = row * cell_h + max((cell_h - img.height) // 2, 0)
            canvas.paste(img, (x, y))
            img.close()
        output = io.BytesIO()
        canvas.save(output, format="PNG", optimize=True)
        canvas.close()
        encoded_bytes = output.getvalue()
        packed_images.append(encoded_bytes)
        packed_meta.append(
            {
                "block_index": len(packed_images),
                "source_pages": list(range(start + 1, start + len(chunk) + 1)),
                "bytes": len(encoded_bytes),
                "format": "png",
                "width": cols * cell_w,
                "height": rows * cell_h,
            }
        )
    return packed_images, packed_meta


def prepare_image_data_urls(
    image_path: Path,
    *,
    max_pages: int,
    render_dpi: int,
    pdf_pages_per_image: int = 1,
    start_page: int = 1,
    image_output_format: str = "png",
    image_jpeg_quality: int = 85,
    image_max_side: int = 0,
    debug_image_dir: Optional[Path] = None,
) -> Tuple[List[str], int, Optional[int]]:
    suffix = image_path.suffix.lower()
    debug_dir: Optional[Path] = None
    debug_render_dir: Optional[Path] = None
    debug_packed_dir: Optional[Path] = None
    if debug_image_dir is not None:
        debug_stem = _sanitize_debug_stem(
            f"{image_path.stem}_p{start_page:05d}_max{max_pages:03d}"
        )
        debug_dir = debug_image_dir / debug_stem
        debug_render_dir = debug_dir / "rendered_pages"
        debug_packed_dir = debug_dir / "packed_images"
        debug_render_dir.mkdir(parents=True, exist_ok=True)
        debug_packed_dir.mkdir(parents=True, exist_ok=True)

    if suffix == ".pdf":
        total_start = time.monotonic()
        pdf_page_count = detect_pdf_page_count(image_path)
        first_page = max(1, start_page)
        if pdf_page_count is not None and first_page > pdf_page_count:
            raise RuntimeError(
                f"PDF page window start {first_page} exceeds page count {pdf_page_count} for {image_path}"
            )
        target_pages = (
            max(1, min(max_pages, pdf_page_count - first_page + 1))
            if pdf_page_count is not None
            else max_pages
        )
        render_start = time.monotonic()
        page_pngs: List[bytes] = []
        rendered_pages_meta: List[Dict[str, Any]] = []
        for page_number in range(first_page, first_page + target_pages):
            png_bytes = render_pdf_page_to_png_bytes(
                image_path,
                page_number=page_number,
                dpi=render_dpi,
            )
            page_pngs.append(png_bytes)
            rendered_pages_meta.append(
                {"page_number": page_number, "raw_bytes": len(png_bytes)}
            )
            if debug_render_dir is not None:
                debug_page_path = debug_render_dir / f"page_{page_number:05d}.png"
                debug_page_path.write_bytes(png_bytes)
        render_seconds = time.monotonic() - render_start

        pack_start = time.monotonic()
        packed_images, packed_meta = compose_png_pages(
            page_pngs,
            pages_per_image=max(1, pdf_pages_per_image),
        )
        final_urls: List[str] = []
        final_meta: List[Dict[str, Any]] = []
        for idx, packed_image in enumerate(packed_images):
            final_url, meta = _encode_raster_bytes(
                packed_image,
                output_format=image_output_format,
                jpeg_quality=image_jpeg_quality,
                image_max_side=max(0, image_max_side),
            )
            final_urls.append(final_url)
            merged_meta = dict(packed_meta[idx]) if idx < len(packed_meta) else {}
            local_pages = merged_meta.get("source_pages")
            absolute_pages: List[int] = []
            if isinstance(local_pages, list):
                for page_number in local_pages:
                    if isinstance(page_number, int) and page_number > 0:
                        absolute_pages.append(first_page + page_number - 1)
            merged_meta.update(
                {
                    "final_bytes": meta.get("bytes"),
                    "final_format": meta.get("format"),
                    "final_width": meta.get("width"),
                    "final_height": meta.get("height"),
                    "source_pages_absolute": absolute_pages,
                }
            )
            final_meta.append(merged_meta)
            if debug_packed_dir is not None:
                ext = ".jpg" if meta.get("format") == "jpeg" else ".png"
                if absolute_pages:
                    debug_packed_name = (
                        f"block_{idx + 1:03d}_"
                        f"p{absolute_pages[0]:05d}-{absolute_pages[-1]:05d}{ext}"
                    )
                else:
                    debug_packed_name = f"block_{idx + 1:03d}{ext}"
                debug_packed_path = debug_packed_dir / debug_packed_name
                debug_packed_path.write_bytes(
                    base64.b64decode(final_url.split(",", 1)[1].encode("ascii"))
                )
        pack_seconds = time.monotonic() - pack_start

        if debug_dir is not None:
            debug_summary = {
                "image_path": str(image_path),
                "start_page": first_page,
                "target_pages": target_pages,
                "pdf_total_pages": pdf_page_count,
                "render_dpi": render_dpi,
                "pages_per_image": max(1, pdf_pages_per_image),
                "image_output_format": image_output_format,
                "image_jpeg_quality": image_jpeg_quality,
                "image_max_side": max(0, image_max_side),
                "render_seconds": round(render_seconds, 4),
                "pack_seconds": round(pack_seconds, 4),
                "total_prepare_seconds": round(time.monotonic() - total_start, 4),
                "rendered_pages": rendered_pages_meta,
                "packed_blocks": final_meta,
            }
            (debug_dir / "summary.json").write_text(
                json.dumps(debug_summary, indent=2),
                encoding="utf-8",
            )
        return (
            final_urls,
            target_pages,
            pdf_page_count,
        )

    image_bytes = image_path.read_bytes()
    url, _meta = _encode_raster_bytes(
        image_bytes,
        output_format=image_output_format,
        jpeg_quality=image_jpeg_quality,
        image_max_side=max(0, image_max_side),
    )
    return [url], 1, None


def post_request(
    *,
    url: str,
    payload: Dict[str, Any],
    api_key: Optional[str],
    extra_headers: Optional[Dict[str, str]],
    timeout: float,
) -> Dict[str, Any]:
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if extra_headers:
        headers.update(extra_headers)

    try:
        response = requests.post(url, json=payload, timeout=timeout, headers=headers)
    except (requests.ConnectionError, requests.Timeout) as exc:
        raise ModelRequestError(f"Request failed for {url}: {exc}", retriable=True) from exc
    except requests.RequestException as exc:
        raise ModelRequestError(f"Request failed for {url}: {exc}", retriable=False) from exc

    if response.status_code in (404, 405):
        raise UnsupportedEndpointError(
            f"HTTP {response.status_code} from {url}: endpoint not supported by this server"
        )
    if response.status_code >= 400:
        snippet = response.text[:500].replace("\n", " ")
        # Detect content-filter rejections early — no point retrying these.
        if _is_content_filter_error(response.text):
            raise ContentFilterError(
                f"HTTP {response.status_code} from {url}: {snippet}",
                status_code=response.status_code,
            )
        retry_after_seconds = _parse_retry_after_seconds(response.headers.get("Retry-After"))
        if response.status_code == 429 and retry_after_seconds is None:
            # Providers often omit Retry-After; use a conservative cooldown.
            retry_after_seconds = 20.0
        retriable = (
            response.status_code in RETRIABLE_HTTP_STATUS_CODES
            or 500 <= response.status_code < 600
        )
        raise ModelRequestError(
            f"HTTP {response.status_code} from {url}: {snippet}",
            retriable=retriable,
            status_code=response.status_code,
            retry_after_seconds=retry_after_seconds,
        )

    try:
        return response.json()
    except json.JSONDecodeError as exc:
        snippet = response.text[:500].replace("\n", " ")
        raise ModelRequestError(
            f"Invalid JSON response from {url}: {snippet}",
            retriable=False,
        ) from exc


def extract_openai_content(data: Dict[str, Any]) -> str:
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ModelRequestError(
            f"Unexpected OpenAI response format: {data}",
            retriable=False,
        ) from exc
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            for key in ("text", "content", "output_text"):
                value = item.get(key)
                if isinstance(value, str):
                    text_parts.append(value)
                    break
        if text_parts:
            return "\n".join(text_parts)
        return ""
    if content is None:
        return ""
    raise ModelRequestError(
        f"Unexpected OpenAI response content type: {type(content)}",
        retriable=False,
    )


def extract_chat_content(data: Dict[str, Any]) -> str:
    output = data.get("output")
    if isinstance(output, list):
        collected: List[str] = []
        for item in output:
            if isinstance(item, str):
                collected.append(item)
                continue
            if isinstance(item, dict):
                value = item.get("content")
                if isinstance(value, str):
                    collected.append(value)
                    continue
                value = item.get("text")
                if isinstance(value, str):
                    collected.append(value)
        if collected:
            return "\n".join(collected)

    for key in ("content", "response", "text"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise ModelRequestError(
        f"Unexpected chat response format: {data}",
        retriable=False,
    )


def ensure_json_dict(content: str) -> Dict[str, Any]:
    candidate = content.strip()
    if not candidate:
        raise ValueError("Model returned an empty message.")

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or start >= end:
            raise
        parsed = json.loads(candidate[start : end + 1])

    if not isinstance(parsed, dict):
        raise TypeError(f"Expected a JSON object, received: {type(parsed)}")
    return parsed


def _coerce_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return max(0, int(round(value)))
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return max(0, int(stripped))
        except ValueError:
            try:
                parsed = float(stripped)
            except ValueError:
                return None
            if math.isnan(parsed) or math.isinf(parsed):
                return None
            return max(0, int(round(parsed)))
    return None


def _coerce_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        parsed = float(value)
        if math.isnan(parsed) or math.isinf(parsed):
            return None
        return parsed
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            parsed = float(stripped)
        except ValueError:
            return None
        if math.isnan(parsed) or math.isinf(parsed):
            return None
        return parsed
    return None


def _pick_int(data: Dict[str, Any], keys: List[str]) -> Optional[int]:
    for key in keys:
        if key in data:
            value = _coerce_int(data.get(key))
            if value is not None:
                return value
    return None


def _pick_float(data: Dict[str, Any], keys: List[str]) -> Optional[float]:
    for key in keys:
        if key in data:
            value = _coerce_float(data.get(key))
            if value is not None:
                return value
    return None


def _first_not_none(values: List[Optional[Any]]) -> Optional[Any]:
    for value in values:
        if value is not None:
            return value
    return None


def extract_response_usage(data: Dict[str, Any]) -> Tuple[Dict[str, int], Optional[float]]:
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    prompt_details = (
        usage.get("prompt_tokens_details")
        if isinstance(usage.get("prompt_tokens_details"), dict)
        else {}
    )
    completion_details = (
        usage.get("completion_tokens_details")
        if isinstance(usage.get("completion_tokens_details"), dict)
        else {}
    )

    prompt_tokens = _pick_int(usage, ["prompt_tokens", "input_tokens"])
    completion_tokens = _pick_int(usage, ["completion_tokens", "output_tokens"])
    total_tokens = _pick_int(usage, ["total_tokens"])
    if total_tokens is None:
        total_tokens = _pick_int(data, ["total_tokens"])
    cache_read_tokens = _first_not_none(
        [
            _pick_int(
                usage,
                ["cache_read_tokens", "cache_read_input_tokens", "cached_tokens"],
            ),
            _pick_int(
                prompt_details,
                ["cache_read_tokens", "cache_read_input_tokens", "cached_tokens"],
            ),
            _pick_int(
                completion_details,
                ["cache_read_tokens", "cache_read_input_tokens", "cached_tokens"],
            ),
        ]
    )
    cache_write_tokens = _first_not_none(
        [
            _pick_int(
                usage,
                ["cache_write_tokens", "cache_creation_input_tokens", "cache_creation_tokens"],
            ),
            _pick_int(
                prompt_details,
                ["cache_write_tokens", "cache_creation_input_tokens", "cache_creation_tokens"],
            ),
            _pick_int(
                completion_details,
                ["cache_write_tokens", "cache_creation_input_tokens", "cache_creation_tokens"],
            ),
        ]
    )

    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens
    if prompt_tokens is None and total_tokens is not None and completion_tokens is not None:
        prompt_tokens = max(total_tokens - completion_tokens, 0)
    if completion_tokens is None and total_tokens is not None and prompt_tokens is not None:
        completion_tokens = max(total_tokens - prompt_tokens, 0)

    normalized_usage: Dict[str, int] = {}
    if prompt_tokens is not None:
        normalized_usage["prompt_tokens"] = prompt_tokens
    if completion_tokens is not None:
        normalized_usage["completion_tokens"] = completion_tokens
    if total_tokens is not None:
        normalized_usage["total_tokens"] = total_tokens
    if cache_read_tokens is not None:
        normalized_usage["cache_read_tokens"] = cache_read_tokens
    if cache_write_tokens is not None:
        normalized_usage["cache_write_tokens"] = cache_write_tokens

    provider_reported_cost_usd = _first_not_none(
        [
            _pick_float(data, ["cost", "total_cost", "usage_cost"]),
            _pick_float(usage, ["cost", "total_cost", "usage_cost"]),
        ]
    )
    return normalized_usage, provider_reported_cost_usd


def _normalize_provider_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _provider_matches_expected(actual_provider: str, expected_provider: str) -> bool:
    actual = _normalize_provider_label(actual_provider)
    expected = expected_provider.strip().lower()
    expected_candidates = [expected]
    if "/" in expected:
        left, right = expected.split("/", 1)
        expected_candidates.extend([left, right])
    for candidate in expected_candidates:
        normalized_candidate = _normalize_provider_label(candidate)
        if not normalized_candidate:
            continue
        if (
            actual == normalized_candidate
            or actual.startswith(normalized_candidate)
            or normalized_candidate.startswith(actual)
        ):
            return True
    return False


def call_model(
    *,
    endpoint: str,
    api_format: str,
    model: str,
    filename: str,
    text: str,
    input_kind: str,
    image_path: Optional[Path],
    image_max_pages: int,
    image_render_dpi: int,
    system_prompt: str,
    api_key: Optional[str],
    timeout: float,
    max_retries: int,
    retry_backoff: float,
    temperature: float,
    max_output_tokens: int,
    reasoning_effort: Optional[str],
    image_detail: str,
    pdf_pages_per_image: int = 1,
    image_output_format: str = "png",
    image_jpeg_quality: int = 85,
    image_max_side: int = 0,
    debug_image_dir: Optional[Path] = None,
    image_start_page: int = 1,
    image_part_index: Optional[int] = None,
    image_part_total: Optional[int] = None,
    request_semaphore: Optional[threading.Semaphore] = None,
    http_referer: Optional[str] = None,
    x_title: Optional[str] = None,
    openrouter_provider: Optional[str] = None,
    openrouter_allow_fallbacks: Optional[bool] = None,
    config_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if input_kind not in {"text", "image"}:
        raise RuntimeError(f"Unsupported input kind: {input_kind}")
    if input_kind == "image" and api_format == "chat":
        raise RuntimeError(
            "Image mode requires OpenAI-style chat completions. Use --api-format openai or auto."
        )
    if input_kind == "image" and image_path is None:
        raise RuntimeError("Image mode requires a valid source file path.")

    effective_api_format = "openai" if (input_kind == "image" and api_format == "auto") else api_format
    targets = build_request_targets(
        endpoint,
        effective_api_format,
        prefer_openai=(input_kind == "image"),
    )
    extra_headers: Dict[str, str] = {}
    if http_referer:
        extra_headers["HTTP-Referer"] = http_referer
    if x_title:
        extra_headers["X-Title"] = x_title
    doc_input = build_text_analysis_input(filename, text)
    image_urls: List[str] = []
    prep_seconds: Optional[float] = None
    source_page_count: Optional[int] = None
    source_total_pages: Optional[int] = None
    if input_kind == "image" and image_path is not None:
        prep_start = time.monotonic()
        image_urls, source_page_count, source_total_pages = prepare_image_data_urls(
            image_path,
            max_pages=max(1, image_max_pages),
            render_dpi=max(72, image_render_dpi),
            pdf_pages_per_image=max(1, pdf_pages_per_image),
            image_output_format=image_output_format,
            image_jpeg_quality=max(1, min(image_jpeg_quality, 95)),
            image_max_side=max(0, image_max_side),
            debug_image_dir=debug_image_dir,
            start_page=max(1, image_start_page),
        )
        prep_seconds = time.monotonic() - prep_start
        doc_input = build_image_analysis_instruction(
            filename,
            source_page_count,
            total_pages=source_total_pages,
            start_page=max(1, image_start_page),
            part_index=image_part_index,
            part_total=image_part_total,
        )

    last_error: Optional[Exception] = None
    normalized_openrouter_provider = (
        openrouter_provider.strip()
        if isinstance(openrouter_provider, str) and openrouter_provider.strip()
        else None
    )
    # Some providers intermittently return blank content while still returning HTTP 200.
    # Give a couple of same-endpoint retries before consuming a full outer retry attempt.
    max_empty_content_retries = 2
    for attempt in range(1, max_retries + 1):
        saw_retriable_error = False
        rate_limit_wait_seconds = 0.0
        for mode, base_url in targets:
            try:
                target_request_attempt = 0
                while True:
                    target_request_attempt += 1

                    def send_request(*, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
                        if request_semaphore is None:
                            return post_request(
                                url=url,
                                payload=payload,
                                api_key=api_key,
                                extra_headers=extra_headers or None,
                                timeout=timeout,
                            )
                        request_semaphore.acquire()
                        try:
                            return post_request(
                                url=url,
                                payload=payload,
                                api_key=api_key,
                                extra_headers=extra_headers or None,
                                timeout=timeout,
                            )
                        finally:
                            request_semaphore.release()

                    if mode == "openai":
                        user_content: Any
                        if input_kind == "image":
                            content_blocks: List[Dict[str, Any]] = [{"type": "text", "text": doc_input}]
                            for image_url in image_urls:
                                image_payload: Dict[str, Any] = {"url": image_url}
                                if image_detail != "auto":
                                    image_payload["detail"] = image_detail
                                content_blocks.append(
                                    {"type": "image_url", "image_url": image_payload}
                                )
                            user_content = content_blocks
                        else:
                            user_content = doc_input
                        payload = {
                            "model": model,
                            "temperature": temperature,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_content},
                            ],
                        }
                        if max_output_tokens > 0:
                            payload["max_tokens"] = max_output_tokens
                        if reasoning_effort:
                            payload["reasoning"] = {"effort": reasoning_effort}
                        # Some OpenAI-compatible servers (for example strict vLLM deployments)
                        # reject unknown top-level fields like "metadata". Keep request payloads
                        # minimal unless targeting providers known to accept the field.
                        if config_metadata and "openrouter.ai" in base_url:
                            payload["metadata"] = config_metadata
                        provider_preferences: Optional[Dict[str, Any]] = None
                        if "openrouter.ai" in base_url:
                            provider_preferences = {}
                            if normalized_openrouter_provider:
                                provider_preferences["order"] = [normalized_openrouter_provider]
                            if openrouter_allow_fallbacks is not None:
                                provider_preferences["allow_fallbacks"] = bool(
                                    openrouter_allow_fallbacks
                                )
                            if normalized_openrouter_provider and openrouter_allow_fallbacks is False:
                                provider_preferences["only"] = [normalized_openrouter_provider]
                            if not provider_preferences:
                                provider_preferences = None
                        if provider_preferences:
                            payload["provider"] = provider_preferences
                        request_started = time.monotonic()
                        data = send_request(url=f"{base_url}/chat/completions", payload=payload)
                        request_seconds = time.monotonic() - request_started
                        usage, provider_reported_cost_usd = extract_response_usage(data)
                        response_provider = (
                            str(data.get("provider")).strip()
                            if isinstance(data.get("provider"), str)
                            else None
                        )
                        if (
                            "openrouter.ai" in base_url
                            and normalized_openrouter_provider
                            and openrouter_allow_fallbacks is False
                        ):
                            if not response_provider:
                                raise ModelRequestError(
                                    "OpenRouter strict provider mode requires a provider label in "
                                    "response, but none was returned.",
                                    retriable=False,
                                )
                            if not _provider_matches_expected(
                                response_provider,
                                normalized_openrouter_provider,
                            ):
                                raise ModelRequestError(
                                    "OpenRouter strict provider mismatch: expected "
                                    f"'{normalized_openrouter_provider}', got '{response_provider}'.",
                                    retriable=False,
                                )
                        content = extract_openai_content(data)
                    else:
                        payload = {
                            "model": model,
                            "system_prompt": system_prompt,
                            "input": doc_input,
                        }
                        provider_preferences = None
                        response_provider = None
                        request_started = time.monotonic()
                        data = send_request(url=f"{base_url}/chat", payload=payload)
                        request_seconds = time.monotonic() - request_started
                        usage, provider_reported_cost_usd = extract_response_usage(data)
                        content = extract_chat_content(data)
                    try:
                        parsed = ensure_json_dict(content)
                        parsed["_request_meta"] = {
                            "mode": mode,
                            "endpoint": base_url,
                            "attempt": attempt,
                            "request_attempt": target_request_attempt,
                            "request_seconds": round(request_seconds, 4),
                            "prep_seconds": round(prep_seconds, 4) if prep_seconds is not None else None,
                            "input_kind": input_kind,
                            "image_blocks": len(image_urls) if input_kind == "image" else 0,
                            "source_page_count": source_page_count,
                            "source_total_pages": source_total_pages,
                            "usage": usage,
                            "provider_reported_cost_usd": provider_reported_cost_usd,
                            "provider_preferences": provider_preferences,
                            "provider": response_provider,
                        }
                        return parsed
                    except (json.JSONDecodeError, TypeError, ValueError) as exc:
                        if (
                            _is_empty_model_message_error(exc)
                            and target_request_attempt <= max_empty_content_retries
                        ):
                            time.sleep(min(1.0, max(0.05, retry_backoff * 0.25)))
                            saw_retriable_error = True
                            continue
                        raise ModelRequestError(
                            f"Invalid JSON model output from {base_url}: {exc}",
                            retriable=True,
                        ) from exc
            except UnsupportedEndpointError as exc:
                last_error = exc
                continue
            except ContentFilterError:
                # Content-filter rejections are permanent — retrying will never
                # succeed.  Bail out immediately to free this slot.
                raise
            except ModelRequestError as exc:
                last_error = exc
                saw_retriable_error = saw_retriable_error or exc.retriable
                if exc.status_code == 429:
                    suggested_wait = (
                        exc.retry_after_seconds
                        if isinstance(exc.retry_after_seconds, (int, float)) and exc.retry_after_seconds > 0
                        else 20.0
                    )
                    # Add slight jitter to avoid synchronized retry bursts when many workers
                    # are throttled at once.
                    suggested_wait += random.uniform(0.0, 1.5)
                    rate_limit_wait_seconds = max(rate_limit_wait_seconds, suggested_wait)
                continue

        if attempt < max_retries and saw_retriable_error:
            sleep_seconds = retry_backoff * (2 ** (attempt - 1))
            if rate_limit_wait_seconds > 0:
                sleep_seconds = max(sleep_seconds, rate_limit_wait_seconds)
            time.sleep(max(0.0, sleep_seconds))
            continue
        break

    candidate_urls = ", ".join(
        f"{base}/{'chat/completions' if mode == 'openai' else 'chat'}"
        for mode, base in targets
    )
    detail = str(last_error) if last_error else "unknown error"
    raise RuntimeError(
        f"Failed to analyze after {attempt} attempt(s). Tried: {candidate_urls}. Last error: {detail}"
    )
