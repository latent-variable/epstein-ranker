#!/usr/bin/env python3
"""Rank Epstein audio/video files using a vision-language model.

Extracts frames from video files and/or transcribes audio, then sends
the frames + transcript to an OpenAI-compatible VL model for analysis.
Output schema matches the PDF pipeline (gpt_ranker.py) JSONL format.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import io
import math
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

# Reuse shared utilities from the existing ranker package
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ranker.model_client import (
    ModelRequestError,
    encode_image_bytes_to_data_url,
    ensure_json_dict,
    extract_openai_content,
    post_request,
)
from ranker.constants import (
    AGENCY_CANONICAL_MAP,
    LEAD_TYPE_CANONICAL_MAP,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AV_VIDEO_SUFFIXES = {".avi", ".mp4", ".mov", ".m4v", ".wmv", ".vob", ".ts", ".3gp", ".mkv", ".flv"}
AV_AUDIO_SUFFIXES = {".m4a", ".wav", ".opus", ".mp3", ".amr", ".aac", ".ogg", ".flac"}
AV_ALL_SUFFIXES = AV_VIDEO_SUFFIXES | AV_AUDIO_SUFFIXES

DEFAULT_ENDPOINT = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "qwen/qwen3-vl-30b-a3b-thinking"
DEFAULT_SYSTEM_PROMPT_PATH = Path("prompts") / "av_system_prompt.txt"
DEFAULT_SECONDS_PER_FRAME = 2.0          # extract 1 frame every X seconds
DEFAULT_MAX_FRAMES = 120   # hard cap so a long video doesn't generate thousands of frames
DEFAULT_FRAME_MAX_SIDE = 1024  # max dimension (px) for extracted frames
DEFAULT_FRAME_JPEG_QUALITY = 80
DEFAULT_GRID_COLS = 2      # pack this many frames horizontally per grid image
DEFAULT_GRID_ROWS = 2      # pack this many frames vertically per grid image
DEFAULT_WHISPER_MODEL = "small"
DEFAULT_MAX_PARALLEL = 4
DEFAULT_TIMEOUT = 300.0    # seconds
DEFAULT_CHUNK_SIZE = 1000  # rows per output chunk file
DEFAULT_MAX_OUTPUT_TOKENS = 4096
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF = 2.0

CHECKPOINT_VERSION = 1


# ---------------------------------------------------------------------------
# Frame extraction (ffmpeg)
# ---------------------------------------------------------------------------

def probe_duration(file_path: Path) -> Optional[float]:
    """Return video/audio duration in seconds, or None on failure."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(file_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        value = result.stdout.strip()
        if value and value != "N/A":
            return float(value)
    except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
        pass
    return None


def extract_frames_jpeg(
    file_path: Path,
    *,
    seconds_per_frame: float,
    max_frames: int,
    max_side: int,
    jpeg_quality: int,
) -> List[bytes]:
    """Extract JPEG frames from a video file at the requested seconds-per-frame rate.

    The number of frames extracted is: max(1, min(max_frames, int(duration / seconds_per_frame))).
    When duration is unknown, falls back to a single first-frame extraction.
    Uses per-frame seek (-ss) for reliability across all video lengths.
    Returns a list of raw JPEG bytes. Returns empty list if extraction fails.
    """
    duration = probe_duration(file_path)
    if duration is None or duration <= 0:
        duration = None  # will attempt first-frame fallback

    # q:v 2-5 is high quality JPEG in ffmpeg (lower = better quality)
    q_val = max(2, min(10, 10 - int(jpeg_quality * 8 / 100)))
    scale_filter = (
        f"scale='if(gt(iw,ih),min({max_side},iw),-2)':"
        f"'if(gt(iw,ih),-2,min({max_side},ih))'"
    )

    frame_bytes: List[bytes] = []

    if duration is not None and duration > 0:
        # Derive frame count from seconds_per_frame, clamped to [1, max_frames]
        num_frames = max(1, min(max_frames, int(duration / max(0.1, seconds_per_frame))))
        # Sample at evenly-spaced timestamps across the video
        # Add a small offset so we don't always land on the very first frame
        step = duration / (num_frames + 1)
        timestamps = [step * (i + 1) for i in range(num_frames)]
        # Clamp so we don't seek past the end
        timestamps = [min(ts, duration - 0.05) for ts in timestamps if ts < duration]
        timestamps = timestamps or [0.0]
    else:
        timestamps = [0.0]

    with tempfile.TemporaryDirectory(prefix="av_ranker_frames_") as tmpdir:
        for i, ts in enumerate(timestamps):
            out_path = os.path.join(tmpdir, f"frame_{i:03d}.jpg")
            cmd = [
                "ffmpeg", "-y",
                "-loglevel", "error",
                "-ss", f"{ts:.4f}",
                "-i", str(file_path),
                "-vf", scale_filter,
                "-frames:v", "1",
                "-q:v", str(q_val),
                out_path,
            ]
            try:
                subprocess.run(cmd, capture_output=True, timeout=60, check=False)
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
            p = Path(out_path)
            if p.exists() and p.stat().st_size > 100:
                try:
                    frame_bytes.append(p.read_bytes())
                except OSError:
                    pass

        if not frame_bytes:
            # Last-resort fallback: first frame without seeking
            fallback_path = os.path.join(tmpdir, "fallback.jpg")
            cmd_fb = [
                "ffmpeg", "-y",
                "-loglevel", "error",
                "-i", str(file_path),
                "-vf", scale_filter,
                "-frames:v", "1",
                "-q:v", str(q_val),
                fallback_path,
            ]
            try:
                subprocess.run(cmd_fb, capture_output=True, timeout=60, check=False)
                p = Path(fallback_path)
                if p.exists() and p.stat().st_size > 100:
                    frame_bytes.append(p.read_bytes())
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

    return frame_bytes


# ---------------------------------------------------------------------------
# Frame grid compositing
# ---------------------------------------------------------------------------

def compose_frame_grid(
    frame_bytes_list: List[bytes],
    *,
    grid_cols: int = DEFAULT_GRID_COLS,
    grid_rows: int = DEFAULT_GRID_ROWS,
    cell_size: int = 384,
    jpeg_quality: int = 80,
    gap: int = 4,
) -> List[bytes]:
    """Pack frames into grid composite images (e.g. 2×2).

    Each frame is resized to fit within *cell_size* × *cell_size*, then placed
    in a grid with *gap* pixel white separators.  Returns a list of JPEG byte
    arrays — one per grid image.  The last grid may have fewer cells (empty
    cells are left black).
    """
    from PIL import Image, ImageDraw, ImageFont

    frames_per_grid = grid_cols * grid_rows
    if frames_per_grid < 2:
        # No point compositing 1×1 grids — return originals
        return frame_bytes_list

    grid_w = grid_cols * cell_size + (grid_cols - 1) * gap
    grid_h = grid_rows * cell_size + (grid_rows - 1) * gap

    grids: List[bytes] = []
    total = len(frame_bytes_list)

    for batch_start in range(0, total, frames_per_grid):
        batch = frame_bytes_list[batch_start : batch_start + frames_per_grid]
        canvas = Image.new("RGB", (grid_w, grid_h), color=(0, 0, 0))
        draw = ImageDraw.Draw(canvas)

        for idx, fb in enumerate(batch):
            row, col = divmod(idx, grid_cols)
            try:
                img = Image.open(io.BytesIO(fb))
                img.thumbnail((cell_size, cell_size), Image.LANCZOS)
                x = col * (cell_size + gap)
                y = row * (cell_size + gap)
                # Center within cell
                x_offset = (cell_size - img.width) // 2
                y_offset = (cell_size - img.height) // 2
                canvas.paste(img, (x + x_offset, y + y_offset))
                # Small frame number label
                frame_num = batch_start + idx + 1
                draw.text((x + 4, y + 4), f"#{frame_num}", fill=(255, 255, 255))
            except Exception:
                continue

        buf = io.BytesIO()
        canvas.save(buf, format="JPEG", quality=jpeg_quality)
        grids.append(buf.getvalue())

    return grids


def extract_audio_track(file_path: Path, *, tmpdir: str) -> Optional[Path]:
    """Extract audio from a video file to a WAV file. Returns path or None."""
    out_path = Path(tmpdir) / "audio.wav"
    cmd = [
        "ffmpeg", "-y",
        "-loglevel", "error",
        "-i", str(file_path),
        "-vn",                    # no video
        "-acodec", "pcm_s16le",   # 16-bit PCM
        "-ar", "16000",           # 16kHz (Whisper standard)
        "-ac", "1",               # mono
        str(out_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120, check=False)
        if out_path.exists() and out_path.stat().st_size > 1000:
            return out_path
        else:
            print(f"  [warn] FFmpeg audio extraction failed or no audio track found for {file_path.name}", file=sys.stderr)
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  [warn] FFmpeg error on {file_path.name}: {e}", file=sys.stderr)
    return None


def convert_audio_to_wav(file_path: Path, *, tmpdir: str) -> Optional[Path]:
    """Convert an audio file to 16kHz mono WAV for Whisper. Returns path or None."""
    out_path = Path(tmpdir) / "audio.wav"
    cmd = [
        "ffmpeg", "-y",
        "-loglevel", "error",
        "-i", str(file_path),
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=120, check=False)
        if out_path.exists() and out_path.stat().st_size > 1000:
            return out_path
        else:
            print(f"  [warn] FFmpeg audio conversion failed for {file_path.name}", file=sys.stderr)
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  [warn] FFmpeg error on {file_path.name}: {e}", file=sys.stderr)
    return None


# ---------------------------------------------------------------------------
# Transcription (Whisper)
# ---------------------------------------------------------------------------

_whisper_lock = threading.Lock()
_whisper_model_cache: Dict[str, Any] = {}


def transcribe_audio(
    wav_path: Path,
    *,
    whisper_model: str = DEFAULT_WHISPER_MODEL,
) -> Optional[str]:
    """Transcribe a WAV file using OpenAI Whisper. Returns text or None."""
    global _whisper_model_cache
    try:
        import whisper  # type: ignore
    except ImportError:
        return None

    with _whisper_lock:
        if whisper_model not in _whisper_model_cache:
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")
                    _whisper_model_cache[whisper_model] = whisper.load_model(whisper_model)
            except Exception as e:
                print(f"  [error] Failed to load Whisper model {whisper_model}: {e}", file=sys.stderr)
                return None
        model = _whisper_model_cache[whisper_model]

    try:
        # Suppress whisper progress bars and ffmpeg spam
        import os, sys
        old_stdout = sys.stdout
        with open(os.devnull, "w") as devnull:
            sys.stdout = devnull
            try:
                # Must hold lock during inference: PyTorch Linear is not thread-safe by default for Whisper
                with _whisper_lock:
                    result = model.transcribe(str(wav_path), fp16=False, verbose=None)
            finally:
                sys.stdout = old_stdout

        text = result.get("text", "").strip()
        return text if text else None
    except Exception as e:
        print(f"  [error] Whisper transcription failed for {wav_path}: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Payload building
# ---------------------------------------------------------------------------

def build_av_user_message(
    filename: str,
    *,
    frame_data_urls: List[str],
    transcript: Optional[str],
    duration_seconds: Optional[float],
    is_audio_only: bool,
    total_frames: int = 0,
    frames_per_grid: int = 0,
) -> Any:
    """Build the user message content (list of text + image blocks)."""
    parts: List[Dict[str, Any]] = []

    # --- Text preamble ---
    file_type = "audio" if is_audio_only else "video"
    duration_str = f"{duration_seconds:.1f}s" if duration_seconds else "unknown"

    intro_lines = [
        f"Analyze this {file_type} file for investigative significance and respond with the JSON schema described in the system prompt.",
        f"Filename: {filename}",
        f"Duration: {duration_str}",
    ]
    if frame_data_urls and frames_per_grid > 1:
        intro_lines.append(
            f"Frames provided: {len(frame_data_urls)} grid image(s) containing {total_frames} evenly-spaced frames "
            f"({frames_per_grid} per grid, numbered in top-left corner). Examine each sub-frame for visual evidence."
        )
    elif frame_data_urls:
        intro_lines.append(f"Frames provided: {len(frame_data_urls)} evenly-spaced frames from the video.")
    if transcript:
        intro_lines.append(f"\nAudio transcript:\n---\n{transcript.strip()}\n---")
    elif not frame_data_urls:
        intro_lines.append("(No content could be extracted from this file.)")

    parts.append({"type": "text", "text": "\n".join(intro_lines)})

    # --- Image blocks ---
    for url in frame_data_urls:
        parts.append({
            "type": "image_url",
            "image_url": {"url": url, "detail": "low"},
        })

    return parts


def _repair_truncated_json(text: str) -> str:
    """Best-effort repair of JSON truncated by token limits.

    Finds the outermost ``{...`` block and attempts to close any
    unterminated strings, arrays, and objects so ``json.loads`` can
    parse it.
    """
    start = text.find("{")
    if start == -1:
        return text
    fragment = text[start:]

    # Close any unterminated string
    in_string = False
    escaped = False
    for ch in fragment:
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
    if in_string:
        fragment += '"'

    # Close unclosed brackets/braces
    stack: list[str] = []
    in_str = False
    esc = False
    for ch in fragment:
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()

    fragment += "".join(reversed(stack))
    return fragment


def call_av_model(
    *,
    endpoint: str,
    model: str,
    api_key: Optional[str],
    system_prompt: str,
    user_content: Any,
    max_output_tokens: int,
    timeout: float,
    max_retries: int,
    retry_backoff: float,
    http_referer: Optional[str],
    x_title: Optional[str],
    openrouter_provider: Optional[str],
    request_semaphore: Optional[threading.Semaphore],
) -> Dict[str, Any]:
    """Call the OpenAI-compatible chat completions endpoint for AV analysis."""
    base_url = endpoint.rstrip("/")
    for suffix in ("/chat/completions", "/chat"):
        if base_url.lower().endswith(suffix):
            base_url = base_url[: -len(suffix)]
            break

    extra_headers: Dict[str, str] = {}
    if http_referer:
        extra_headers["HTTP-Referer"] = http_referer
    if x_title:
        extra_headers["X-Title"] = x_title

    payload: Dict[str, Any] = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": max_output_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    }
    if openrouter_provider and "openrouter.ai" in base_url:
        payload["provider"] = {"order": [openrouter_provider], "allow_fallbacks": True}

    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            def _do_request() -> Dict[str, Any]:
                if request_semaphore is None:
                    return post_request(
                        url=f"{base_url}/chat/completions",
                        payload=payload,
                        api_key=api_key,
                        extra_headers=extra_headers or None,
                        timeout=timeout,
                    )
                request_semaphore.acquire()
                try:
                    return post_request(
                        url=f"{base_url}/chat/completions",
                        payload=payload,
                        api_key=api_key,
                        extra_headers=extra_headers or None,
                        timeout=timeout,
                    )
                finally:
                    request_semaphore.release()

            t0 = time.monotonic()
            data = _do_request()
            request_seconds = time.monotonic() - t0
            content = extract_openai_content(data)

            # Strip <think>...</think> blocks that thinking models produce
            import re as _re
            cleaned = _re.sub(r"<think>.*?</think>", "", content, flags=_re.DOTALL).strip()
            if not cleaned and content:
                # Model only returned thinking content, no JSON
                print(f"  [debug] Model returned only thinking content ({len(content)} chars), no JSON body")
            text_to_parse = cleaned if cleaned else content

            # Attempt to repair truncated JSON from token-limited responses
            try:
                parsed = ensure_json_dict(text_to_parse)
            except (json.JSONDecodeError, ValueError):
                repaired = _repair_truncated_json(text_to_parse)
                parsed = ensure_json_dict(repaired)
                parsed["_json_repaired"] = True

            parsed["_request_meta"] = {
                "attempt": attempt,
                "request_seconds": round(request_seconds, 4),
                "model": model,
                "endpoint": base_url,
            }
            return parsed

        except ModelRequestError as exc:
            last_error = exc
            if not exc.retriable or attempt >= max_retries:
                break
            wait = retry_backoff * (2 ** (attempt - 1))
            if exc.status_code == 429:
                wait = max(wait, exc.retry_after_seconds or 20.0)
            time.sleep(wait)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(retry_backoff)

    raise RuntimeError(
        f"Model call failed after {max_retries} attempt(s): {last_error}"
    )


# ---------------------------------------------------------------------------
# File processing
# ---------------------------------------------------------------------------

def process_av_file(
    file_path: Path,
    *,
    system_prompt: str,
    endpoint: str,
    model: str,
    api_key: Optional[str],
    max_output_tokens: int,
    timeout: float,
    max_retries: int,
    retry_backoff: float,
    http_referer: Optional[str],
    x_title: Optional[str],
    openrouter_provider: Optional[str],
    request_semaphore: Optional[threading.Semaphore],
    seconds_per_frame: float,
    max_frames: int,
    frame_max_side: int,
    frame_jpeg_quality: int,
    whisper_model: str,
    enable_transcription: bool,
    grid_cols: int = DEFAULT_GRID_COLS,
    grid_rows: int = DEFAULT_GRID_ROWS,
    enable_grid: bool = True,
) -> Dict[str, Any]:
    """Process a single AV file: extract frames + transcript, then call the model."""
    suffix = file_path.suffix.lower()
    is_audio_only = suffix in AV_AUDIO_SUFFIXES
    filename = file_path.name

    duration_seconds = probe_duration(file_path)
    frame_data_urls: List[str] = []
    transcript: Optional[str] = None
    total_raw_frames = 0
    frames_per_grid = 0
    prep_start = time.monotonic()

    with tempfile.TemporaryDirectory(prefix="av_ranker_proc_") as tmpdir:
        # --- Extract frames (video only) ---
        if not is_audio_only:
            frame_bytes_list = extract_frames_jpeg(
                file_path,
                seconds_per_frame=seconds_per_frame,
                max_frames=max_frames,
                max_side=frame_max_side,
                jpeg_quality=frame_jpeg_quality,
            )
            total_raw_frames = len(frame_bytes_list)

            # --- Composite into grids if enabled ---
            frames_per_grid = grid_cols * grid_rows
            if enable_grid and frames_per_grid > 1 and len(frame_bytes_list) > 1:
                grid_cell = max(256, frame_max_side // grid_cols)
                grid_bytes = compose_frame_grid(
                    frame_bytes_list,
                    grid_cols=grid_cols,
                    grid_rows=grid_rows,
                    cell_size=grid_cell,
                    jpeg_quality=frame_jpeg_quality,
                )
                for gb in grid_bytes:
                    frame_data_urls.append(
                        encode_image_bytes_to_data_url(gb, mime="image/jpeg")
                    )
            else:
                frames_per_grid = 0  # signal: no grid in use
                for fb in frame_bytes_list:
                    frame_data_urls.append(
                        encode_image_bytes_to_data_url(fb, mime="image/jpeg")
                    )

        # --- Extract + transcribe audio ---
        whisper_seconds = 0.0
        if enable_transcription:
            if is_audio_only:
                wav_path = convert_audio_to_wav(file_path, tmpdir=tmpdir)
            else:
                wav_path = extract_audio_track(file_path, tmpdir=tmpdir)

            if wav_path:
                t0_w = time.monotonic()
                transcript = transcribe_audio(wav_path, whisper_model=whisper_model)
                whisper_seconds = time.monotonic() - t0_w

    prep_seconds = time.monotonic() - prep_start
    total_raw_frames = total_raw_frames if not is_audio_only else 0
    frames_per_grid = frames_per_grid if not is_audio_only else 0

    user_content = build_av_user_message(
        filename,
        frame_data_urls=frame_data_urls,
        transcript=transcript,
        duration_seconds=duration_seconds,
        is_audio_only=is_audio_only,
        total_frames=total_raw_frames,
        frames_per_grid=frames_per_grid,
    )

    result = call_av_model(
        endpoint=endpoint,
        model=model,
        api_key=api_key,
        system_prompt=system_prompt,
        user_content=user_content,
        max_output_tokens=max_output_tokens,
        timeout=timeout,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
        http_referer=http_referer,
        x_title=x_title,
        openrouter_provider=openrouter_provider,
        request_semaphore=request_semaphore,
    )

    # Attach AV metadata
    result["_av_meta"] = {
        "filename": filename,
        "suffix": suffix,
        "file_type": "audio" if is_audio_only else "video",
        "file_size_bytes": file_path.stat().st_size,
        "duration_seconds": duration_seconds,
        "frames_extracted": total_raw_frames,
        "images_sent": len(frame_data_urls),
        "grid_layout": f"{grid_cols}x{grid_rows}" if (enable_grid and frames_per_grid > 1) else "none",
        "seconds_per_frame_requested": seconds_per_frame,
        "has_transcript": transcript is not None,
        "transcript_chars": len(transcript) if transcript else 0,
        "whisper_seconds": round(whisper_seconds, 4),
        "prep_seconds": round(prep_seconds, 4),
    }
    if result["_request_meta"]:
        result["_request_meta"]["prep_seconds"] = round(prep_seconds, 4)

    return result


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def load_checkpoint(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    done: Set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            sid = line.strip()
            if sid:
                done.add(sid)
    return done


def append_checkpoint(path: Path, source_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(source_id + "\n")


# ---------------------------------------------------------------------------
# JSONL output
# ---------------------------------------------------------------------------

def build_output_row(
    source_id: str,
    file_path: Path,
    volume: Optional[int],
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble a JSONL output row matching the PDF pipeline schema."""
    schema_keys = [
        "headline", "importance_score", "reason",
        "key_insights", "tags", "power_mentions",
        "agency_involvement", "lead_types",
    ]
    row: Dict[str, Any] = {
        "source_id": source_id,
        "filename": file_path.name,
        "volume": volume,
    }
    for key in schema_keys:
        row[key] = result.get(key, [] if key != "importance_score" else 0)
    row["_av_meta"] = result.get("_av_meta", {})
    row["_request_meta"] = result.get("_request_meta", {})
    return row


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_av_files(
    search_dirs: List[Path],
    *,
    suffixes: Optional[Set[str]] = None,
    only_source_ids: Optional[Set[str]] = None,
) -> List[Tuple[str, Path]]:
    """Find AV files and return [(source_id, path)] pairs."""
    target_suffixes = suffixes or AV_ALL_SUFFIXES
    efta_pattern = re.compile(r"^(EFTA\d+)", re.IGNORECASE)
    results: List[Tuple[str, Path]] = []
    seen: Set[str] = set()

    for search_dir in search_dirs:
        if not search_dir.is_dir():
            continue
        for file_path in sorted(search_dir.rglob("*")):
            if file_path.suffix.lower() not in target_suffixes:
                continue
            m = efta_pattern.match(file_path.stem)
            source_id = m.group(1).upper() if m else file_path.stem.upper()
            if source_id in seen:
                continue
            if only_source_ids and source_id not in only_source_ids:
                continue
            seen.add(source_id)
            results.append((source_id, file_path))

    return results


def load_only_source_ids(path: Path) -> Optional[Set[str]]:
    """Load a set of source IDs from a newline or JSONL file."""
    if not path.exists():
        return None
    ids: Set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and "source_id" in obj:
                    ids.add(str(obj["source_id"]).upper())
                    continue
            except json.JSONDecodeError:
                pass
            ids.add(line.upper())
    return ids or None


# ---------------------------------------------------------------------------
# Chunked output writer
# ---------------------------------------------------------------------------


class ChunkedWriter:
    """Writes JSONL rows into numbered chunk files and maintains a manifest."""

    def __init__(self, out_dir: Path, chunk_size: int, prefix: str = "av_ranked") -> None:
        self._out_dir = out_dir
        self._chunk_size = chunk_size
        self._prefix = prefix
        self._rows_written = 0
        self._current_chunk_rows = 0
        self._current_handle: Optional[Any] = None
        self._current_path: Optional[Path] = None
        self._chunk_paths: list[Path] = []
        self._lock = threading.Lock()
        # Count rows from existing chunk files to support resume
        self._scan_existing()

    def _scan_existing(self) -> None:
        import re
        pattern = re.compile(rf"{re.escape(self._prefix)}_(\d+)_(\d+)\.jsonl$")
        total = 0
        for f in sorted(self._out_dir.glob(f"{self._prefix}_*.jsonl")):
            if pattern.match(f.name):
                self._chunk_paths.append(f)
                with f.open("r", encoding="utf-8") as fh:
                    count = sum(1 for line in fh if line.strip())
                total += count
        self._rows_written = total
        # Figure out where we are in the current chunk
        if total > 0 and total % self._chunk_size != 0:
            self._current_chunk_rows = total % self._chunk_size
            # Reopen the last chunk for appending
            if self._chunk_paths:
                self._current_path = self._chunk_paths[-1]
                self._current_handle = self._current_path.open("a", encoding="utf-8")
        else:
            self._current_chunk_rows = 0

    def _chunk_path_for_range(self, start: int, end: int) -> Path:
        return self._out_dir / f"{self._prefix}_{start:05d}_{end:05d}.jsonl"

    def _rotate_if_needed(self) -> None:
        if self._current_handle is not None and self._current_chunk_rows < self._chunk_size:
            return
        if self._current_handle is not None:
            self._current_handle.close()
            self._current_handle = None
        start = self._rows_written + 1
        end = self._rows_written + self._chunk_size
        self._current_path = self._chunk_path_for_range(start, end)
        self._current_handle = self._current_path.open("a", encoding="utf-8")
        if self._current_path not in self._chunk_paths:
            self._chunk_paths.append(self._current_path)
        self._current_chunk_rows = 0

    def write_row(self, row_json: str) -> None:
        with self._lock:
            self._rotate_if_needed()
            assert self._current_handle is not None
            self._current_handle.write(row_json + "\n")
            self._current_handle.flush()
            self._rows_written += 1
            self._current_chunk_rows += 1

    def close(self) -> None:
        if self._current_handle is not None:
            self._current_handle.close()
            self._current_handle = None

    def write_manifest(self) -> Path:
        """Write a chunks.json manifest for this volume."""
        import re
        pattern = re.compile(rf"{re.escape(self._prefix)}_(\d+)_(\d+)\.jsonl$")
        chunks = []
        total_rows = 0
        for fp in sorted(self._chunk_paths):
            m = pattern.match(fp.name)
            if not m:
                continue
            row_count = 0
            if fp.exists():
                with fp.open("r", encoding="utf-8") as fh:
                    row_count = sum(1 for line in fh if line.strip())
            chunks.append({
                "start_row": int(m.group(1)),
                "end_row": int(m.group(2)),
                "json": str(fp),
                "row_count": row_count,
            })
            total_rows += row_count
        manifest = {
            "metadata": {
                "rows_processed": total_rows,
                "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "chunks": chunks,
        }
        manifest_path = self._out_dir / "chunks.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        return manifest_path

    def all_source_ids(self) -> Set[str]:
        """Read all source_id values from existing chunk files for resume."""
        ids: Set[str] = set()
        for fp in self._chunk_paths:
            if not fp.exists():
                continue
            with fp.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        sid = obj.get("source_id", "")
                        if sid:
                            ids.add(str(sid).upper())
                    except json.JSONDecodeError:
                        pass
        return ids


# ---------------------------------------------------------------------------
# Progress + stats
# ---------------------------------------------------------------------------

class Stats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.processed = 0
        self.skipped = 0
        self.failed = 0
        self.total_request_seconds = 0.0
        self.total_prep_seconds = 0.0
        self._start = time.monotonic()

    def record_success(self, request_seconds: float, prep_seconds: float) -> None:
        with self._lock:
            self.processed += 1
            self.total_request_seconds += request_seconds
            self.total_prep_seconds += prep_seconds

    def record_skip(self) -> None:
        with self._lock:
            self.skipped += 1

    def record_failure(self) -> None:
        with self._lock:
            self.failed += 1

    def summary(self) -> str:
        elapsed = time.monotonic() - self._start
        return (
            f"processed={self.processed} skipped={self.skipped} failed={self.failed} "
            f"elapsed={elapsed:.1f}s "
            f"avg_req={self.total_request_seconds / max(1, self.processed):.1f}s"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank Epstein audio/video files using a vision-language model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--volume", type=int, required=True,
        help="DOJ volume number (e.g. 10 or 11).",
    )
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path("data/new_data"),
        help="Root directory containing VOL*/NATIVES/ subdirectories.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("contrib/fta"),
        help="Output directory for JSONL results. A VOL subdirectory is created automatically.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Checkpoint file tracking processed source IDs. Auto-set per volume.",
    )
    parser.add_argument(
        "--endpoint", default=DEFAULT_ENDPOINT,
        help="OpenAI-compatible endpoint base URL.",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="Model ID to use.",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="API key (falls back to OPENROUTER_API_KEY env var).",
    )
    parser.add_argument(
        "--http-referer", default=None,
        help="HTTP-Referer header (recommended by OpenRouter).",
    )
    parser.add_argument(
        "--x-title", default=None,
        help="X-Title header (recommended by OpenRouter).",
    )
    parser.add_argument(
        "--openrouter-provider", default=None,
        help="OpenRouter provider slug for routing (e.g. nvidia).",
    )
    parser.add_argument(
        "--prompt-file", type=Path, default=DEFAULT_SYSTEM_PROMPT_PATH,
        help="Path to system prompt text file.",
    )
    parser.add_argument(
        "--seconds-per-frame", type=float, default=DEFAULT_SECONDS_PER_FRAME,
        help="Extract one frame every N seconds (default: 1.0). Use e.g. 2.0 to get half the frames.",
    )
    parser.add_argument(
        "--max-frames", type=int, default=DEFAULT_MAX_FRAMES,
        help="Hard cap on total frames extracted per video regardless of seconds-per-frame (default: 120).",
    )
    parser.add_argument(
        "--frame-max-side", type=int, default=DEFAULT_FRAME_MAX_SIDE,
        help="Max pixel dimension for extracted frames.",
    )
    parser.add_argument(
        "--frame-jpeg-quality", type=int, default=DEFAULT_FRAME_JPEG_QUALITY,
        help="JPEG quality for extracted frames (1-95).",
    )
    parser.add_argument(
        "--whisper-model", default=DEFAULT_WHISPER_MODEL,
        help="Whisper model size: tiny, base, small, medium, large.",
    )
    parser.add_argument(
        "--no-transcription", action="store_true",
        help="Disable Whisper transcription (frames only for video, skip audio-only files).",
    )
    parser.add_argument(
        "--max-parallel", type=int, default=DEFAULT_MAX_PARALLEL,
        help="Maximum concurrent model requests.",
    )
    parser.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT,
        help="HTTP request timeout in seconds.",
    )
    parser.add_argument(
        "--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS,
        help="Max completion tokens per request.",
    )
    parser.add_argument(
        "--max-retries", type=int, default=DEFAULT_MAX_RETRIES,
        help="Max retry attempts on transient failures.",
    )
    parser.add_argument(
        "--retry-backoff", type=float, default=DEFAULT_RETRY_BACKOFF,
        help="Base seconds for exponential retry backoff.",
    )
    parser.add_argument(
        "--only-source-ids-file", type=Path, default=None,
        help="File listing source IDs to process (skips all others).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip files already in the checkpoint.",
    )
    parser.add_argument(
        "--max-files", type=int, default=None,
        help="Limit total files to process (useful for tests).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Discover files and print what would be processed, without calling the model.",
    )
    parser.add_argument(
        "--file-types", choices=["all", "video", "audio"], default="all",
        help="Filter to process only video, only audio, or all AV files.",
    )
    parser.add_argument(
        "--grid-cols", type=int, default=DEFAULT_GRID_COLS,
        help="Columns in frame grid compositing (default: 2).",
    )
    parser.add_argument(
        "--grid-rows", type=int, default=DEFAULT_GRID_ROWS,
        help="Rows in frame grid compositing (default: 2).",
    )
    parser.add_argument(
        "--no-grid", action="store_true",
        help="Disable frame grid compositing (send each frame individually).",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
        help=f"Rows per output chunk file (default: {DEFAULT_CHUNK_SIZE}).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # --- Resolve API key ---
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    http_referer = args.http_referer or os.environ.get("OPENROUTER_REFERER")
    x_title = args.x_title or os.environ.get("OPENROUTER_TITLE")
    openrouter_provider = args.openrouter_provider or os.environ.get("OPENROUTER_PROVIDER")

    # --- Load system prompt ---
    prompt_path = args.prompt_file
    if not prompt_path.exists():
        print(f"[error] System prompt not found: {prompt_path}", file=sys.stderr)
        return 1
    system_prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not system_prompt:
        print(f"[error] System prompt is empty: {prompt_path}", file=sys.stderr)
        return 1

    # --- Determine search directories ---
    vol_tag = f"VOL{args.volume:05d}"
    search_root = args.data_dir / vol_tag / "NATIVES"
    if not search_root.is_dir():
        # Some volumes use a flat NATIVES layout without sub-volume dirs
        search_root = args.data_dir / vol_tag
    if not search_root.is_dir():
        print(f"[error] Volume directory not found: {search_root}", file=sys.stderr)
        return 1

    # --- Select file type filter ---
    if args.file_types == "video":
        target_suffixes = AV_VIDEO_SUFFIXES
    elif args.file_types == "audio":
        target_suffixes = AV_AUDIO_SUFFIXES
    else:
        target_suffixes = AV_ALL_SUFFIXES

    # --- Load filter/exclude lists ---
    only_source_ids: Optional[Set[str]] = None
    if args.only_source_ids_file:
        only_source_ids = load_only_source_ids(args.only_source_ids_file)
        if only_source_ids is None:
            print(f"[warn] --only-source-ids-file is empty or unreadable: {args.only_source_ids_file}")

    # --- Discover files ---
    print(f"[discover] Scanning {search_root} for {args.file_types} files...")
    av_files = discover_av_files(
        [search_root],
        suffixes=target_suffixes,
        only_source_ids=only_source_ids,
    )
    if not av_files:
        print(f"[done] No {args.file_types} files found in {search_root}.")
        return 0
    print(f"[discover] Found {len(av_files)} AV file(s).")

    # --- Setup output paths ---
    out_vol_dir = args.output_dir / vol_tag
    out_vol_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = args.checkpoint or out_vol_dir / ".av_checkpoint"

    # --- Initialize chunked writer ---
    writer = ChunkedWriter(out_vol_dir, chunk_size=args.chunk_size)

    # --- Load checkpoint + chunk source IDs for resume ---
    done_ids: Set[str] = set()
    if args.resume:
        done_ids = load_checkpoint(checkpoint_path)
        # Also include IDs already in chunk files
        done_ids |= writer.all_source_ids()
        if done_ids:
            print(f"[resume] Skipping {len(done_ids)} already-processed file(s).")

    # --- Filter already-processed ---
    pending = [(sid, fp) for sid, fp in av_files if sid not in done_ids]
    if len(done_ids) > 0:
        print(f"[filter] {len(av_files) - len(pending)} already done, {len(pending)} pending.")

    if args.max_files:
        pending = pending[: args.max_files]
        print(f"[filter] --max-files {args.max_files}: processing {len(pending)} file(s).")

    if not pending:
        print("[done] Nothing to process.")
        return 0

    if args.dry_run:
        print(f"[dry-run] Would process {len(pending)} file(s):")
        for sid, fp in pending[:20]:
            print(f"  {sid}  {fp.suffix}  {fp.stat().st_size:,} bytes")
        if len(pending) > 20:
            print(f"  ... and {len(pending) - 20} more")
        return 0

    enable_transcription = not args.no_transcription

    enable_grid = not args.no_grid

    print(f"[config] endpoint={args.endpoint}")
    print(f"[config] model={args.model}")
    print(f"[config] seconds_per_frame={args.seconds_per_frame} | max_frames={args.max_frames} | frame_max_side={args.frame_max_side}")
    grid_str = f"{args.grid_cols}x{args.grid_rows}" if enable_grid else "disabled"
    print(f"[config] grid={grid_str} | transcription={'enabled (' + args.whisper_model + ')' if enable_transcription else 'disabled'}")
    print(f"[config] max_parallel={args.max_parallel} | chunk_size={args.chunk_size} | output_dir={out_vol_dir}")
    print()

    semaphore = threading.Semaphore(args.max_parallel)
    stats = Stats()
    output_lock = threading.Lock()

    def process_one(sid: str, fp: Path) -> None:
        t0 = time.monotonic()
        try:
            result = process_av_file(
                fp,
                system_prompt=system_prompt,
                endpoint=args.endpoint,
                model=args.model,
                api_key=api_key,
                max_output_tokens=args.max_output_tokens,
                timeout=args.timeout,
                max_retries=args.max_retries,
                retry_backoff=args.retry_backoff,
                http_referer=http_referer,
                x_title=x_title,
                openrouter_provider=openrouter_provider,
                request_semaphore=semaphore,
                seconds_per_frame=args.seconds_per_frame,
                max_frames=args.max_frames,
                frame_max_side=args.frame_max_side,
                frame_jpeg_quality=args.frame_jpeg_quality,
                whisper_model=args.whisper_model,
                enable_transcription=enable_transcription,
                grid_cols=args.grid_cols,
                grid_rows=args.grid_rows,
                enable_grid=enable_grid,
            )
            row = build_output_row(sid, fp, args.volume, result)
            row_json = json.dumps(row, ensure_ascii=False)

            with output_lock:
                writer.write_row(row_json)
                append_checkpoint(checkpoint_path, sid)

            req_secs = result.get("_request_meta", {}).get("request_seconds", 0.0) or 0.0
            prep_secs = result.get("_av_meta", {}).get("prep_seconds", 0.0) or 0.0
            stats.record_success(req_secs, prep_secs)
            score = result.get("importance_score", "?")
            
            # Formatted log with exactly what was extracted
            frames_raw = result.get("_av_meta", {}).get("frames_extracted", 0)
            frames_out = result.get("_av_meta", {}).get("images_sent", 0)
            tx_len = result.get("_av_meta", {}).get("transcript_chars", 0)
            has_tx = result.get("_av_meta", {}).get("has_transcript", False)
            whisper_sec = result.get("_av_meta", {}).get("whisper_seconds", 0.0)
            
            tx_str = f"tx={tx_len}c ({whisper_sec:.1f}s)" if has_tx else "tx=none"
            frames_str = f"frames={frames_raw}/{frames_out}imgs" if frames_out else "frames=0/0imgs"
            elapsed = time.monotonic() - t0
            
            print(
                f"  [ok] {sid:<12} score={score:<2} | {frames_str:<21} | {tx_str:<20} | total={elapsed:.1f}s"
            )
        except Exception as exc:
            stats.record_failure()
            elapsed = time.monotonic() - t0
            print(f"  [fail] {sid}  {elapsed:.1f}s  {exc}", file=sys.stderr)

    print(f"[run] Processing {len(pending)} file(s) with up to {args.max_parallel} concurrent workers...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        futures = [executor.submit(process_one, sid, fp) for sid, fp in pending]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"[executor] Unhandled exception: {exc}", file=sys.stderr)

    writer.close()
    manifest_path = writer.write_manifest()

    print()
    print(f"[done] {stats.summary()}")
    print(f"[done] Output: {out_vol_dir} ({writer._rows_written} total rows, manifest: {manifest_path})")
    return 0


if __name__ == "__main__":
    # Load .env.openrouter if present (same convention as the rest of the pipeline)
    env_path = Path(__file__).parent / ".env.openrouter"
    if env_path.exists():
        with env_path.open() as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip("'\"")
                os.environ.setdefault(key, val)

    sys.exit(main())
