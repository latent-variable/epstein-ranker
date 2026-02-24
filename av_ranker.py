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
DEFAULT_MODEL = "nvidia/nemotron-nano-12b-v2-vl:free"
DEFAULT_SYSTEM_PROMPT_PATH = Path("prompts") / "av_system_prompt.txt"
DEFAULT_FPS = 1.0          # frames per second to extract from video
DEFAULT_MAX_FRAMES = 120   # hard cap so a long video doesn't generate thousands of frames
DEFAULT_FRAME_MAX_SIDE = 768  # max dimension (px) for extracted frames
DEFAULT_FRAME_JPEG_QUALITY = 80
DEFAULT_WHISPER_MODEL = "base"
DEFAULT_MAX_PARALLEL = 4
DEFAULT_TIMEOUT = 300.0    # seconds
DEFAULT_MAX_OUTPUT_TOKENS = 1000
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
    fps: float,
    max_frames: int,
    max_side: int,
    jpeg_quality: int,
) -> List[bytes]:
    """Extract JPEG frames from a video file at the requested frames-per-second rate.

    The number of frames extracted is: max(1, min(max_frames, int(duration * fps))).
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
        # Derive frame count from fps, clamped to [1, max_frames]
        num_frames = max(1, min(max_frames, int(duration * fps)))
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


def extract_audio_track(file_path: Path, *, tmpdir: str) -> Optional[Path]:
    """Extract audio from a video file to a WAV file. Returns path or None."""
    out_path = Path(tmpdir) / "audio.wav"
    cmd = [
        "ffmpeg", "-y",
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
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def convert_audio_to_wav(file_path: Path, *, tmpdir: str) -> Optional[Path]:
    """Convert an audio file to 16kHz mono WAV for Whisper. Returns path or None."""
    out_path = Path(tmpdir) / "audio.wav"
    cmd = [
        "ffmpeg", "-y",
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
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
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
                _whisper_model_cache[whisper_model] = whisper.load_model(whisper_model)
            except Exception:
                return None
        model = _whisper_model_cache[whisper_model]

    try:
        result = model.transcribe(str(wav_path), fp16=False, verbose=False)
        text = result.get("text", "").strip()
        return text if text else None
    except Exception:
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
    if frame_data_urls:
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
            parsed = ensure_json_dict(content)
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
    fps: float,
    max_frames: int,
    frame_max_side: int,
    frame_jpeg_quality: int,
    whisper_model: str,
    enable_transcription: bool,
) -> Dict[str, Any]:
    """Process a single AV file: extract frames + transcript, then call the model."""
    suffix = file_path.suffix.lower()
    is_audio_only = suffix in AV_AUDIO_SUFFIXES
    filename = file_path.name

    duration_seconds = probe_duration(file_path)
    frame_data_urls: List[str] = []
    transcript: Optional[str] = None
    prep_start = time.monotonic()

    with tempfile.TemporaryDirectory(prefix="av_ranker_proc_") as tmpdir:
        # --- Extract frames (video only) ---
        if not is_audio_only:
            frame_bytes_list = extract_frames_jpeg(
                file_path,
                fps=fps,
                max_frames=max_frames,
                max_side=frame_max_side,
                jpeg_quality=frame_jpeg_quality,
            )
            for fb in frame_bytes_list:
                frame_data_urls.append(
                    encode_image_bytes_to_data_url(fb, mime="image/jpeg")
                )

        # --- Extract + transcribe audio ---
        if enable_transcription:
            if is_audio_only:
                wav_path = convert_audio_to_wav(file_path, tmpdir=tmpdir)
            else:
                wav_path = extract_audio_track(file_path, tmpdir=tmpdir)

            if wav_path:
                transcript = transcribe_audio(wav_path, whisper_model=whisper_model)

    prep_seconds = time.monotonic() - prep_start

    user_content = build_av_user_message(
        filename,
        frame_data_urls=frame_data_urls,
        transcript=transcript,
        duration_seconds=duration_seconds,
        is_audio_only=is_audio_only,
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
        "frames_extracted": len(frame_data_urls),
        "fps_requested": fps,
        "has_transcript": transcript is not None,
        "transcript_chars": len(transcript) if transcript else 0,
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
        "--fps", type=float, default=DEFAULT_FPS,
        help="Frames per second to extract from video files (default: 1.0).",
    )
    parser.add_argument(
        "--max-frames", type=int, default=DEFAULT_MAX_FRAMES,
        help="Hard cap on total frames extracted per video regardless of fps (default: 120).",
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
    output_jsonl = out_vol_dir / f"av_ranked_{vol_tag}.jsonl"

    checkpoint_path = args.checkpoint or out_vol_dir / ".av_checkpoint"

    # --- Load checkpoint ---
    done_ids: Set[str] = set()
    if args.resume:
        done_ids = load_checkpoint(checkpoint_path)
        if done_ids:
            print(f"[resume] Skipping {len(done_ids)} already-processed file(s).")

    # Also skip IDs already in the output JSONL
    if output_jsonl.exists():
        with output_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    sid = obj.get("source_id", "")
                    if sid:
                        done_ids.add(str(sid).upper())
                except json.JSONDecodeError:
                    pass

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

    print(f"[config] endpoint={args.endpoint}")
    print(f"[config] model={args.model}")
    print(f"[config] fps={args.fps} | max_frames={args.max_frames} | frame_max_side={args.frame_max_side}")
    print(f"[config] transcription={'enabled (' + args.whisper_model + ')' if enable_transcription else 'disabled'}")
    print(f"[config] max_parallel={args.max_parallel} | output={output_jsonl}")
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
                fps=args.fps,
                max_frames=args.max_frames,
                frame_max_side=args.frame_max_side,
                frame_jpeg_quality=args.frame_jpeg_quality,
                whisper_model=args.whisper_model,
                enable_transcription=enable_transcription,
            )
            row = build_output_row(sid, fp, args.volume, result)
            row_json = json.dumps(row, ensure_ascii=False)

            with output_lock:
                with output_jsonl.open("a", encoding="utf-8") as out:
                    out.write(row_json + "\n")
                append_checkpoint(checkpoint_path, sid)

            req_secs = result.get("_request_meta", {}).get("request_seconds", 0.0) or 0.0
            prep_secs = result.get("_av_meta", {}).get("prep_seconds", 0.0) or 0.0
            stats.record_success(req_secs, prep_secs)
            score = result.get("importance_score", "?")
            frames = result.get("_av_meta", {}).get("frames_extracted", 0)
            has_tx = result.get("_av_meta", {}).get("has_transcript", False)
            elapsed = time.monotonic() - t0
            print(
                f"  [ok] {sid}  score={score}  frames={frames}  tx={'y' if has_tx else 'n'}  {elapsed:.1f}s"
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

    print()
    print(f"[done] {stats.summary()}")
    print(f"[done] Output: {output_jsonl}")
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
