#!/usr/bin/env python3
"""Rank Epstein Office document files using a vision-language model.

Converts .docx, .xlsx, .doc, .xls, .csv, .ppt, .pptx files to PDF using
LibreOffice, then renders pages as images and feeds them through the same
VLM pipeline as the PDF ranker (gpt_ranker.py).

Output schema matches the PDF pipeline JSONL format with an added `file_type`
field of "office" and `_office_meta` for original file info.
"""

from __future__ import annotations

import argparse
import concurrent.futures
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

# ---------------------------------------------------------------------------
# Bootstrap sys.path so `ranker` package is importable
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ranker.model_client import (
    ModelRequestError,
    call_model,
    detect_pdf_page_count,
    ensure_json_dict,
)
from ranker.constants import (
    AGENCY_CANONICAL_MAP,
    DEFAULT_JUSTICE_FILES_BASE_URL,
    LEAD_TYPE_CANONICAL_MAP,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OFFICE_SUFFIXES: Set[str] = {
    ".docx", ".xlsx", ".doc", ".xls",
    ".csv", ".ppt", ".pptx", ".ods", ".odt",
}

EFTA_PATTERN = re.compile(r"(EFTA\d{8})", re.IGNORECASE)

DEFAULT_ENDPOINT = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "qwen/qwen3-vl-30b-a3b-thinking"
DEFAULT_SYSTEM_PROMPT_PATH = Path("prompts") / "default_system_prompt.txt"

DEFAULT_MAX_PAGES = 8       # max PDF pages to render per office file
DEFAULT_IMAGE_DPI = 150
DEFAULT_JPEG_QUALITY = 85
DEFAULT_IMAGE_MAX_SIDE = 1024
DEFAULT_MAX_PARALLEL = 4
DEFAULT_TIMEOUT = 300.0
DEFAULT_MAX_OUTPUT_TOKENS = 0
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF = 2.0

# LibreOffice command candidates (PATH names + common macOS .app bundle paths)
_SOFFICE_CANDIDATES = [
    "soffice",
    "libreoffice",
    "/Applications/LibreOffice.app/Contents/MacOS/soffice",
    "/opt/homebrew/bin/soffice",
    "/usr/local/bin/soffice",
]


# ---------------------------------------------------------------------------
# LibreOffice conversion
# ---------------------------------------------------------------------------

def _find_soffice() -> Optional[str]:
    """Return the first available LibreOffice executable name, or None."""
    for name in _SOFFICE_CANDIDATES:
        try:
            result = subprocess.run(
                [name, "--version"],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0:
                return name
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


_SOFFICE_CMD: Optional[str] = None
_SOFFICE_LOCK = threading.Lock()


def get_soffice_cmd() -> str:
    """Cache and return the soffice executable name, raising if not found."""
    global _SOFFICE_CMD
    with _SOFFICE_LOCK:
        if _SOFFICE_CMD is None:
            _SOFFICE_CMD = _find_soffice()
        if _SOFFICE_CMD is None:
            raise RuntimeError(
                "LibreOffice (soffice / libreoffice) is required for office file conversion "
                "but was not found on PATH. Install it with: brew install libreoffice  "
                "or: apt-get install libreoffice"
            )
        return _SOFFICE_CMD


def convert_office_to_pdf(src: Path, outdir: Path) -> Optional[Path]:
    """Convert an Office document to PDF using LibreOffice.

    Returns the path to the converted PDF, or None on failure.
    LibreOffice is single-threaded per-profile; we use a per-call temp
    user profile to allow parallel conversion.
    """
    soffice = get_soffice_cmd()
    # Use a per-process temp user profile so parallel runs don't collide
    user_profile = outdir / ".soffice_profile"
    user_profile.mkdir(parents=True, exist_ok=True)

    cmd = [
        soffice,
        "--headless",
        f"-env:UserInstallation=file://{user_profile.resolve()}",
        "--convert-to", "pdf",
        "--outdir", str(outdir),
        str(src),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=180,
        )
        # LibreOffice outputs the converted filename to stdout
        out_pdf = outdir / (src.stem + ".pdf")
        if out_pdf.exists() and out_pdf.stat().st_size > 0:
            return out_pdf
        # Some LibreOffice versions lowercase the stem
        for candidate in outdir.glob("*.pdf"):
            if candidate.stem.lower() == src.stem.lower():
                return candidate
        if result.returncode != 0:
            stderr = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(f"LibreOffice exited {result.returncode}: {stderr}")
        return None
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"LibreOffice conversion timed out for {src.name}")
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"LibreOffice conversion failed for {src.name}: {exc}") from exc


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_office_files(
    search_roots: List[Path],
    *,
    suffixes: Set[str],
    only_source_ids: Optional[Set[str]] = None,
) -> List[Tuple[str, Path]]:
    """Walk directories and return (source_id, path) pairs for EFTA office files.

    source_id is the EFTA ID extracted from the filename (e.g. "EFTA01234567").
    """
    seen: Set[str] = set()
    results: List[Tuple[str, Path]] = []

    for root in search_roots:
        if not root.is_dir():
            continue
        for file_path in sorted(root.rglob("*")):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in suffixes:
                continue
            if file_path.stat().st_size == 0:
                continue  # skip empty placeholder files
            m = EFTA_PATTERN.search(file_path.stem)
            if not m:
                continue
            source_id = m.group(1).upper()
            if source_id in seen:
                continue
            if only_source_ids is not None and source_id not in only_source_ids:
                continue
            seen.add(source_id)
            results.append((source_id, file_path))

    return results


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _normalize_list(items: Any) -> List[str]:
    if not isinstance(items, list):
        return []
    return [str(x).strip() for x in items if x and str(x).strip()]


def _normalize_agencies(raw: List[str]) -> List[str]:
    canonical_lower = {k.lower(): k for k in AGENCY_CANONICAL_MAP}
    synonym_map: Dict[str, str] = {}
    for canonical, synonyms in AGENCY_CANONICAL_MAP.items():
        for syn in synonyms:
            synonym_map[str(syn).lower()] = canonical

    out: List[str] = []
    seen: Set[str] = set()
    for item in raw:
        normalized = str(item).strip().lower()
        if not normalized:
            continue
        canonical = canonical_lower.get(normalized) or synonym_map.get(normalized) or str(item).strip()
        if canonical and canonical not in seen:
            seen.add(canonical)
            out.append(canonical)
    return out


def _normalize_lead_types(raw: Any) -> List[str]:
    canonical_lower = {k.lower(): k for k in LEAD_TYPE_CANONICAL_MAP}
    synonym_map: Dict[str, str] = {}
    for canonical, synonyms in LEAD_TYPE_CANONICAL_MAP.items():
        for syn in synonyms:
            synonym_map[str(syn).lower()] = canonical

    items = raw if isinstance(raw, list) else []
    out: List[str] = []
    seen: Set[str] = set()
    for item in items:
        normalized = str(item).strip().lower()
        if not normalized:
            continue
        canonical = canonical_lower.get(normalized) or synonym_map.get(normalized) or str(item).strip()
        if canonical and canonical not in seen:
            seen.add(canonical)
            out.append(canonical)
    return out


def derive_justice_url(
    source_id: str,
    original_suffix: str,
    *,
    volume: int,
    base_url: str = DEFAULT_JUSTICE_FILES_BASE_URL,
) -> Optional[str]:
    """Build a DOJ justice.gov URL for an office file using its original extension."""
    if not source_id:
        return None
    m = EFTA_PATTERN.search(source_id)
    if not m:
        return None
    efta_id = m.group(1).upper()
    ext = original_suffix.lstrip(".")
    dataset_number = volume
    return f"{base_url}/DataSet%20{dataset_number}/{efta_id}.{ext}"


def build_output_row(
    source_id: str,
    file_path: Path,
    volume: int,
    result: Dict[str, Any],
    *,
    prep_seconds: float,
    justice_url: Optional[str],
) -> Dict[str, Any]:
    """Construct the output JSONL row matching the PDF pipeline schema."""
    raw_result = result if isinstance(result, dict) else {}

    key_insights = _normalize_list(raw_result.get("key_insights"))
    tags = _normalize_list(raw_result.get("tags"))
    power_mentions = _normalize_list(raw_result.get("power_mentions"))
    agency_involvement = _normalize_agencies(_normalize_list(raw_result.get("agency_involvement")))
    lead_types = _normalize_lead_types(raw_result.get("lead_types"))

    request_meta = raw_result.get("_request_meta") if isinstance(raw_result.get("_request_meta"), dict) else {}
    usage = request_meta.get("usage") if isinstance(request_meta.get("usage"), dict) else {}
    model_cost = request_meta.get("model_cost") if isinstance(request_meta.get("model_cost"), dict) else {}

    # Page count was captured before tmpdir cleanup and stored in the result dict
    page_count = raw_result.get("_office_meta_page_count")

    office_meta = {
        "filename": file_path.name,
        "suffix": file_path.suffix.lower(),
        "file_type": "office",
        "file_size_bytes": file_path.stat().st_size if file_path.exists() else 0,
        "converted_pages": page_count,
        "prep_seconds": round(prep_seconds, 3),
    }

    return {
        "source_id": source_id,
        "filename": file_path.name,
        "volume": volume,
        "headline": str(raw_result.get("headline", "")).strip(),
        "importance_score": raw_result.get("importance_score", 0),
        "reason": str(raw_result.get("reason", "")).strip(),
        "key_insights": key_insights,
        "tags": tags,
        "power_mentions": power_mentions,
        "agency_involvement": agency_involvement,
        "lead_types": lead_types,
        "file_type": "office",
        "source_pdf_url": justice_url,
        "_office_meta": office_meta,
        "_request_meta": {
            "attempt": request_meta.get("attempt", 1),
            "request_seconds": round(request_meta.get("request_seconds", 0.0), 4),
            "model": request_meta.get("model", ""),
            "endpoint": request_meta.get("endpoint", ""),
            "prep_seconds": round(prep_seconds, 3),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "cost_usd": model_cost.get("total_cost_usd"),
        },
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    done: Set[str] = set()
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            sid = line.strip()
            if sid:
                done.add(sid.upper())
    except OSError:
        pass
    return done


def append_checkpoint(path: Path, source_id: str) -> None:
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(source_id.upper() + "\n")
    except OSError:
        pass


def load_only_source_ids(path: Path) -> Optional[Set[str]]:
    if not path.exists():
        return None
    ids: Set[str] = set()
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            sid = line.strip()
            if sid:
                ids.add(sid.upper())
    except OSError:
        return None
    return ids if ids else None


# ---------------------------------------------------------------------------
# Stats tracker
# ---------------------------------------------------------------------------

class Stats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.ok = 0
        self.fail = 0
        self.total_req_secs = 0.0
        self.total_prep_secs = 0.0

    def record_success(self, req_secs: float, prep_secs: float) -> None:
        with self._lock:
            self.ok += 1
            self.total_req_secs += req_secs
            self.total_prep_secs += prep_secs

    def record_failure(self) -> None:
        with self._lock:
            self.fail += 1

    def summary(self) -> str:
        total = self.ok + self.fail
        avg_req = self.total_req_secs / self.ok if self.ok else 0.0
        avg_prep = self.total_prep_secs / self.ok if self.ok else 0.0
        return (
            f"{self.ok}/{total} succeeded | "
            f"avg request {avg_req:.1f}s | avg conversion {avg_prep:.1f}s"
        )


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_office_file(
    file_path: Path,
    *,
    system_prompt: str,
    endpoint: str,
    model: str,
    api_key: Optional[str],
    max_pages: int,
    image_render_dpi: int,
    image_jpeg_quality: int,
    image_max_side: int,
    max_output_tokens: int,
    timeout: float,
    max_retries: int,
    retry_backoff: float,
    http_referer: Optional[str],
    x_title: Optional[str],
    openrouter_provider: Optional[str],
    request_semaphore: Optional[threading.Semaphore],
) -> Tuple[Dict[str, Any], float]:
    """Convert an office file to PDF, render pages, call the VLM, return result.

    Returns (result_dict, prep_seconds). The page count is stored in
    result["_office_meta_page_count"] before the tmpdir is cleaned up.
    """
    prep_start = time.monotonic()

    with tempfile.TemporaryDirectory(prefix="office_ranker_") as tmpdir:
        tmp_path = Path(tmpdir)

        # 1. Convert to PDF
        converted_pdf = convert_office_to_pdf(file_path, tmp_path)
        if converted_pdf is None or not converted_pdf.exists():
            raise RuntimeError(f"LibreOffice produced no PDF for {file_path.name}")

        prep_seconds = time.monotonic() - prep_start

        # 2. Call the VLM via the same call_model used by the PDF pipeline
        if request_semaphore is not None:
            request_semaphore.acquire()
        try:
            result = call_model(
                endpoint=endpoint,
                api_format="auto",
                model=model,
                filename=file_path.name,
                text="",
                input_kind="image",
                image_path=converted_pdf,
                image_max_pages=max_pages,
                image_render_dpi=image_render_dpi,
                system_prompt=system_prompt,
                api_key=api_key,
                timeout=timeout,
                max_retries=max_retries,
                retry_backoff=retry_backoff,
                temperature=0.2,
                max_output_tokens=max_output_tokens,
                reasoning_effort=None,
                image_detail="high",
                pdf_pages_per_image=1,
                image_output_format="jpeg",
                image_jpeg_quality=image_jpeg_quality,
                image_max_side=image_max_side,
                debug_image_dir=None,
                image_start_page=1,
                image_part_index=None,
                image_part_total=None,
                request_semaphore=None,  # we already hold the semaphore
                http_referer=http_referer,
                x_title=x_title,
                openrouter_provider=openrouter_provider,
                openrouter_allow_fallbacks=None,
                config_metadata=None,
            )
        finally:
            if request_semaphore is not None:
                request_semaphore.release()

        # 3. Get page count while PDF is still in the temp dir, before tmpdir cleanup
        page_count = detect_pdf_page_count(converted_pdf)
        result["_office_meta_page_count"] = page_count

        return result, prep_seconds


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank Epstein Office documents (docx/xlsx/ppt/etc) via VLM after PDF conversion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--volume", type=int, required=True, help="DOJ volume number (e.g. 8).")
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path("data/new_data"),
        help="Root directory containing VOL000XX subdirectories.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("contrib/fta"),
        help="Root directory for output JSONL files.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Path to checkpoint file (default: <output-dir>/VOL.../. office_checkpoint).",
    )
    parser.add_argument(
        "--endpoint", type=str,
        default=os.environ.get("OPENROUTER_ENDPOINT", DEFAULT_ENDPOINT),
        help="OpenAI-compatible API endpoint.",
    )
    parser.add_argument(
        "--model", type=str,
        default=os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL),
        help="Model ID.",
    )
    parser.add_argument(
        "--api-key", type=str,
        default=os.environ.get("OPENROUTER_API_KEY"),
        help="API key (or set OPENROUTER_API_KEY env var).",
    )
    parser.add_argument(
        "--openrouter-provider", type=str, default=None,
        help="OpenRouter provider routing hint (e.g. alibaba).",
    )
    parser.add_argument(
        "--http-referer", type=str, default=None,
        help="HTTP-Referer header value for OpenRouter.",
    )
    parser.add_argument(
        "--x-title", type=str, default=None,
        help="X-Title header value for OpenRouter.",
    )
    parser.add_argument(
        "--prompt-file", type=Path,
        default=DEFAULT_SYSTEM_PROMPT_PATH,
        help="Path to system prompt text file.",
    )
    parser.add_argument(
        "--max-pages", type=int, default=DEFAULT_MAX_PAGES,
        help="Max PDF pages to render per office file.",
    )
    parser.add_argument(
        "--image-render-dpi", type=int, default=DEFAULT_IMAGE_DPI,
        help="DPI for PDF page rendering.",
    )
    parser.add_argument(
        "--image-jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY,
        help="JPEG quality (1-95) for rendered pages.",
    )
    parser.add_argument(
        "--image-max-side", type=int, default=DEFAULT_IMAGE_MAX_SIDE,
        help="Max image side in pixels (0=unlimited).",
    )
    parser.add_argument(
        "--max-parallel", type=int, default=DEFAULT_MAX_PARALLEL,
        help="Maximum concurrent model requests (and LibreOffice conversions).",
    )
    parser.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT,
        help="HTTP request timeout in seconds.",
    )
    parser.add_argument(
        "--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS,
        help="Max completion tokens per request (default: 0, no cap).",
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
        help="Limit total files to process (useful for smoke tests).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be processed without calling the model.",
    )
    parser.add_argument(
        "--justice-files-base-url", type=str,
        default=DEFAULT_JUSTICE_FILES_BASE_URL,
        help="Base URL for justice.gov file links.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    if args.max_output_tokens < 0:
        print("[error] --max-output-tokens must be >= 0", file=sys.stderr)
        return 1

    # Resolve API credentials
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    http_referer = args.http_referer or os.environ.get("OPENROUTER_REFERER")
    x_title = args.x_title or os.environ.get("OPENROUTER_TITLE")
    openrouter_provider = args.openrouter_provider or os.environ.get("OFFICE_PROVIDER") or os.environ.get("OPENROUTER_PROVIDER")

    # Load system prompt
    prompt_path = args.prompt_file
    if not prompt_path.exists():
        # Try relative to this script's directory
        prompt_path = Path(__file__).parent / args.prompt_file
    if not prompt_path.exists():
        print(f"[error] System prompt not found: {prompt_path}", file=sys.stderr)
        return 1
    system_prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not system_prompt:
        print(f"[error] System prompt is empty: {prompt_path}", file=sys.stderr)
        return 1

    # Validate LibreOffice is available (unless dry run)
    if not args.dry_run:
        try:
            get_soffice_cmd()
        except RuntimeError as exc:
            print(f"[error] {exc}", file=sys.stderr)
            return 1

    # Determine volume directory
    vol_tag = f"VOL{args.volume:05d}"
    search_root = args.data_dir / vol_tag / "NATIVES"
    if not search_root.is_dir():
        search_root = args.data_dir / vol_tag
    if not search_root.is_dir():
        print(f"[error] Volume directory not found: {search_root}", file=sys.stderr)
        return 1

    # Load filter list
    only_source_ids: Optional[Set[str]] = None
    if args.only_source_ids_file:
        only_source_ids = load_only_source_ids(args.only_source_ids_file)
        if only_source_ids is None:
            print(f"[warn] --only-source-ids-file is empty or unreadable: {args.only_source_ids_file}")

    # Discover office files
    print(f"[discover] Scanning {search_root} for office files...")
    office_files = discover_office_files(
        [search_root],
        suffixes=OFFICE_SUFFIXES,
        only_source_ids=only_source_ids,
    )
    if not office_files:
        print(f"[done] No office files found in {search_root}.")
        return 0

    suffix_counts: Dict[str, int] = {}
    for _, fp in office_files:
        suffix_counts[fp.suffix.lower()] = suffix_counts.get(fp.suffix.lower(), 0) + 1
    suffix_summary = " | ".join(f"{ext}:{n}" for ext, n in sorted(suffix_counts.items()))
    print(f"[discover] Found {len(office_files)} office file(s): {suffix_summary}")

    # Setup output
    out_vol_dir = args.output_dir / vol_tag
    out_vol_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = out_vol_dir / f"office_ranked_{vol_tag}.jsonl"
    checkpoint_path = args.checkpoint or out_vol_dir / ".office_checkpoint"

    # Load already-processed IDs
    done_ids: Set[str] = set()
    if args.resume:
        done_ids = load_checkpoint(checkpoint_path)
        if done_ids:
            print(f"[resume] Loaded {len(done_ids)} already-processed ID(s) from checkpoint.")

    # Also scan existing output JSONL
    if output_jsonl.exists():
        with output_jsonl.open("r", encoding="utf-8") as fh:
            for line in fh:
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

    # Filter
    pending = [(sid, fp) for sid, fp in office_files if sid not in done_ids]
    if done_ids:
        print(f"[filter] {len(office_files) - len(pending)} already done, {len(pending)} pending.")

    if args.max_files:
        pending = pending[: args.max_files]
        print(f"[filter] --max-files {args.max_files}: processing {len(pending)} file(s).")

    if not pending:
        print("[done] Nothing to process.")
        return 0

    if args.dry_run:
        print(f"[dry-run] Would process {len(pending)} file(s):")
        for sid, fp in pending[:30]:
            size_kb = fp.stat().st_size // 1024
            print(f"  {sid}  {fp.suffix.lower()}  {size_kb:,} KB")
        if len(pending) > 30:
            print(f"  ... and {len(pending) - 30} more")
        return 0

    print(f"[config] endpoint={args.endpoint}")
    print(f"[config] model={args.model}")
    print(f"[config] max_pages={args.max_pages} | dpi={args.image_render_dpi} | provider={openrouter_provider or 'auto'}")
    print(f"[config] max_parallel={args.max_parallel} | output={output_jsonl}")
    print()

    semaphore = threading.Semaphore(args.max_parallel)
    stats = Stats()
    output_lock = threading.Lock()

    def process_one(sid: str, fp: Path) -> None:
        t0 = time.monotonic()
        justice_url = derive_justice_url(
            sid, fp.suffix,
            volume=args.volume,
            base_url=args.justice_files_base_url,
        )
        try:
            result, prep_secs = process_office_file(
                fp,
                system_prompt=system_prompt,
                endpoint=args.endpoint,
                model=args.model,
                api_key=api_key,
                max_pages=args.max_pages,
                image_render_dpi=args.image_render_dpi,
                image_jpeg_quality=args.image_jpeg_quality,
                image_max_side=args.image_max_side,
                max_output_tokens=args.max_output_tokens,
                timeout=args.timeout,
                max_retries=args.max_retries,
                retry_backoff=args.retry_backoff,
                http_referer=http_referer,
                x_title=x_title,
                openrouter_provider=openrouter_provider,
                request_semaphore=semaphore,
            )
            row = build_output_row(
                sid, fp, args.volume, result,
                prep_seconds=prep_secs,
                justice_url=justice_url,
            )
            row_json = json.dumps(row, ensure_ascii=False)

            with output_lock:
                with output_jsonl.open("a", encoding="utf-8") as out:
                    out.write(row_json + "\n")
                append_checkpoint(checkpoint_path, sid)

            req_secs = result.get("_request_meta", {}).get("request_seconds", 0.0) or 0.0
            stats.record_success(req_secs, prep_secs)
            score = result.get("importance_score", "?")
            pages = result.get("_office_meta_page_count") or row.get("_office_meta", {}).get("converted_pages") or "?"
            elapsed = time.monotonic() - t0
            print(f"  [ok] {sid}  {fp.suffix.lower()}  score={score}  pages={pages}  {elapsed:.1f}s")

        except Exception as exc:
            stats.record_failure()
            elapsed = time.monotonic() - t0
            print(f"  [fail] {sid}  {fp.suffix.lower()}  {elapsed:.1f}s  {exc}", file=sys.stderr)

    print(
        f"[run] Processing {len(pending)} file(s) with up to {args.max_parallel} concurrent workers..."
    )
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
    # Load .env.openrouter if present
    env_path = Path(__file__).parent / ".env.openrouter"
    if env_path.exists():
        with env_path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip("'\"")
                os.environ.setdefault(key, val)

    sys.exit(main())
