#!/usr/bin/env python3
"""Retroactive metadata enrichment for JSONL chunk files.

Adds three fields to existing ranked JSONL records without requiring an LLM:
  - inappropriate_content (bool)  — blocklist lookup
  - page_count            (int|null) — pdfinfo / cache
  - content_type           ("text"|"image"|"mixed") — pdftotext heuristic

Usage:
  python enrich_metadata.py --volume 1
  python enrich_metadata.py --all-volumes
  python enrich_metadata.py --volume 1 --dry-run
  python enrich_metadata.py --volume 1 --text-threshold 30
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
CONTRIB_FTA = SCRIPT_DIR / "contrib" / "fta"
WORKSPACES = SCRIPT_DIR / "data" / "workspaces"
WORKSPACE_PREFIX = "standardworks_epstein_files_vol"

ENRICHED_FIELDS = {"inappropriate_content", "page_count", "content_type"}
DEFAULT_TEXT_THRESHOLD = 50  # words per page to classify as "text page"


# ---------------------------------------------------------------------------
# Blocklist
# ---------------------------------------------------------------------------

def load_blocklist(volume_number: int) -> Set[str]:
    """Load the provider content-filter blocklist for a volume.

    Returns a set of *base* source IDs (``::part_*`` suffix stripped) so that
    every part of a multi-part PDF is matched when the base document was
    blocked.
    """
    vol_tag = f"{volume_number:05d}"
    blocklist_path = WORKSPACES / f"{WORKSPACE_PREFIX}{vol_tag}" / "metadata" / "provider_content_filter_blocklist.txt"
    if not blocklist_path.is_file():
        return set()
    ids: Set[str] = set()
    try:
        with blocklist_path.open(encoding="utf-8") as fh:
            for line in fh:
                raw = line.strip()
                if not raw:
                    continue
                # Strip ::part_NNNN_pNNNNN-NNNNN suffix to get base source id
                base = re.sub(r"::part_\d+.*$", "", raw)
                ids.add(base)
                ids.add(raw)  # keep exact entry too
    except OSError:
        pass
    return ids


def is_inappropriate(record: Dict[str, Any], blocklist: Set[str]) -> bool:
    """Check whether a record's source file is in the blocklist."""
    if not blocklist:
        return False
    # Try source_id first (newer schema), then filename (VOL00001 schema)
    source_id = record.get("source_id") or record.get("filename", "")
    if not source_id:
        return False
    if source_id in blocklist:
        return True
    # Also check with part suffix stripped
    base = re.sub(r"::part_\d+.*$", "", source_id)
    return base in blocklist


# ---------------------------------------------------------------------------
# Page count cache (reads existing, also writes back)
# ---------------------------------------------------------------------------

class PageCountCache:
    """Reads the existing .pdf_page_counts.json cache from the workspace."""

    def __init__(self, volume_number: int) -> None:
        vol_tag = f"{volume_number:05d}"
        self._path = WORKSPACES / f"{WORKSPACE_PREFIX}{vol_tag}" / "state" / ".pdf_page_counts.json"
        self._entries: Dict[str, Dict[str, Any]] = {}
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self._path.is_file():
            return
        try:
            with self._path.open(encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return
        if isinstance(data, dict) and isinstance(data.get("entries"), dict):
            entries = data["entries"]
        elif isinstance(data, dict):
            entries = data
        else:
            return
        for key, value in entries.items():
            if not isinstance(key, str) or not isinstance(value, dict):
                continue
            pages = value.get("pages")
            if pages is not None and (not isinstance(pages, int) or pages <= 0):
                pages = None
            self._entries[key] = value

    def lookup(self, pdf_path: str) -> Optional[int]:
        """Return cached page count or None if not cached / stale."""
        self._load()
        # Try both the raw path and its absolute posix form
        for key in (pdf_path, str(Path(pdf_path).absolute().as_posix())):
            entry = self._entries.get(key)
            if not entry:
                continue
            # Validate stat freshness if the file exists
            try:
                stat = Path(pdf_path).stat() if key == pdf_path else Path(key).stat()
                if entry.get("size") == stat.st_size and entry.get("mtime_ns") == stat.st_mtime_ns:
                    pages = entry.get("pages")
                    return pages if isinstance(pages, int) and pages > 0 else None
            except OSError:
                # File doesn't exist on disk; trust cache
                pages = entry.get("pages")
                return pages if isinstance(pages, int) and pages > 0 else None
        return None


# ---------------------------------------------------------------------------
# PDF utilities
# ---------------------------------------------------------------------------

def detect_pdf_page_count(pdf_path: str) -> Optional[int]:
    """Get page count via ``pdfinfo``."""
    try:
        result = subprocess.run(
            ["pdfinfo", pdf_path],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except FileNotFoundError:
        return None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    match = re.search(r"^Pages:\s+(\d+)\s*$", result.stdout, flags=re.MULTILINE)
    if not match:
        return None
    try:
        count = int(match.group(1))
    except ValueError:
        return None
    return count if count > 0 else None


def detect_content_type(pdf_path: str, text_threshold: int = DEFAULT_TEXT_THRESHOLD) -> str:
    """Classify a PDF as 'text', 'image', or 'mixed' using pdftotext.

    Runs ``pdftotext -layout PDF -`` and splits on form-feed to get per-page
    text.  Pages with >= *text_threshold* words are "text pages"; the rest are
    "image pages".
    """
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", pdf_path, "-"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return "image"

    if result.returncode != 0 or not result.stdout:
        return "image"

    pages = result.stdout.split("\f")
    # pdftotext may emit a trailing empty string after the last form-feed
    if pages and not pages[-1].strip():
        pages = pages[:-1]
    if not pages:
        return "image"

    text_pages = 0
    image_pages = 0
    for page_text in pages:
        word_count = len(page_text.split())
        if word_count >= text_threshold:
            text_pages += 1
        else:
            image_pages += 1

    if text_pages == 0:
        return "image"
    if image_pages == 0:
        return "text"
    return "mixed"


# ---------------------------------------------------------------------------
# Content type cache (new, mirrors PdfPageCountCache pattern)
# ---------------------------------------------------------------------------

class ContentTypeCache:
    """Persistent cache for PDF content-type classification."""

    def __init__(self, volume_number: int) -> None:
        vol_tag = f"{volume_number:05d}"
        self._path = WORKSPACES / f"{WORKSPACE_PREFIX}{vol_tag}" / "state" / ".pdf_content_types.json"
        self._entries: Dict[str, Dict[str, Any]] = {}
        self._loaded = False
        self._dirty = False
        self._lookups_since_flush = 0

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self._path.is_file():
            return
        try:
            with self._path.open(encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return
        if isinstance(data, dict) and isinstance(data.get("entries"), dict):
            entries = data["entries"]
        elif isinstance(data, dict):
            entries = data
        else:
            return
        for key, value in entries.items():
            if isinstance(key, str) and isinstance(value, dict):
                self._entries[key] = value

    def _cache_key(self, pdf_path: str) -> str:
        try:
            return str(Path(pdf_path).absolute().as_posix())
        except OSError:
            return pdf_path

    def get(self, pdf_path: str, text_threshold: int = DEFAULT_TEXT_THRESHOLD) -> str:
        """Return content type for *pdf_path*, computing and caching if needed."""
        self._load()
        key = self._cache_key(pdf_path)
        entry = self._entries.get(key)
        if entry:
            try:
                stat = Path(pdf_path).stat()
                if entry.get("size") == stat.st_size and entry.get("mtime_ns") == stat.st_mtime_ns:
                    ct = entry.get("content_type", "image")
                    if ct in ("text", "image", "mixed"):
                        return ct
            except OSError:
                # File missing; trust cache
                ct = entry.get("content_type", "image")
                if ct in ("text", "image", "mixed"):
                    return ct

        # Compute
        ct = detect_content_type(pdf_path, text_threshold)

        try:
            stat = Path(pdf_path).stat()
            self._entries[key] = {
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "content_type": ct,
            }
        except OSError:
            self._entries[key] = {"size": 0, "mtime_ns": 0, "content_type": ct}
        self._dirty = True
        self._lookups_since_flush += 1
        if self._lookups_since_flush >= 10:
            self.flush()
        return ct

    def flush(self) -> None:
        if not self._dirty:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "entries": self._entries,
        }
        tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            tmp_path.replace(self._path)
        except OSError:
            pass
        self._dirty = False
        self._lookups_since_flush = 0


# ---------------------------------------------------------------------------
# Record enrichment
# ---------------------------------------------------------------------------

def get_source_path(record: Dict[str, Any]) -> Optional[str]:
    """Extract the physical PDF path from a record's metadata."""
    meta = record.get("metadata") or {}
    original_row = meta.get("original_row") or {}
    sp = original_row.get("source_path")
    if sp:
        return str(sp).strip()
    return None


def record_needs_enrichment(record: Dict[str, Any]) -> bool:
    """Return True if any enrichment field is missing."""
    return not ENRICHED_FIELDS.issubset(record.keys())


def insert_after_filename(record: Dict[str, Any], new_fields: Dict[str, Any]) -> OrderedDict:
    """Return a new ordered dict with *new_fields* inserted right after 'filename'."""
    result: OrderedDict[str, Any] = OrderedDict()
    inserted = False
    for key, value in record.items():
        result[key] = value
        if key == "filename" and not inserted:
            for nk, nv in new_fields.items():
                result[nk] = nv
            inserted = True
    # If 'filename' was never found, append at the end
    if not inserted:
        for nk, nv in new_fields.items():
            result[nk] = nv
    return result


def enrich_record(
    record: Dict[str, Any],
    blocklist: Set[str],
    page_count_cache: PageCountCache,
    content_type_cache: ContentTypeCache,
    text_threshold: int,
    # Per-PDF memoization dicts (shared across records in the same file)
    pdf_page_counts: Dict[str, Optional[int]],
    pdf_content_types: Dict[str, str],
) -> Dict[str, Any]:
    """Add missing enrichment fields to *record*; return updated record."""

    if not record_needs_enrichment(record):
        return record

    source_path = get_source_path(record)

    # --- inappropriate_content ---
    if "inappropriate_content" not in record:
        inappropriate = is_inappropriate(record, blocklist)
    else:
        inappropriate = record["inappropriate_content"]

    # --- page_count ---
    if "page_count" not in record:
        if source_path and source_path not in pdf_page_counts:
            # Try cache first, then pdfinfo
            cached = page_count_cache.lookup(source_path)
            if cached is not None:
                pdf_page_counts[source_path] = cached
            elif Path(source_path).is_file():
                pdf_page_counts[source_path] = detect_pdf_page_count(source_path)
            else:
                pdf_page_counts[source_path] = None
        page_count = pdf_page_counts.get(source_path) if source_path else None
    else:
        page_count = record["page_count"]

    # --- content_type ---
    if "content_type" not in record:
        if source_path and source_path not in pdf_content_types:
            if Path(source_path).is_file():
                pdf_content_types[source_path] = content_type_cache.get(source_path, text_threshold)
            else:
                pdf_content_types[source_path] = "image"
        content_type = pdf_content_types.get(source_path, "image") if source_path else "image"
    else:
        content_type = record["content_type"]

    # Build new fields dict (only those that were missing)
    new_fields: Dict[str, Any] = {}
    if "inappropriate_content" not in record:
        new_fields["inappropriate_content"] = inappropriate
    if "page_count" not in record:
        new_fields["page_count"] = page_count
    if "content_type" not in record:
        new_fields["content_type"] = content_type

    if not new_fields:
        return record

    return insert_after_filename(record, new_fields)


# ---------------------------------------------------------------------------
# File processing
# ---------------------------------------------------------------------------

def process_jsonl_file(
    jsonl_path: Path,
    blocklist: Set[str],
    page_count_cache: PageCountCache,
    content_type_cache: ContentTypeCache,
    text_threshold: int,
    dry_run: bool,
) -> Dict[str, int]:
    """Enrich a single JSONL file. Returns stats dict."""
    stats = {"total": 0, "enriched": 0, "skipped": 0, "errors": 0}
    pdf_page_counts: Dict[str, Optional[int]] = {}
    pdf_content_types: Dict[str, str] = {}

    try:
        with jsonl_path.open(encoding="utf-8") as fh:
            lines = fh.readlines()
    except OSError as exc:
        print(f"  ERROR reading {jsonl_path}: {exc}", file=sys.stderr)
        stats["errors"] += 1
        return stats

    enriched_lines: List[str] = []
    any_changed = False

    for line_no, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if not stripped:
            enriched_lines.append(raw_line)
            continue
        stats["total"] += 1
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError as exc:
            print(f"  WARNING: {jsonl_path}:{line_no}: invalid JSON: {exc}", file=sys.stderr)
            enriched_lines.append(raw_line)
            stats["errors"] += 1
            continue

        if not record_needs_enrichment(record):
            enriched_lines.append(raw_line)
            stats["skipped"] += 1
            continue

        enriched = enrich_record(
            record, blocklist, page_count_cache, content_type_cache,
            text_threshold, pdf_page_counts, pdf_content_types,
        )
        enriched_lines.append(json.dumps(enriched, ensure_ascii=False) + "\n")
        stats["enriched"] += 1
        any_changed = True

    if not any_changed:
        return stats

    if dry_run:
        return stats

    # Atomic write: write to temp, then rename
    tmp_path = jsonl_path.with_suffix(".enrich_tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as fh:
            fh.writelines(enriched_lines)
        tmp_path.replace(jsonl_path)
    except OSError as exc:
        print(f"  ERROR writing {jsonl_path}: {exc}", file=sys.stderr)
        # Clean up temp file if it exists
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        stats["errors"] += 1

    return stats


# ---------------------------------------------------------------------------
# Volume processing
# ---------------------------------------------------------------------------

def volume_dir(volume_number: int) -> Path:
    return CONTRIB_FTA / f"VOL{volume_number:05d}"


def process_volume(
    volume_number: int,
    text_threshold: int,
    dry_run: bool,
) -> None:
    vol_dir = volume_dir(volume_number)
    if not vol_dir.is_dir():
        print(f"Volume directory not found: {vol_dir}", file=sys.stderr)
        return

    jsonl_files = sorted(vol_dir.glob("epstein_ranked_*.jsonl"))
    if not jsonl_files:
        print(f"No JSONL files found in {vol_dir}")
        return

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Processing VOL{volume_number:05d} ({len(jsonl_files)} files)")

    blocklist = load_blocklist(volume_number)
    if blocklist:
        print(f"  Blocklist loaded: {len(blocklist)} entries")
    else:
        print("  No blocklist found (all records → inappropriate_content=False)")

    page_count_cache = PageCountCache(volume_number)
    content_type_cache = ContentTypeCache(volume_number)

    totals = {"total": 0, "enriched": 0, "skipped": 0, "errors": 0}

    for jsonl_path in jsonl_files:
        print(f"  {jsonl_path.name} ... ", end="", flush=True)
        stats = process_jsonl_file(
            jsonl_path, blocklist, page_count_cache, content_type_cache,
            text_threshold, dry_run,
        )
        totals["total"] += stats["total"]
        totals["enriched"] += stats["enriched"]
        totals["skipped"] += stats["skipped"]
        totals["errors"] += stats["errors"]
        print(f"{stats['enriched']} enriched, {stats['skipped']} skipped, {stats['errors']} errors")

    # Final cache flush
    content_type_cache.flush()

    print(f"  Summary: {totals['total']} records, {totals['enriched']} enriched, "
          f"{totals['skipped']} already complete, {totals['errors']} errors")


def discover_volumes() -> List[int]:
    """Return sorted list of volume numbers found under contrib/fta."""
    volumes = []
    if not CONTRIB_FTA.is_dir():
        return volumes
    for entry in CONTRIB_FTA.iterdir():
        if entry.is_dir() and re.match(r"^VOL(\d{5})$", entry.name):
            volumes.append(int(entry.name[3:]))
    return sorted(volumes)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich existing ranked JSONL files with metadata derived without an LLM.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--volume", type=int, metavar="N",
        help="Process a single volume number (e.g. 1 for VOL00001)",
    )
    group.add_argument(
        "--all-volumes", action="store_true",
        help="Process all volumes found under contrib/fta/",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview enrichment without writing any files",
    )
    parser.add_argument(
        "--text-threshold", type=int, default=DEFAULT_TEXT_THRESHOLD,
        metavar="WORDS",
        help=f"Min words per page to classify as 'text' (default: {DEFAULT_TEXT_THRESHOLD})",
    )

    args = parser.parse_args()

    if args.all_volumes:
        volumes = discover_volumes()
        if not volumes:
            print("No volumes found under contrib/fta/", file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(volumes)} volumes: {', '.join(f'VOL{v:05d}' for v in volumes)}")
        for vol in volumes:
            process_volume(vol, args.text_threshold, args.dry_run)
    else:
        process_volume(args.volume, args.text_threshold, args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
