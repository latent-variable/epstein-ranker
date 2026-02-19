#!/usr/bin/env python3
"""Rank Epstein file rows by querying a local GPT server."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import re
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Set, Tuple

import requests

from ranker.cli import (
    apply_config_defaults,
    apply_dataset_workspace_defaults,
    explicit_cli_destinations,
    infer_dataset_tag,
    load_system_prompt,
    parse_args,
    sanitize_dataset_tag,
)
from ranker.constants import (
    AGENCY_CANONICAL_MAP,
    DEFAULT_JUSTICE_FILES_BASE_URL,
    IMAGE_SUFFIXES,
    LEAD_TYPE_CANONICAL_MAP,
    RETRIABLE_HTTP_STATUS_CODES,
    TEXT_SUFFIXES,
)
from ranker.model_client import (
    ModelRequestError,
    UnsupportedEndpointError,
    build_image_analysis_instruction,
    build_request_targets,
    build_text_analysis_input,
    detect_pdf_page_count,
    derive_localhost_fallback_endpoints,
    ensure_json_dict,
    extract_chat_content,
    extract_openai_content,
    image_path_to_data_url,
    infer_api_priority,
    post_request,
    prepare_image_data_urls,
    render_pdf_page_to_data_url,
)
from ranker import model_client as _model_client


try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)

AGENCY_LOOKUP_TERMS: Set[str] = {
    canonical.lower().strip()
    for canonical in AGENCY_CANONICAL_MAP
}
for _canonical, _synonyms in AGENCY_CANONICAL_MAP.items():
    AGENCY_LOOKUP_TERMS.update(syn.lower().strip() for syn in _synonyms if isinstance(syn, str))
AGENCY_LOOKUP_TERMS.update({"ntoc", "ntsc", "new york field office"})

AGENCY_LABEL_HINT_RE = re.compile(
    r"\b("
    r"field office|office|department|agency|committee|bureau|division|unit|"
    r"operations center|operation center|task force|police"
    r")\b",
    flags=re.IGNORECASE,
)

_PDF_GROUNDING_TEXT_CACHE: Dict[str, str] = {}
_PDF_GROUNDING_TEXT_CACHE_ORDER: Deque[str] = deque()
_PDF_GROUNDING_TEXT_CACHE_MAX = 256


def call_model(**kwargs: Any) -> Dict[str, Any]:
    """Compatibility wrapper to keep monkeypatching gpt_ranker.post_request working in tests."""
    original_post_request = _model_client.post_request
    try:
        _model_client.post_request = post_request
        return _model_client.call_model(**kwargs)
    finally:
        _model_client.post_request = original_post_request

def infer_row_input_kind(file_path: Path, processing_mode: str) -> Optional[str]:
    suffix = file_path.suffix.lower()
    if processing_mode == "text":
        return "text" if suffix in TEXT_SUFFIXES else None
    if processing_mode == "image":
        return "image" if suffix in IMAGE_SUFFIXES else None
    if suffix in TEXT_SUFFIXES:
        return "text"
    if suffix in IMAGE_SUFFIXES:
        return "image"
    return None


class PdfPageCountCache:
    """Persistent cache for PDF page counts across runs."""

    def __init__(self, path: Optional[Path]) -> None:
        self.path = path
        self._loaded = False
        self._dirty = False
        self._entries: Dict[str, Dict[str, Any]] = {}

    def _cache_key(self, pdf_path: Path) -> str:
        try:
            return str(pdf_path.resolve().as_posix())
        except OSError:
            return str(pdf_path.as_posix())

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.path or not self.path.exists():
            return
        try:
            with self.path.open(encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return
        if isinstance(data, dict) and isinstance(data.get("entries"), dict):
            entries = data["entries"]
        elif isinstance(data, dict):
            # Backward compatibility: accept a flat dict.
            entries = data
        else:
            return
        for key, value in entries.items():
            if not isinstance(key, str) or not isinstance(value, dict):
                continue
            size = value.get("size")
            mtime_ns = value.get("mtime_ns")
            pages = value.get("pages")
            if not isinstance(size, int) or size < 0:
                continue
            if not isinstance(mtime_ns, int) or mtime_ns < 0:
                continue
            if pages is not None and (not isinstance(pages, int) or pages <= 0):
                pages = None
            self._entries[key] = {
                "size": size,
                "mtime_ns": mtime_ns,
                "pages": pages,
            }

    def get_page_count(self, pdf_path: Path) -> Optional[int]:
        self._load()
        try:
            stat = pdf_path.stat()
        except OSError:
            return detect_pdf_page_count(pdf_path)
        key = self._cache_key(pdf_path)
        entry = self._entries.get(key)
        if entry and entry.get("size") == stat.st_size and entry.get("mtime_ns") == stat.st_mtime_ns:
            pages = entry.get("pages")
            return pages if isinstance(pages, int) and pages > 0 else None

        pages = detect_pdf_page_count(pdf_path)
        self._entries[key] = {
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "pages": pages if isinstance(pages, int) and pages > 0 else None,
        }
        self._dirty = True
        return pages

    def flush(self) -> None:
        if not self.path or not self._dirty:
            return
        payload = {
            "version": 1,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "entries": self._entries,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        tmp_path.replace(self.path)
        self._dirty = False


def iter_rows(
    path: Path,
    *,
    input_glob: str = "*.txt",
    include_text: bool = True,
    processing_mode: str = "auto",
    pdf_part_pages: int = 0,
    pdf_page_count_cache: Optional[PdfPageCountCache] = None,
) -> Iterable[Dict[str, Any]]:
    if path.is_dir():
        pdf_file_index = 0
        for file_path in sorted(path.rglob(input_glob)):
            if not file_path.is_file():
                continue
            input_kind = infer_row_input_kind(file_path, processing_mode)
            if not input_kind:
                continue
            source_pdf_index: Optional[int] = None
            if input_kind == "image" and file_path.suffix.lower() == ".pdf":
                pdf_file_index += 1
                source_pdf_index = pdf_file_index
            rel_filename = file_path.relative_to(path).as_posix()
            if (
                input_kind == "image"
                and file_path.suffix.lower() == ".pdf"
                and pdf_part_pages > 0
            ):
                total_pages = (
                    pdf_page_count_cache.get_page_count(file_path)
                    if pdf_page_count_cache is not None
                    else detect_pdf_page_count(file_path)
                )
                if total_pages is not None and total_pages > pdf_part_pages:
                    total_parts = max(1, math.ceil(total_pages / pdf_part_pages))
                    for part_index in range(1, total_parts + 1):
                        part_start_page = (part_index - 1) * pdf_part_pages + 1
                        part_end_page = min(total_pages, part_start_page + pdf_part_pages - 1)
                        source_id = (
                            f"{rel_filename}::part_{part_index:04d}_p{part_start_page:05d}-{part_end_page:05d}"
                        )
                        yield {
                            "filename": rel_filename,
                            "source_id": source_id,
                            "text": "",
                            "input_kind": input_kind,
                            "source_path": str(file_path),
                            "part_index": part_index,
                            "part_total": total_parts,
                            "part_start_page": part_start_page,
                            "part_end_page": part_end_page,
                            "document_total_pages": total_pages,
                            "source_pdf_index": source_pdf_index,
                            "document_part": (
                                f"part_{part_index:04d}_of_{total_parts:04d}_p{part_start_page:05d}-{part_end_page:05d}"
                            ),
                            "analysis_filename": (
                                f"{rel_filename} "
                                f"(part {part_index}/{total_parts}, pages {part_start_page}-{part_end_page} of {total_pages})"
                            ),
                        }
                    continue
            row: Dict[str, Any] = {
                "filename": rel_filename,
                "source_id": rel_filename,
                "text": "",
                "input_kind": input_kind,
                "source_path": str(file_path),
                "part_index": 1,
                "part_total": 1,
                "part_start_page": 1,
                "part_end_page": None,
                "document_total_pages": None,
                "source_pdf_index": source_pdf_index,
                "document_part": "",
                "analysis_filename": rel_filename,
            }
            if include_text and input_kind == "text":
                row["text"] = file_path.read_text(encoding="utf-8", errors="replace")
            yield row
        return

    if processing_mode == "image":
        raise ValueError("Image processing mode requires --input to be a directory, not CSV.")

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if "text" not in row or "filename" not in row:
                raise ValueError("Input CSV must contain 'filename' and 'text' columns.")
            if include_text:
                yield {
                    "filename": row["filename"],
                    "source_id": row.get("source_id") or row["filename"],
                    "text": row["text"],
                    "input_kind": "text",
                    "source_path": None,
                    "part_index": 1,
                    "part_total": 1,
                    "part_start_page": 1,
                    "part_end_page": None,
                    "document_total_pages": None,
                    "source_pdf_index": None,
                    "document_part": "",
                    "analysis_filename": row["filename"],
                }
            else:
                yield {
                    "filename": row["filename"],
                    "source_id": row.get("source_id") or row["filename"],
                    "text": "",
                    "input_kind": "text",
                    "source_path": None,
                    "part_index": 1,
                    "part_total": 1,
                    "part_start_page": 1,
                    "part_end_page": None,
                    "document_total_pages": None,
                    "source_pdf_index": None,
                    "document_part": "",
                    "analysis_filename": row["filename"],
                }


def row_matches_pdf_range(
    row: Dict[str, Any],
    *,
    start_pdf: int,
    end_pdf: Optional[int],
) -> bool:
    if start_pdf <= 1 and end_pdf is None:
        return True
    pdf_index = row.get("source_pdf_index")
    if not isinstance(pdf_index, int):
        return False
    if pdf_index < start_pdf:
        return False
    if end_pdf is not None and pdf_index > end_pdf:
        return False
    return True


def row_is_after_pdf_range(row: Dict[str, Any], *, end_pdf: Optional[int]) -> bool:
    if end_pdf is None:
        return False
    pdf_index = row.get("source_pdf_index")
    return isinstance(pdf_index, int) and pdf_index > end_pdf


def load_checkpoint(path: Optional[Path]) -> Set[str]:
    completed: Set[str] = set()
    if not path or not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            entry = line.strip()
            if entry:
                completed.add(entry)
    return completed


def row_source_id(row: Dict[str, Any]) -> str:
    source_id = row.get("source_id")
    if isinstance(source_id, str) and source_id.strip():
        return source_id.strip()
    filename = row.get("filename")
    if isinstance(filename, str) and filename.strip():
        return filename.strip()
    return ""


def load_jsonl_filenames(path: Optional[Path]) -> Set[str]:
    completed: Set[str] = set()
    if not path or not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            source_id = record.get("source_id")
            if isinstance(source_id, str) and source_id.strip():
                completed.add(source_id.strip())
                continue
            metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
            if isinstance(metadata, dict):
                meta_source_id = metadata.get("source_id")
                if isinstance(meta_source_id, str) and meta_source_id.strip():
                    completed.add(meta_source_id.strip())
                    continue
            filename = record.get("filename")
            if isinstance(filename, str) and filename.strip():
                completed.add(filename.strip())
    return completed


def extract_source_id_from_record(record: Dict[str, Any]) -> Optional[str]:
    source_id = record.get("source_id")
    if isinstance(source_id, str) and source_id.strip():
        return source_id.strip()
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    if isinstance(metadata, dict):
        meta_source_id = metadata.get("source_id")
        if isinstance(meta_source_id, str) and meta_source_id.strip():
            return meta_source_id.strip()
    filename = record.get("filename")
    if isinstance(filename, str) and filename.strip():
        return filename.strip()
    return None


def load_source_id_filter(path: Optional[Path]) -> Optional[Set[str]]:
    if not path:
        return None
    source_ids: Set[str] = set()
    if not path.exists():
        return source_ids

    raw_text = path.read_text(encoding="utf-8", errors="replace")
    stripped = raw_text.strip()
    if not stripped:
        return source_ids

    # Support JSON arrays/objects in addition to line-oriented text/JSONL.
    if stripped.startswith("[") or stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, str) and item.strip():
                    source_ids.add(item.strip())
                elif isinstance(item, dict):
                    record_source_id = extract_source_id_from_record(item)
                    if record_source_id:
                        source_ids.add(record_source_id)
        elif isinstance(payload, dict):
            listed = payload.get("source_ids")
            if isinstance(listed, list):
                for item in listed:
                    if isinstance(item, str) and item.strip():
                        source_ids.add(item.strip())
            else:
                record_source_id = extract_source_id_from_record(payload)
                if record_source_id:
                    source_ids.add(record_source_id)
        if source_ids:
            return source_ids

    for line in raw_text.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        if candidate.startswith("{"):
            try:
                record = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                record_source_id = extract_source_id_from_record(record)
                if record_source_id:
                    source_ids.add(record_source_id)
            continue
        source_ids.add(candidate)
    return source_ids


def load_chunk_source_ids(chunk_dir: Path) -> Set[str]:
    completed: Set[str] = set()
    if not chunk_dir.exists():
        return completed
    for chunk_path in sorted(chunk_dir.glob("epstein_ranked_*.jsonl")):
        if not chunk_path.is_file():
            continue
        completed |= load_jsonl_filenames(chunk_path)
    return completed


def load_resume_completed_ids(args: argparse.Namespace) -> Set[str]:
    completed_source_ids: Set[str] = set()
    if args.resume:
        checkpoint_ids = load_checkpoint(args.checkpoint)
        if checkpoint_ids:
            completed_source_ids |= checkpoint_ids
        else:
            # Compatibility path for legacy runs that have outputs but no checkpoint.
            output_backed_ids: Set[str] = set()
            output_backed_ids |= load_jsonl_filenames(args.json_output)
            if args.chunk_size > 0:
                output_backed_ids |= load_chunk_source_ids(args.chunk_dir)
            if output_backed_ids:
                completed_source_ids |= output_backed_ids
                print(
                    "Resume checkpoint missing/empty; bootstrapped processed IDs from outputs. "
                    "Future resumes will use the checkpoint for faster startup.",
                    flush=True,
                )
                if args.checkpoint:
                    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    with args.checkpoint.open("a", encoding="utf-8") as handle:
                        for source_id in sorted(output_backed_ids):
                            handle.write(source_id + "\n")
            else:
                print(
                    "Resume checkpoint missing/empty and no prior outputs found; "
                    "starting from the beginning.",
                    flush=True,
                )

    for extra_json in args.known_json:
        completed_source_ids |= load_jsonl_filenames(Path(extra_json))
    return completed_source_ids


def classify_failure_reason(error_text: str) -> str:
    lowered = error_text.lower()
    if "data_inspection_failed" in lowered or "inappropriate content" in lowered:
        return "provider_content_filter"
    if "http 500" in lowered or "internal server error" in lowered:
        return "provider_server_error"
    if "invalid json" in lowered or "empty message" in lowered:
        return "model_output_error"
    if "timeout" in lowered:
        return "timeout"
    return "request_error"


def append_failure_log(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def ensure_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    return [str(value)]


def canonicalize_from_map(value: str, mapping: Dict[str, Set[str]], *, title_case: bool = False, upper_case: bool = False) -> Optional[str]:
    cleaned = " ".join(value.strip().split())
    if not cleaned:
        return None
    lookup = cleaned.lower()
    for canonical, synonyms in mapping.items():
        if lookup in synonyms:
            return canonical if not upper_case else canonical.upper()
    if upper_case:
        return cleaned.upper()
    if title_case:
        return cleaned.title()
    return cleaned


def normalize_lead_types(values: List[str]) -> List[str]:
    normalized: List[str] = []
    seen: Set[str] = set()
    for value in values:
        canonical = canonicalize_from_map(value, LEAD_TYPE_CANONICAL_MAP, title_case=False)
        if not canonical:
            continue
        canonical = canonical.lower()
        if canonical not in seen:
            normalized.append(canonical)
            seen.add(canonical)
    return normalized


def normalize_agencies(values: List[str]) -> List[str]:
    normalized: List[str] = []
    seen: Set[str] = set()
    for value in values:
        canonical = canonicalize_from_map(value, AGENCY_CANONICAL_MAP, upper_case=False)
        if not canonical:
            continue
        canonical = canonical.strip()
        if canonical not in seen:
            normalized.append(canonical)
            seen.add(canonical)
    return normalized


def clean_entity_label(value: str) -> str:
    cleaned = " ".join(value.strip().split())
    if not cleaned:
        return ""
    cleaned = re.sub(r"\s*\(.*?\)\s*", " ", cleaned)
    for delimiter in (" - ", " – ", " — "):
        if delimiter in cleaned:
            cleaned = cleaned.split(delimiter, 1)[0]
    return " ".join(cleaned.split())


def normalize_text_list(values: List[str], *, strip_descriptor: bool = False) -> List[str]:
    normalized: List[str] = []
    seen: Set[str] = set()
    for value in values:
        cleaned = clean_entity_label(value) if strip_descriptor else " ".join(value.strip().split())
        if not cleaned:
            continue
        if cleaned not in seen:
            normalized.append(cleaned)
            seen.add(cleaned)
    return normalized


def _name_tokens(value: str) -> List[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z]+", value)]


def _is_full_name_candidate(value: str) -> bool:
    tokens = _name_tokens(value)
    if len(tokens) < 2:
        return False
    # Canonical full names should not start/end with initials.
    if len(tokens[0]) <= 1 or len(tokens[-1]) <= 1:
        return False
    # Require at least two substantive tokens.
    return sum(1 for token in tokens if len(token) > 1) >= 2


def _is_alias_of_full_name(alias: str, canonical_full: str) -> bool:
    alias_tokens = _name_tokens(alias)
    full_tokens = _name_tokens(canonical_full)
    if not alias_tokens or not full_tokens:
        return False
    if alias_tokens == full_tokens:
        return False

    full_initials = [token[0] for token in full_tokens]

    # Single-token aliases like first name, last name, or first initial.
    if len(alias_tokens) == 1:
        token = alias_tokens[0]
        if token == full_tokens[0] or token == full_tokens[-1]:
            return True
        if len(token) == 1 and token == full_initials[0]:
            return True

    # Initial-only aliases (e.g., "J.E." -> ["j", "e"]).
    if len(alias_tokens) <= len(full_initials) and all(len(token) == 1 for token in alias_tokens):
        if alias_tokens == full_initials[: len(alias_tokens)]:
            return True

    # Last-name forms with initials/partials (e.g., "J. Epstein", "J. E. Epstein").
    if alias_tokens[-1] == full_tokens[-1]:
        prefix = alias_tokens[:-1]
        if not prefix:
            return True
        if all(len(token) == 1 for token in prefix) and prefix[0] == full_initials[0]:
            return True
        if len(alias_tokens) > len(full_tokens):
            return False
        matches_prefix = True
        for idx, token in enumerate(prefix):
            if idx >= len(full_tokens) - 1:
                matches_prefix = False
                break
            full_token = full_tokens[idx]
            if token == full_token:
                continue
            if len(token) == 1 and token == full_token[0]:
                continue
            matches_prefix = False
            break
        if matches_prefix:
            return True

    # First-name partial forms (e.g., "Jeffrey E.").
    if alias_tokens[0] == full_tokens[0] and len(alias_tokens) < len(full_tokens):
        matches = True
        for idx, token in enumerate(alias_tokens):
            if idx >= len(full_tokens):
                matches = False
                break
            full_token = full_tokens[idx]
            if token == full_token:
                continue
            if len(token) == 1 and token == full_token[0]:
                continue
            matches = False
            break
        if matches:
            return True

    return False


def normalize_power_mentions(values: List[str]) -> List[str]:
    normalized = normalize_text_list(values, strip_descriptor=True)
    canonical_full_names = [value for value in normalized if _is_full_name_candidate(value)]
    if not canonical_full_names:
        return normalized
    collapsed: List[str] = []
    for value in normalized:
        if value in canonical_full_names:
            collapsed.append(value)
            continue
        if any(_is_alias_of_full_name(value, full_name) for full_name in canonical_full_names):
            continue
        collapsed.append(value)
    return collapsed


def _looks_like_agency_label(value: str, explicit_agency_terms: Set[str]) -> bool:
    lookup = " ".join(value.strip().lower().split())
    if not lookup:
        return True
    if lookup in explicit_agency_terms or lookup in AGENCY_LOOKUP_TERMS:
        return True
    if AGENCY_LABEL_HINT_RE.search(lookup):
        return True
    return False


def _mention_supported_by_context(mention: str, support_text: str) -> bool:
    lookup = " ".join(mention.strip().lower().split())
    if not lookup:
        return False
    if lookup in support_text:
        return True
    tokens = _name_tokens(lookup)
    if len(tokens) >= 2:
        return tokens[0] in support_text and tokens[-1] in support_text
    return len(tokens) == 1 and len(tokens[0]) >= 4 and tokens[0] in support_text


def filter_power_mentions(
    values: List[str],
    *,
    tags: List[str],
    reason: str,
    key_insights: List[str],
    agency_involvement: List[str],
    source_support_text: str = "",
) -> List[str]:
    normalized = normalize_power_mentions(values)
    explicit_agencies = normalize_agencies(agency_involvement)
    explicit_agency_terms = {value.lower().strip() for value in explicit_agencies if value}
    support_parts = [reason or ""] + key_insights + tags
    support_text = " ".join(part.lower() for part in support_parts if isinstance(part, str))

    filtered: List[str] = []
    seen: Set[str] = set()
    for mention in normalized:
        cleaned = " ".join(mention.strip().split())
        if not cleaned:
            continue
        if _looks_like_agency_label(cleaned, explicit_agency_terms):
            continue
        if len(_name_tokens(cleaned)) < 2:
            continue
        # Recall-first policy: preserve person-like mentions even when source grounding
        # is inconclusive (common for image-only or mixed image/text PDFs).
        # We only hard-filter obvious non-person labels above.
        if cleaned not in seen:
            filtered.append(cleaned)
            seen.add(cleaned)
    return filtered


def load_pdf_grounding_text(source_path: Optional[str]) -> str:
    if not source_path:
        return ""
    normalized_path = str(source_path).strip()
    if not normalized_path.lower().endswith(".pdf"):
        return ""
    cached = _PDF_GROUNDING_TEXT_CACHE.get(normalized_path)
    if cached is not None:
        return cached
    try:
        completed = subprocess.run(
            ["pdftotext", "-layout", normalized_path, "-"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        extracted = ""
    else:
        extracted = " ".join((completed.stdout or "").lower().split()) if completed.returncode == 0 else ""

    _PDF_GROUNDING_TEXT_CACHE[normalized_path] = extracted
    _PDF_GROUNDING_TEXT_CACHE_ORDER.append(normalized_path)
    while len(_PDF_GROUNDING_TEXT_CACHE_ORDER) > _PDF_GROUNDING_TEXT_CACHE_MAX:
        oldest = _PDF_GROUNDING_TEXT_CACHE_ORDER.popleft()
        _PDF_GROUNDING_TEXT_CACHE.pop(oldest, None)
    return extracted


def max_repeated_char_run(text: str) -> int:
    longest = 0
    current = 0
    prev = ""
    for ch in text:
        if ch == prev:
            current += 1
        else:
            current = 1
            prev = ch
        if current > longest:
            longest = current
    return longest


def assess_text_quality(text: str) -> Dict[str, Any]:
    compact = " ".join((text or "").split())
    char_count = len(compact)
    tokens = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", compact)
    word_count = len(tokens)
    token_lengths = [len(token) for token in tokens]
    alpha_count = sum(1 for ch in compact if ch.isalpha())
    unique_word_count = len({token.lower() for token in tokens})
    short_token_count = sum(1 for length in token_lengths if length <= 2)
    long_word_count = sum(1 for length in token_lengths if length >= 4)
    avg_word_length = (
        sum(token_lengths) / word_count if word_count else 0.0
    )
    alpha_ratio = (alpha_count / char_count) if char_count else 0.0
    unique_word_ratio = (unique_word_count / word_count) if word_count else 0.0
    short_token_ratio = (short_token_count / word_count) if word_count else 0.0
    repeated_run = max_repeated_char_run(compact)
    return {
        "char_count": char_count,
        "word_count": word_count,
        "avg_word_length": round(avg_word_length, 4),
        "alpha_ratio": round(alpha_ratio, 4),
        "unique_word_ratio": round(unique_word_ratio, 4),
        "short_token_ratio": round(short_token_ratio, 4),
        "long_word_count": long_word_count,
        "max_repeated_char_run": repeated_run,
    }


def build_skip_reason(quality: Dict[str, Any], args: argparse.Namespace) -> Optional[str]:
    if quality["char_count"] == 0:
        return "empty text"
    if quality["char_count"] < args.min_text_chars:
        return (
            f"too short ({quality['char_count']} chars < minimum {args.min_text_chars})"
        )
    if quality["word_count"] < args.min_text_words:
        return (
            f"too few words ({quality['word_count']} words < minimum {args.min_text_words})"
        )
    if quality["alpha_ratio"] < args.min_alpha_ratio:
        return (
            f"low alphabetic density ({quality['alpha_ratio']:.2f} < minimum {args.min_alpha_ratio:.2f})"
        )
    if quality["word_count"] > 0 and quality["unique_word_ratio"] < args.min_unique_word_ratio:
        return (
            "high repetition/noise "
            f"({quality['unique_word_ratio']:.2f} unique ratio < minimum {args.min_unique_word_ratio:.2f})"
        )
    if quality["word_count"] > 0 and quality["short_token_ratio"] > args.max_short_token_ratio:
        return (
            "too many short/noisy tokens "
            f"({quality['short_token_ratio']:.2f} > max {args.max_short_token_ratio:.2f})"
        )
    if quality["word_count"] > 0 and quality["avg_word_length"] < args.min_avg_word_length:
        return (
            "average token length too low/noisy OCR "
            f"({quality['avg_word_length']:.2f} < minimum {args.min_avg_word_length:.2f})"
        )
    if quality["word_count"] >= args.min_text_words and quality["long_word_count"] < args.min_long_word_count:
        return (
            "too few substantive words "
            f"({quality['long_word_count']} words >=4 chars < minimum {args.min_long_word_count})"
        )
    if quality["max_repeated_char_run"] > args.max_repeated_char_run:
        return (
            "contains long repeated-character sequences "
            f"({quality['max_repeated_char_run']} > max {args.max_repeated_char_run})"
        )
    return None


def build_skipped_model_result(skip_reason: str) -> Dict[str, Any]:
    return {
        "headline": "Skipped low-signal document",
        "importance_score": 0,
        "reason": f"Skipped before model call: {skip_reason}.",
        "key_insights": [],
        "tags": ["skipped"],
        "power_mentions": [],
        "agency_involvement": [],
        "lead_types": [],
    }


def _extract_dataset_number(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    text = str(value)
    dataset_match = re.search(r"DataSet(?:%20|\s)*0*(\d+)", text, flags=re.IGNORECASE)
    if dataset_match:
        try:
            dataset_num = int(dataset_match.group(1))
            return dataset_num if dataset_num > 0 else None
        except ValueError:
            return None
    volume_match = re.search(
        r"(?:^|[/_\\-])vol(?:ume)?[_-]?0*(\d{1,5})(?:$|[/_\\-])",
        text,
        flags=re.IGNORECASE,
    )
    if volume_match:
        try:
            dataset_num = int(volume_match.group(1))
            return dataset_num if dataset_num > 0 else None
        except ValueError:
            return None
    return None


def derive_justice_pdf_url(
    filename: str,
    base_url: str = DEFAULT_JUSTICE_FILES_BASE_URL,
    *,
    source_path: Optional[str] = None,
    dataset_tag: Optional[str] = None,
) -> Optional[str]:
    if not filename and not source_path:
        return None
    efta_source = " ".join(
        value for value in (filename, source_path) if isinstance(value, str) and value.strip()
    )
    efta_match = re.search(r"(EFTA\d{8})", efta_source, flags=re.IGNORECASE)
    if not efta_match:
        return None
    dataset_num = (
        _extract_dataset_number(filename)
        or _extract_dataset_number(source_path)
        or _extract_dataset_number(dataset_tag)
    )
    if not dataset_num:
        return None
    efta_id = efta_match.group(1).upper()
    base = base_url.rstrip("/")
    return f"{base}/DataSet%20{dataset_num}/{efta_id}.pdf"


def derive_local_source_url(
    source_path: Optional[str],
    source_filename: Optional[str],
    *,
    source_files_base_url: Optional[str],
) -> Optional[str]:
    if not source_path and not source_filename:
        return None
    path_obj = Path(source_path) if source_path else None
    normalized = path_obj.as_posix() if path_obj else ""
    if source_files_base_url:
        base = source_files_base_url.rstrip("/")
        if source_filename:
            return f"{base}/{source_filename.lstrip('/')}"
        if path_obj is not None:
            return f"{base}/{path_obj.name}" if path_obj.is_absolute() else f"{base}/{normalized.lstrip('/')}"
    if path_obj is None:
        return None
    parts = path_obj.parts
    if "data" in parts:
        data_index = parts.index("data")
        web_path = Path(*parts[data_index:]).as_posix()
        return f"/{web_path}"
    return None


def count_total_rows(
    path: Path,
    *,
    input_glob: str = "*.txt",
    processing_mode: str = "auto",
    pdf_part_pages: int = 0,
    start_pdf: int = 1,
    end_pdf: Optional[int] = None,
    allowed_source_ids: Optional[Set[str]] = None,
    excluded_source_ids: Optional[Set[str]] = None,
    progress_label: Optional[str] = None,
    pdf_page_count_cache: Optional[PdfPageCountCache] = None,
) -> int:
    """Count total number of source rows for CSV or directory input."""
    last_progress_at = time.monotonic()
    scanned_rows = 0
    selected_rows = 0

    def maybe_log_progress(force: bool = False) -> None:
        nonlocal last_progress_at
        if not progress_label:
            return
        now = time.monotonic()
        if not force and now - last_progress_at < 10:
            return
        print(f"{progress_label}: scanned {scanned_rows:,} row(s)...", flush=True)
        last_progress_at = now

    if path.is_dir():
        for idx, row in enumerate(
            iter_rows(
                path,
                input_glob=input_glob,
                include_text=False,
                processing_mode=processing_mode,
                pdf_part_pages=pdf_part_pages,
                pdf_page_count_cache=pdf_page_count_cache,
            ),
            start=1,
        ):
            scanned_rows = idx
            if row_is_after_pdf_range(row, end_pdf=end_pdf):
                break
            if not row_matches_pdf_range(row, start_pdf=start_pdf, end_pdf=end_pdf):
                continue
            row_id = row_source_id(row)
            if allowed_source_ids is not None and row_id not in allowed_source_ids:
                continue
            if excluded_source_ids is not None and row_id in excluded_source_ids:
                continue
            selected_rows += 1
            maybe_log_progress()
        maybe_log_progress(force=True)
        return selected_rows

    for idx, row in enumerate(
        iter_rows(
            path,
            input_glob=input_glob,
            include_text=False,
            processing_mode=processing_mode,
            pdf_part_pages=pdf_part_pages,
            pdf_page_count_cache=pdf_page_count_cache,
        ),
        start=1,
    ):
        scanned_rows = idx
        row_id = row_source_id(row)
        if allowed_source_ids is not None and row_id not in allowed_source_ids:
            continue
        if excluded_source_ids is not None and row_id in excluded_source_ids:
            continue
        selected_rows += 1
        maybe_log_progress()
    maybe_log_progress(force=True)
    return selected_rows


def calculate_workload(
    path: Path,
    *,
    input_glob: str,
    processing_mode: str,
    pdf_part_pages: int,
    max_rows: Optional[int],
    completed_filenames: Set[str],
    start_row: int,
    end_row: Optional[int],
    start_pdf: int,
    end_pdf: Optional[int],
    allowed_source_ids: Optional[Set[str]] = None,
    excluded_source_ids: Optional[Set[str]] = None,
    progress_label: Optional[str] = None,
    pdf_page_count_cache: Optional[PdfPageCountCache] = None,
) -> Dict[str, int]:
    total = 0
    already_done = 0
    workload = 0
    scanned_rows = 0
    last_progress_at = time.monotonic()

    for idx, row in enumerate(
        iter_rows(
            path,
            input_glob=input_glob,
            include_text=False,
            processing_mode=processing_mode,
            pdf_part_pages=pdf_part_pages,
            pdf_page_count_cache=pdf_page_count_cache,
        ),
        start=1,
    ):
        scanned_rows = idx
        if progress_label:
            now = time.monotonic()
            if now - last_progress_at >= 10:
                print(
                    f"{progress_label}: scanned {scanned_rows:,} row(s) | "
                    f"in-range {total:,} | pending {workload:,} | already-done {already_done:,}",
                    flush=True,
                )
                last_progress_at = now
        if idx < start_row:
            continue
        if end_row is not None and idx > end_row:
            break
        if row_is_after_pdf_range(row, end_pdf=end_pdf):
            break
        if not row_matches_pdf_range(row, start_pdf=start_pdf, end_pdf=end_pdf):
            continue
        row_id = row_source_id(row)
        if allowed_source_ids is not None and row_id not in allowed_source_ids:
            continue
        if excluded_source_ids is not None and row_id in excluded_source_ids:
            continue
        total += 1
        if completed_filenames and row_id in completed_filenames:
            already_done += 1
        else:
            workload += 1
        if max_rows is not None and workload >= max_rows:
            break
    if progress_label:
        print(
            f"{progress_label}: complete after scanning {scanned_rows:,} row(s) | "
            f"in-range {total:,} | pending {workload:,} | already-done {already_done:,}",
            flush=True,
        )
    return {"total": total, "already_done": already_done, "workload": workload}


def estimate_workload_fast(
    path: Path,
    *,
    input_glob: str,
    processing_mode: str,
    pdf_part_pages: int,
    completed_source_ids: Set[str],
    start_row: int,
    end_row: Optional[int],
    start_pdf: int,
    end_pdf: Optional[int],
    max_rows: Optional[int],
    sample_size: int,
    allowed_source_ids: Optional[Set[str]] = None,
    excluded_source_ids: Optional[Set[str]] = None,
    pdf_page_count_cache: Optional[PdfPageCountCache] = None,
) -> Optional[Dict[str, Any]]:
    if allowed_source_ids is not None:
        allowed_ids = set(allowed_source_ids)
        if excluded_source_ids:
            allowed_ids -= excluded_source_ids
        estimated_total_in_range = len(allowed_ids)
        estimated_already_done = sum(
            1 for source_id in allowed_ids if source_id in completed_source_ids
        )
        estimated_workload = max(0, estimated_total_in_range - estimated_already_done)
        if max_rows is not None:
            estimated_workload = min(estimated_workload, max_rows)
        return {
            "estimated_total": estimated_total_in_range,
            "estimated_total_in_range": estimated_total_in_range,
            "estimated_workload": estimated_workload,
            "estimated_already_done": estimated_already_done,
            "total_files": 0,
            "pdf_files": 0,
            "sampled_pdf_count": 0,
            "avg_parts_per_pdf": 1.0,
            "avg_pages_per_pdf": None,
        }

    if not path.is_dir():
        return None

    total_files = 0
    pdf_files = 0
    base_rows = 0
    sampled_pdf_paths: List[Path] = []
    pdf_range_active = start_pdf > 1 or end_pdf is not None
    seen_pdf_files = 0
    for file_path in sorted(path.rglob(input_glob)):
        if not file_path.is_file():
            continue
        input_kind = infer_row_input_kind(file_path, processing_mode)
        if not input_kind:
            continue
        is_pdf = input_kind == "image" and file_path.suffix.lower() == ".pdf"
        if is_pdf:
            seen_pdf_files += 1
            if seen_pdf_files < start_pdf:
                continue
            if end_pdf is not None and seen_pdf_files > end_pdf:
                break
        elif pdf_range_active:
            continue
        total_files += 1
        if (
            is_pdf
            and pdf_part_pages > 0
        ):
            pdf_files += 1
            if len(sampled_pdf_paths) < sample_size:
                sampled_pdf_paths.append(file_path)
        else:
            base_rows += 1

    if total_files == 0:
        return {
            "estimated_total": 0,
            "estimated_total_in_range": 0,
            "estimated_workload": 0,
            "estimated_already_done": 0,
            "total_files": 0,
            "pdf_files": 0,
            "sampled_pdf_count": 0,
            "avg_parts_per_pdf": 1.0,
            "avg_pages_per_pdf": None,
        }

    avg_parts_per_pdf = 1.0
    avg_pages_per_pdf: Optional[float] = None
    sampled_pdf_count = 0
    if pdf_files > 0 and pdf_part_pages > 0 and sampled_pdf_paths:
        sampled_parts: List[int] = []
        sampled_pages: List[int] = []
        for pdf_path in sampled_pdf_paths:
            total_pages = (
                pdf_page_count_cache.get_page_count(pdf_path)
                if pdf_page_count_cache is not None
                else detect_pdf_page_count(pdf_path)
            )
            if total_pages is None or total_pages <= 0:
                continue
            sampled_pdf_count += 1
            sampled_pages.append(total_pages)
            sampled_parts.append(max(1, math.ceil(total_pages / pdf_part_pages)))
        if sampled_parts:
            avg_parts_per_pdf = sum(sampled_parts) / len(sampled_parts)
            avg_pages_per_pdf = sum(sampled_pages) / len(sampled_pages)

    estimated_total = int(round(base_rows + (pdf_files * avg_parts_per_pdf)))
    start_index = max(1, start_row)
    end_index = end_row if end_row is not None else estimated_total
    estimated_total_in_range = max(0, min(end_index, estimated_total) - start_index + 1)
    estimated_already_done = min(
        estimated_total_in_range,
        len(completed_source_ids) if completed_source_ids else 0,
    )
    estimated_workload = max(0, estimated_total_in_range - estimated_already_done)
    if max_rows is not None:
        estimated_workload = min(estimated_workload, max_rows)

    return {
        "estimated_total": estimated_total,
        "estimated_total_in_range": estimated_total_in_range,
        "estimated_workload": estimated_workload,
        "estimated_already_done": estimated_already_done,
        "total_files": total_files,
        "pdf_files": pdf_files,
        "sampled_pdf_count": sampled_pdf_count,
        "avg_parts_per_pdf": avg_parts_per_pdf,
        "avg_pages_per_pdf": avg_pages_per_pdf,
    }


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_eta(
    start_time: float,
    processed: int,
    total: int,
    power_watts: Optional[float] = None,
    electric_rate: Optional[float] = None,
    api_cost_usd_total: Optional[float] = None,
    completion_times: Optional[Deque[float]] = None,
) -> str:
    if total <= 0:
        return ""
    if processed == 0:
        return "(ETA estimating...)"
    elapsed = time.monotonic() - start_time
    if elapsed <= 0:
        return "(ETA --:--:--)"
    rate = processed / elapsed
    if completion_times and len(completion_times) >= 2:
        window_span = completion_times[-1] - completion_times[0]
        if window_span > 0:
            window_rate = (len(completion_times) - 1) / window_span
            if rate > 0:
                # Blend global average with recent throughput to reduce jitter.
                rate = (rate * 0.35) + (window_rate * 0.65)
            else:
                rate = window_rate
    if rate <= 0:
        return "(ETA --:--:--)"
    remaining = max(total - processed, 0)
    eta_seconds = remaining / rate if rate else 0

    # Base ETA message
    eta_msg = f"(ETA {format_duration(eta_seconds)})"

    # Add energy/cost estimates if available
    if power_watts is not None and electric_rate is not None:
        total_estimated_hours = (elapsed + eta_seconds) / 3600
        energy_cost = calculate_energy_cost(power_watts, electric_rate, total_estimated_hours)
        if energy_cost:
            eta_msg += f" | Est. total: {energy_cost['energy_kwh']:.2f} kWh / ${energy_cost['cost_usd']:.2f}"
    if api_cost_usd_total is not None and api_cost_usd_total > 0 and processed > 0:
        total_model_cost = (api_cost_usd_total / processed) * total
        eta_msg += f" | API est. ${total_model_cost:.2f}"

    return eta_msg


def _as_int(value: Any) -> Optional[int]:
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


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            return None
        return numeric
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            numeric = float(stripped)
        except ValueError:
            return None
        if math.isnan(numeric) or math.isinf(numeric):
            return None
        return numeric
    return None


def normalize_token_usage(value: Any) -> Dict[str, int]:
    if not isinstance(value, dict):
        return {}
    prompt_tokens = _as_int(value.get("prompt_tokens"))
    completion_tokens = _as_int(value.get("completion_tokens"))
    total_tokens = _as_int(value.get("total_tokens"))
    cache_read_tokens = _as_int(value.get("cache_read_tokens"))
    cache_write_tokens = _as_int(value.get("cache_write_tokens"))

    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens
    if prompt_tokens is None and total_tokens is not None and completion_tokens is not None:
        prompt_tokens = max(total_tokens - completion_tokens, 0)
    if completion_tokens is None and total_tokens is not None and prompt_tokens is not None:
        completion_tokens = max(total_tokens - prompt_tokens, 0)

    normalized: Dict[str, int] = {}
    if prompt_tokens is not None:
        normalized["prompt_tokens"] = prompt_tokens
    if completion_tokens is not None:
        normalized["completion_tokens"] = completion_tokens
    if total_tokens is not None:
        normalized["total_tokens"] = total_tokens
    if cache_read_tokens is not None:
        normalized["cache_read_tokens"] = cache_read_tokens
    if cache_write_tokens is not None:
        normalized["cache_write_tokens"] = cache_write_tokens
    return normalized


def estimate_token_cost_from_usage(
    usage: Dict[str, int],
    *,
    input_price_per_1m: Optional[float],
    output_price_per_1m: Optional[float],
    cache_read_price_per_1m: Optional[float],
    cache_write_price_per_1m: Optional[float],
) -> Optional[Dict[str, float]]:
    if (
        input_price_per_1m is None
        and output_price_per_1m is None
        and cache_read_price_per_1m is None
        and cache_write_price_per_1m is None
    ):
        return None

    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    cache_read_tokens = usage.get("cache_read_tokens", 0)
    cache_write_tokens = usage.get("cache_write_tokens", 0)
    billable_input_tokens = prompt_tokens
    if cache_read_price_per_1m is not None:
        billable_input_tokens = max(0, billable_input_tokens - cache_read_tokens)
    if cache_write_price_per_1m is not None:
        billable_input_tokens = max(0, billable_input_tokens - cache_write_tokens)

    usage_million = 1_000_000.0
    input_cost_usd = (
        (billable_input_tokens / usage_million) * input_price_per_1m
        if input_price_per_1m is not None
        else 0.0
    )
    output_cost_usd = (
        (completion_tokens / usage_million) * output_price_per_1m
        if output_price_per_1m is not None
        else 0.0
    )
    cache_read_cost_usd = (
        (cache_read_tokens / usage_million) * cache_read_price_per_1m
        if cache_read_price_per_1m is not None
        else 0.0
    )
    cache_write_cost_usd = (
        (cache_write_tokens / usage_million) * cache_write_price_per_1m
        if cache_write_price_per_1m is not None
        else 0.0
    )
    total_cost_usd = (
        input_cost_usd + output_cost_usd + cache_read_cost_usd + cache_write_cost_usd
    )
    return {
        "input_cost_usd": input_cost_usd,
        "output_cost_usd": output_cost_usd,
        "cache_read_cost_usd": cache_read_cost_usd,
        "cache_write_cost_usd": cache_write_cost_usd,
        "total_cost_usd": total_cost_usd,
    }


def attach_request_usage_and_cost(
    result: Dict[str, Any],
    args: argparse.Namespace,
) -> None:
    meta = result.get("_request_meta") if isinstance(result, dict) else None
    if not isinstance(meta, dict):
        return
    usage = normalize_token_usage(meta.get("usage"))
    if usage:
        meta["usage"] = usage

    provider_cost = _as_float(meta.get("provider_reported_cost_usd"))
    estimated = (
        estimate_token_cost_from_usage(
            usage,
            input_price_per_1m=args.input_price_per_1m,
            output_price_per_1m=args.output_price_per_1m,
            cache_read_price_per_1m=args.cache_read_price_per_1m,
            cache_write_price_per_1m=args.cache_write_price_per_1m,
        )
        if usage
        else None
    )
    if provider_cost is None and estimated is None:
        return

    total_cost_usd = (
        provider_cost
        if provider_cost is not None
        else estimated["total_cost_usd"]
        if estimated is not None
        else None
    )
    model_cost: Dict[str, Any] = {
        "source": "provider_reported" if provider_cost is not None else "estimated",
        "provider_reported_cost_usd": provider_cost,
        "estimated_cost_usd": estimated["total_cost_usd"] if estimated is not None else None,
        "total_cost_usd": total_cost_usd,
        "input_price_per_1m": args.input_price_per_1m,
        "output_price_per_1m": args.output_price_per_1m,
        "cache_read_price_per_1m": args.cache_read_price_per_1m,
        "cache_write_price_per_1m": args.cache_write_price_per_1m,
    }
    if estimated is not None:
        model_cost.update(
            {
                "input_cost_usd": estimated["input_cost_usd"],
                "output_cost_usd": estimated["output_cost_usd"],
                "cache_read_cost_usd": estimated["cache_read_cost_usd"],
                "cache_write_cost_usd": estimated["cache_write_cost_usd"],
            }
        )
    meta["model_cost"] = model_cost


def format_request_flow_details(result: Dict[str, Any], *, wall_seconds: Optional[float] = None) -> str:
    meta = result.get("_request_meta") if isinstance(result, dict) else None
    if not isinstance(meta, dict):
        return f"wall={wall_seconds:.2f}s" if wall_seconds is not None else "timing=n/a"
    parts: List[str] = []
    prep = meta.get("prep_seconds")
    req = meta.get("request_seconds")
    blocks = meta.get("image_blocks")
    pages = meta.get("source_page_count")
    total_pages = meta.get("source_total_pages")
    endpoint = meta.get("endpoint")
    attempt = meta.get("attempt")
    local_fallback = meta.get("local_fallback")
    usage = normalize_token_usage(meta.get("usage"))
    model_cost = meta.get("model_cost") if isinstance(meta.get("model_cost"), dict) else {}
    total_cost_usd = _as_float(model_cost.get("total_cost_usd"))
    if total_cost_usd is None:
        total_cost_usd = _as_float(meta.get("provider_reported_cost_usd"))
    if prep is not None:
        parts.append(f"prep={prep:.2f}s")
    if req is not None:
        parts.append(f"request={req:.2f}s")
    if wall_seconds is not None:
        parts.append(f"wall={wall_seconds:.2f}s")
    if isinstance(blocks, int) and blocks > 0:
        parts.append(f"blocks={blocks}")
    if isinstance(pages, int) and pages > 0:
        if isinstance(total_pages, int) and total_pages > 0:
            parts.append(f"pages={pages}/{total_pages}")
        else:
            parts.append(f"pages={pages}")
    if isinstance(attempt, int):
        parts.append(f"attempt={attempt}")
    if isinstance(local_fallback, dict):
        trigger = local_fallback.get("trigger")
        primary_endpoint = local_fallback.get("primary_endpoint")
        if isinstance(trigger, str) and trigger:
            parts.append(f"fallback={trigger}")
        if isinstance(primary_endpoint, str) and primary_endpoint:
            parts.append(f"fallback_from={primary_endpoint}")
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    if prompt_tokens is not None or completion_tokens is not None or total_tokens is not None:
        token_parts = []
        if prompt_tokens is not None:
            token_parts.append(f"in={prompt_tokens}")
        if completion_tokens is not None:
            token_parts.append(f"out={completion_tokens}")
        if total_tokens is not None:
            token_parts.append(f"total={total_tokens}")
        parts.append("tokens(" + ", ".join(token_parts) + ")")
    if total_cost_usd is not None:
        parts.append(f"cost=${total_cost_usd:.6f}")
    if isinstance(endpoint, str) and endpoint:
        parts.append(f"endpoint={endpoint}")
    return " | ".join(parts) if parts else (f"wall={wall_seconds:.2f}s" if wall_seconds is not None else "timing=n/a")


class OutputRouter:
    def __init__(self, args: argparse.Namespace, fieldnames: List[str]):
        self.args = args
        self.fieldnames = fieldnames
        self.chunk_size = max(0, args.chunk_size)
        self.mode = "chunk" if self.chunk_size > 0 else "single"
        self.include_action_items = args.include_action_items
        self.csv_handle = None
        self.json_handle = None
        self.csv_writer = None
        self.current_chunk: Optional[Tuple[int, int]] = None
        self.current_json_path: Optional[Path] = None
        if self.mode == "single":
            self._init_single()
        else:
            self._init_chunk_state()

    def _init_single(self) -> None:
        csv_mode = "a" if self.args.resume and self.args.output.exists() else "w"
        self.args.output.parent.mkdir(parents=True, exist_ok=True)
        self.csv_handle = self.args.output.open(csv_mode, newline="", encoding="utf-8")
        self.csv_writer = csv.DictWriter(self.csv_handle, fieldnames=self.fieldnames)
        if csv_mode == "w":
            self.csv_writer.writeheader()
        json_mode = "a" if self.args.resume and self.args.json_output.exists() else "w"
        self.args.json_output.parent.mkdir(parents=True, exist_ok=True)
        self.json_handle = self.args.json_output.open(json_mode, encoding="utf-8")

    def _init_chunk_state(self) -> None:
        self.chunk_dir: Path = self.args.chunk_dir
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_manifest: Path = self.args.chunk_manifest
        self._loaded_manifest_rows_processed: Optional[int] = None
        self.manifest_entries = self._load_manifest()
        self.manifest_dirty = False
        self.rows_processed_total = (
            self._loaded_manifest_rows_processed
            if isinstance(self._loaded_manifest_rows_processed, int)
            and self._loaded_manifest_rows_processed >= 0
            else 0
        )
        if self._loaded_manifest_rows_processed is None and self.manifest_entries:
            rebuilt_total = 0
            for key, entry in self.manifest_entries.items():
                row_count = entry.get("row_count")
                if not isinstance(row_count, int) or row_count < 0:
                    chunk_path = Path(str(entry.get("json", "")))
                    row_count = self._count_lines(chunk_path)
                    entry["row_count"] = row_count
                    self.manifest_entries[key] = entry
                    self.manifest_dirty = True
                rebuilt_total += row_count
            self.rows_processed_total = rebuilt_total
        self.current_chunk_row_count = 0
        self.total_dataset_rows = None  # Will be set by main()

    def _load_manifest(self) -> Dict[Tuple[int, int], Dict[str, Any]]:
        if not self.chunk_manifest.exists():
            return {}
        try:
            with self.chunk_manifest.open(encoding="utf-8") as handle:
                data = json.load(handle)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

        # Handle both old format (array) and new format (object with chunks)
        metadata = {}
        if isinstance(data, list):
            # Old format: array of chunks
            chunk_list = data
        elif isinstance(data, dict) and "chunks" in data:
            # New format: object with metadata and chunks
            chunk_list = data["chunks"]
            metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        else:
            return {}
        rows_processed = metadata.get("rows_processed")
        if isinstance(rows_processed, int) and rows_processed >= 0:
            self._loaded_manifest_rows_processed = rows_processed

        entries: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for entry in chunk_list:
            key = (entry.get("start_row"), entry.get("end_row"))
            if not isinstance(key[0], int) or not isinstance(key[1], int):
                continue
            entries[key] = entry
        return entries

    def _count_lines(self, path: Path) -> int:
        if not path.exists():
            return 0
        with path.open(encoding="utf-8") as handle:
            return sum(1 for _ in handle)

    def write(self, row_idx: int, csv_row: Dict[str, Any], json_record: Dict[str, Any]) -> None:
        if self.mode == "single":
            self.csv_writer.writerow(csv_row)
            self.csv_handle.flush()
            self.json_handle.write(json.dumps(json_record, ensure_ascii=False) + "\n")
            self.json_handle.flush()
            return

        chunk_bounds = self._chunk_bounds(row_idx)
        self._ensure_chunk(chunk_bounds)
        self.json_handle.write(json.dumps(json_record, ensure_ascii=False) + "\n")
        self.json_handle.flush()
        self.current_chunk_row_count += 1
        self.rows_processed_total += 1

    def _chunk_bounds(self, row_idx: int) -> Tuple[int, int]:
        chunk_start = ((row_idx - 1) // self.chunk_size) * self.chunk_size + 1
        chunk_end = chunk_start + self.chunk_size - 1
        if self.args.end_row is not None:
            chunk_end = min(chunk_end, self.args.end_row)
        return (chunk_start, chunk_end)

    def _ensure_chunk(self, chunk_bounds: Tuple[int, int]) -> None:
        if self.current_chunk == chunk_bounds:
            return
        self._close_chunk()
        self.current_chunk = chunk_bounds
        chunk_start, chunk_end = chunk_bounds
        base = f"epstein_ranked_{chunk_start:05d}_{chunk_end:05d}"
        json_path = self.chunk_dir / f"{base}.jsonl"
        json_exists = json_path.exists()
        if json_exists and not self.args.resume and not self.args.overwrite_output:
            raise FileExistsError(
                f"Chunk JSON {json_path} exists. Use --resume or --overwrite-output."
            )
        json_mode = "a" if self.args.resume and json_exists else "w"
        existing_entry = self.manifest_entries.get(chunk_bounds)
        existing_row_count = (
            existing_entry.get("row_count")
            if isinstance(existing_entry, dict)
            else None
        )
        if not isinstance(existing_row_count, int) or existing_row_count < 0:
            existing_row_count = self._count_lines(json_path) if json_exists else 0
        if json_mode == "w" and json_exists:
            self.rows_processed_total = max(0, self.rows_processed_total - existing_row_count)
            self.manifest_entries.pop(chunk_bounds, None)
            existing_row_count = 0
        self.current_chunk_row_count = existing_row_count
        self.json_handle = json_path.open(json_mode, encoding="utf-8")
        self.current_json_path = json_path

    def _close_chunk(self) -> None:
        if self.json_handle:
            self.json_handle.close()
            self.json_handle = None
        if self.current_chunk and self.current_json_path:
            chunk_start, chunk_end = self.current_chunk
            entry = {
                "start_row": chunk_start,
                "end_row": chunk_end,
                "json": str(self.current_json_path.as_posix()),
                "row_count": self.current_chunk_row_count,
            }
            self.manifest_entries[self.current_chunk] = entry
            self.manifest_dirty = True
            # Write manifest immediately after each chunk closes
            self._write_manifest()
        self.current_chunk = None
        self.current_json_path = None
        self.current_chunk_row_count = 0

    def _write_manifest(self) -> None:
        """Write the manifest file to disk."""
        if not self.manifest_dirty:
            return
        entries = sorted(self.manifest_entries.values(), key=lambda e: e["start_row"])

        # Build manifest with metadata
        manifest = {
            "metadata": {
                "total_dataset_rows": self.total_dataset_rows if self.total_dataset_rows else "unknown",
                "rows_processed": max(0, int(self.rows_processed_total)),
                "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "chunks": entries
        }

        self.chunk_manifest.parent.mkdir(parents=True, exist_ok=True)
        with self.chunk_manifest.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        self.manifest_dirty = False

    def close(self) -> None:
        if self.mode == "single":
            if self.csv_handle:
                self.csv_handle.close()
            if self.json_handle:
                self.json_handle.close()
            return
        self._close_chunk()
        self._write_manifest()

def build_config_metadata(args: argparse.Namespace, prompt_source: str) -> Dict[str, Any]:
    """Build metadata dictionary from config for inclusion in requests and outputs."""
    local_fallback_enabled = (
        args.local_fallback_on_content_filter
        or args.local_fallback_on_model_output_error
    )
    metadata = {
        "endpoint": args.endpoint,
        "api_format": args.api_format,
        "processing_mode": args.processing_mode,
        "model": args.model,
        "justice_files_base_url": args.justice_files_base_url,
        "source_files_base_url": args.source_files_base_url,
        "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
        "http_referer": args.http_referer,
        "x_title": args.x_title,
        "max_parallel_requests": args.max_parallel_requests,
        "max_retries": args.max_retries,
        "retry_backoff": args.retry_backoff,
        "image_max_pages": args.image_max_pages,
        "pdf_pages_per_image": args.pdf_pages_per_image,
        "pdf_part_pages": args.pdf_part_pages,
        "start_pdf": args.start_pdf,
        "end_pdf": args.end_pdf,
        "only_source_ids_file": str(args.only_source_ids_file) if args.only_source_ids_file else None,
        "exclude_source_ids_file": str(args.exclude_source_ids_file) if args.exclude_source_ids_file else None,
        "failure_log": str(args.failure_log) if args.failure_log else None,
        "content_filter_blocklist": (
            str(args.content_filter_blocklist) if args.content_filter_blocklist else None
        ),
        "local_fallback_on_content_filter": args.local_fallback_on_content_filter,
        "local_fallback_on_model_output_error": args.local_fallback_on_model_output_error,
        "local_fallback_endpoint": (
            args.local_fallback_endpoint if local_fallback_enabled else None
        ),
        "local_fallback_model": (
            args.local_fallback_model if local_fallback_enabled else None
        ),
        "local_fallback_api_format": (
            args.local_fallback_api_format if local_fallback_enabled else None
        ),
        "image_render_dpi": args.image_render_dpi,
        "image_detail": args.image_detail,
        "image_output_format": args.image_output_format,
        "image_jpeg_quality": args.image_jpeg_quality,
        "image_max_side": args.image_max_side,
        "debug_image_dir": str(args.debug_image_dir) if args.debug_image_dir else None,
        "flow_logs": args.flow_logs,
        "prompt_source": prompt_source,
    }
    if args.dataset_tag:
        metadata["dataset_tag"] = args.dataset_tag
    if args.dataset_workspace_root:
        metadata["dataset_workspace_root"] = str(args.dataset_workspace_root)
    if args.dataset_source_label:
        metadata["dataset_source_label"] = args.dataset_source_label
    if args.dataset_source_url:
        metadata["dataset_source_url"] = args.dataset_source_url
    if args.reasoning_effort is not None:
        metadata["reasoning_effort"] = args.reasoning_effort
    if args.openrouter_provider:
        metadata["openrouter_provider"] = args.openrouter_provider
        metadata["openrouter_allow_fallbacks"] = not args.openrouter_no_fallbacks
    metadata["skip_low_quality"] = args.skip_low_quality
    metadata["skip_thresholds"] = {
        "min_text_chars": args.min_text_chars,
        "min_text_words": args.min_text_words,
        "min_alpha_ratio": args.min_alpha_ratio,
        "min_unique_word_ratio": args.min_unique_word_ratio,
        "max_short_token_ratio": args.max_short_token_ratio,
        "min_avg_word_length": args.min_avg_word_length,
        "min_long_word_count": args.min_long_word_count,
        "max_repeated_char_run": args.max_repeated_char_run,
    }
    if args.api_key:
        metadata["api_key_used"] = True
    if args.power_watts is not None:
        metadata["power_watts"] = args.power_watts
    if args.electric_rate is not None:
        metadata["electric_rate"] = args.electric_rate
    if args.input_price_per_1m is not None:
        metadata["input_price_per_1m"] = args.input_price_per_1m
    if args.output_price_per_1m is not None:
        metadata["output_price_per_1m"] = args.output_price_per_1m
    if args.cache_read_price_per_1m is not None:
        metadata["cache_read_price_per_1m"] = args.cache_read_price_per_1m
    if args.cache_write_price_per_1m is not None:
        metadata["cache_write_price_per_1m"] = args.cache_write_price_per_1m
    return metadata


def load_dataset_profile(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Dataset metadata file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Dataset metadata file must contain a JSON object: {path}")
    return data


def write_run_metadata(
    *,
    args: argparse.Namespace,
    prompt_source: str,
    config_metadata: Dict[str, Any],
    workload_stats: Dict[str, int],
    total_dataset_rows: Optional[int],
    dataset_profile: Optional[Dict[str, Any]],
) -> None:
    if not args.run_metadata_file:
        return

    payload: Dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": {
            "tag": args.dataset_tag,
            "source_label": args.dataset_source_label,
            "source_url": args.dataset_source_url,
        },
        "input": {
            "path": str(args.input),
            "is_directory": args.input.is_dir(),
            "input_glob": args.input_glob if args.input.is_dir() else None,
            "start_row": getattr(args, "start_row", 1),
            "end_row": getattr(args, "end_row", None),
            "start_pdf": getattr(args, "start_pdf", 1) if args.input.is_dir() else None,
            "end_pdf": getattr(args, "end_pdf", None) if args.input.is_dir() else None,
            "only_source_ids_file": str(args.only_source_ids_file) if getattr(args, "only_source_ids_file", None) else None,
            "exclude_source_ids_file": str(args.exclude_source_ids_file) if getattr(args, "exclude_source_ids_file", None) else None,
            "total_rows": total_dataset_rows if total_dataset_rows is not None else "unknown",
        },
        "workload": workload_stats,
        "outputs": {
            "output_csv": str(args.output),
            "output_jsonl": str(args.json_output),
            "checkpoint": str(args.checkpoint),
            "chunk_size": args.chunk_size,
            "chunk_dir": str(args.chunk_dir),
            "chunk_manifest": str(args.chunk_manifest),
            "failure_log": str(args.failure_log) if getattr(args, "failure_log", None) else None,
            "content_filter_blocklist": (
                str(args.content_filter_blocklist)
                if getattr(args, "content_filter_blocklist", None)
                else None
            ),
        },
        "config": config_metadata,
        "prompt_source": prompt_source,
    }
    if dataset_profile is not None:
        payload["dataset_profile"] = dataset_profile

    run_metadata_file = Path(args.run_metadata_file)
    run_metadata_file.parent.mkdir(parents=True, exist_ok=True)
    with run_metadata_file.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def calculate_energy_cost(
    power_watts: Optional[float],
    electric_rate: Optional[float],
    hours: float,
) -> Optional[Dict[str, float]]:
    """Calculate energy consumption and cost."""
    if power_watts is None or electric_rate is None or hours <= 0:
        return None
    energy_kwh = (power_watts * hours) / 1000.0
    cost = energy_kwh * electric_rate
    return {"energy_kwh": energy_kwh, "cost_usd": cost}


def format_cost_summary(
    power_watts: Optional[float],
    electric_rate: Optional[float],
    elapsed_hours: float,
    override_hours: Optional[float],
) -> Optional[str]:
    if power_watts is None or electric_rate is None:
        return None
    hours = override_hours if override_hours is not None else elapsed_hours
    if hours <= 0:
        return None
    energy_kwh = (power_watts * hours) / 1000.0
    cost = energy_kwh * electric_rate
    return f"Energy ≈ {energy_kwh:.2f} kWh | Est. cost ≈ ${cost:.2f} (rate ${electric_rate}/kWh)"


def list_models(endpoint: str, api_key: Optional[str], timeout: float) -> None:
    """Print available model IDs from the endpoint."""
    url = f"{endpoint.rstrip('/')}/models"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    models = data.get("data", [])
    if not models and isinstance(data.get("models"), list):
        models = [
            {
                "id": entry.get("id") or entry.get("key"),
                "created": entry.get("created"),
            }
            for entry in data["models"]
            if isinstance(entry, dict)
        ]
    if not models:
        print("No models reported by endpoint.")
        return
    print("Available models:")
    for entry in models:
        model_id = entry.get("id")
        created = entry.get("created")
        extra = f" (created {created})" if created else ""
        print(f" - {model_id}{extra}")


def rebuild_manifest(chunk_dir: Path, manifest_path: Path) -> None:
    """Scan chunk directory and rebuild the manifest file."""
    import re

    # Pattern to match chunk files: epstein_ranked_XXXXX_YYYYY.jsonl
    pattern = re.compile(r"epstein_ranked_(\d{5})_(\d{5})\.jsonl")

    chunks = []
    if not chunk_dir.exists():
        print(f"Chunk directory not found: {chunk_dir}")
        return

    for file_path in sorted(chunk_dir.glob("epstein_ranked_*.jsonl")):
        match = pattern.match(file_path.name)
        if not match:
            print(f"Skipping non-matching file: {file_path.name}")
            continue

        start_row = int(match.group(1))
        end_row = int(match.group(2))

        # Use relative path: chunk_dir/filename
        relative_path = chunk_dir / file_path.name

        chunks.append({
            "start_row": start_row,
            "end_row": end_row,
            "json": str(relative_path.as_posix()),
        })

    if not chunks:
        print(f"No chunk files found in {chunk_dir}")
        return

    # Sort by start_row
    chunks.sort(key=lambda c: c["start_row"])

    # Count total rows processed
    total_processed = 0
    for chunk in chunks:
        chunk_path = Path(chunk["json"])
        if chunk_path.exists():
            with chunk_path.open(encoding="utf-8") as f:
                row_count = sum(1 for _ in f)
        else:
            row_count = 0
        chunk["row_count"] = row_count
        total_processed += row_count

    # Try to get total dataset rows from the source CSV
    # Look for common CSV filenames in data/ directory
    csv_candidates = [
        Path("data/EPS_FILES_20K_NOV2026.csv"),
        Path("data") / "epstein_files.csv",
    ]
    total_dataset_rows = None
    for csv_path in csv_candidates:
        if csv_path.exists():
            print(f"Counting total rows in {csv_path}...")
            try:
                total_dataset_rows = count_total_rows(csv_path)
                print(f"Found {total_dataset_rows:,} total rows in dataset")
                break
            except Exception as e:
                print(f"Error counting rows: {e}")

    # Fall back to highest end_row if CSV not found
    if total_dataset_rows is None:
        total_dataset_rows = max((c["end_row"] for c in chunks), default=0) if chunks else "unknown"
        print(f"Source CSV not found, using highest chunk end_row: {total_dataset_rows}")

    # Build manifest with metadata
    manifest = {
        "metadata": {
            "total_dataset_rows": total_dataset_rows,
            "rows_processed": total_processed,
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "chunks": chunks
    }

    # Write manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Rebuilt manifest with {len(chunks)} chunks:")
    for chunk in chunks:
        print(f"  - Rows {chunk['start_row']:,}–{chunk['end_row']:,}: {chunk['json']}")
    if isinstance(total_dataset_rows, int):
        print(f"Total: {total_processed:,} rows processed out of {total_dataset_rows:,} dataset rows")
    else:
        print(f"Total: {total_processed:,} rows processed")
    print(f"Manifest written to: {manifest_path}")


def build_output_records(
    *,
    idx: int,
    source_row: Dict[str, Any],
    result: Dict[str, Any],
    args: argparse.Namespace,
    config_metadata: Dict[str, Any],
    quality: Dict[str, Any],
    processing_status: str,
    skip_reason: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    source_id = row_source_id(source_row)
    part_index = source_row.get("part_index")
    part_total = source_row.get("part_total")
    part_start_page = source_row.get("part_start_page")
    part_end_page = source_row.get("part_end_page")
    document_total_pages = source_row.get("document_total_pages")
    document_part = source_row.get("document_part") or ""
    justice_source_url = derive_justice_pdf_url(
        source_row.get("filename", ""),
        base_url=args.justice_files_base_url,
        source_path=source_row.get("source_path"),
        dataset_tag=getattr(args, "dataset_tag", None),
    )
    local_source_url = derive_local_source_url(
        source_row.get("source_path"),
        source_row.get("filename"),
        source_files_base_url=getattr(args, "source_files_base_url", None),
    )
    source_url = justice_source_url or local_source_url
    key_insights = normalize_text_list(ensure_list(result.get("key_insights")))
    tags = normalize_text_list(ensure_list(result.get("tags")))
    agency_involvement = normalize_agencies(
        normalize_text_list(ensure_list(result.get("agency_involvement")), strip_descriptor=True)
    )
    lead_types = normalize_lead_types(ensure_list(result.get("lead_types")))
    raw_power_mentions = ensure_list(result.get("power_mentions"))
    power_mentions = filter_power_mentions(
        raw_power_mentions,
        tags=tags,
        reason=str(result.get("reason", "")),
        key_insights=key_insights,
        agency_involvement=agency_involvement,
    )
    raw_power_mentions_normalized = normalize_power_mentions(raw_power_mentions)
    unresolved_person_mentions = [
        mention
        for mention in raw_power_mentions_normalized
        if mention not in power_mentions
        and not _looks_like_agency_label(mention, {agency.lower() for agency in agency_involvement})
        and len(_name_tokens(mention)) >= 2
    ]
    if unresolved_person_mentions:
        source_support_text = load_pdf_grounding_text(source_row.get("source_path"))
        if source_support_text:
            power_mentions = filter_power_mentions(
                raw_power_mentions,
                tags=tags,
                reason=str(result.get("reason", "")),
                key_insights=key_insights,
                agency_involvement=agency_involvement,
                source_support_text=source_support_text,
            )
    action_items = (
        normalize_text_list(ensure_list(result.get("action_items")))
        if args.include_action_items
        else []
    )
    request_meta = result.get("_request_meta") if isinstance(result.get("_request_meta"), dict) else {}
    token_usage = normalize_token_usage(request_meta.get("usage"))
    model_cost = (
        request_meta.get("model_cost")
        if isinstance(request_meta.get("model_cost"), dict)
        else {}
    )
    model_cost_total = _as_float(model_cost.get("total_cost_usd"))
    model_cost_source = model_cost.get("source") if isinstance(model_cost.get("source"), str) else ""

    csv_row = {
        "source_id": source_id,
        "filename": source_row["filename"],
        "document_part": document_part,
        "part_index": part_index if part_index is not None else "",
        "part_total": part_total if part_total is not None else "",
        "part_start_page": part_start_page if part_start_page is not None else "",
        "part_end_page": part_end_page if part_end_page is not None else "",
        "document_total_pages": document_total_pages if document_total_pages is not None else "",
        "source_row_index": idx,
        "source_pdf_url": source_url or "",
        "processing_status": processing_status,
        "skip_reason": skip_reason,
        "headline": result.get("headline", ""),
        "importance_score": result.get("importance_score", ""),
        "reason": result.get("reason", ""),
        "key_insights": "; ".join(key_insights),
        "tags": "; ".join(tags),
        "power_mentions": "; ".join(power_mentions),
        "agency_involvement": "; ".join(agency_involvement),
        "lead_types": "; ".join(lead_types),
        "api_prompt_tokens": token_usage.get("prompt_tokens", ""),
        "api_completion_tokens": token_usage.get("completion_tokens", ""),
        "api_total_tokens": token_usage.get("total_tokens", ""),
        "api_cache_read_tokens": token_usage.get("cache_read_tokens", ""),
        "api_cache_write_tokens": token_usage.get("cache_write_tokens", ""),
        "api_cost_usd": f"{model_cost_total:.8f}" if model_cost_total is not None else "",
        "api_cost_source": model_cost_source,
    }
    if args.include_action_items:
        csv_row["action_items"] = "; ".join(action_items)

    json_record: Dict[str, Any] = {
        "source_id": source_id,
        "filename": source_row["filename"],
        "document_part": document_part,
        "part_index": part_index,
        "part_total": part_total,
        "part_start_page": part_start_page,
        "part_end_page": part_end_page,
        "document_total_pages": document_total_pages,
        "source_pdf_url": source_url,
        "headline": result.get("headline", ""),
        "importance_score": result.get("importance_score", ""),
        "reason": result.get("reason", ""),
        "key_insights": key_insights,
        "tags": tags,
        "power_mentions": power_mentions,
        "agency_involvement": agency_involvement,
        "lead_types": lead_types,
        "api_prompt_tokens": token_usage.get("prompt_tokens"),
        "api_completion_tokens": token_usage.get("completion_tokens"),
        "api_total_tokens": token_usage.get("total_tokens"),
        "api_cache_read_tokens": token_usage.get("cache_read_tokens"),
        "api_cache_write_tokens": token_usage.get("cache_write_tokens"),
        "api_cost_usd": model_cost_total,
        "api_cost_source": model_cost_source or None,
        "metadata": {
            "source_row_index": idx,
            "source_id": source_id,
            "document_part": document_part,
            "part_index": part_index,
            "part_total": part_total,
            "part_start_page": part_start_page,
            "part_end_page": part_end_page,
            "document_total_pages": document_total_pages,
            "original_row": source_row,
            "config": config_metadata,
            "source_pdf_url": source_url,
            "justice_source_url": justice_source_url,
            "local_source_url": local_source_url,
            "processing_status": processing_status,
            "skip_reason": skip_reason,
            "text_quality": quality,
            "request_meta": request_meta,
            "token_usage": token_usage,
            "model_cost": model_cost,
        },
    }
    if args.include_action_items:
        json_record["action_items"] = action_items
    return csv_row, json_record


def resolve_processing_mode(args: argparse.Namespace) -> str:
    if args.processing_mode in {"text", "image"}:
        return args.processing_mode
    if args.input.is_file():
        return "text"

    for row in iter_rows(
        args.input,
        input_glob=args.input_glob,
        include_text=False,
        processing_mode="auto",
        pdf_part_pages=getattr(args, "pdf_part_pages", 0),
    ):
        return row.get("input_kind", "text")

    # Helpful fallback for image-only corpora when users keep the default *.txt glob.
    if args.input_glob == "*.txt":
        for pattern in ("*.pdf", "*.png", "*.jpg", "*.jpeg", "*.webp", "*.tif", "*.tiff", "*.bmp"):
            first_match = next(args.input.rglob(pattern), None)
            if first_match:
                args.input_glob = pattern
                return "image"
    return "text"


def main() -> None:
    args = parse_args()
    if args.list_models:
        list_models(args.endpoint, args.api_key, args.timeout)
        return

    if args.rebuild_manifest:
        rebuild_manifest(args.chunk_dir, args.chunk_manifest)
        return

    # Load system prompt from file or use inline/default
    system_prompt, prompt_source = load_system_prompt(args)

    if not args.input.exists():
        sys.exit(f"Input path not found: {args.input}")
    active_processing_mode = resolve_processing_mode(args)
    args.processing_mode = active_processing_mode
    if args.source_files_base_url is not None and not str(args.source_files_base_url).strip():
        args.source_files_base_url = None
    if args.dataset_metadata_file and not args.dataset_metadata_file.exists():
        sys.exit(f"Dataset metadata file not found: {args.dataset_metadata_file}")
    if args.max_parallel_requests < 1:
        sys.exit("--max-parallel-requests must be >= 1")
    if args.image_prefetch < 0:
        sys.exit("--image-prefetch must be >= 0")
    if args.parallel_scheduling not in {"auto", "window", "batch"}:
        sys.exit("--parallel-scheduling must be one of: auto, window, batch")
    if args.max_retries < 1:
        sys.exit("--max-retries must be >= 1")
    if args.retry_backoff < 0:
        sys.exit("--retry-backoff must be >= 0")
    if args.min_text_chars < 0:
        sys.exit("--min-text-chars must be >= 0")
    if args.min_text_words < 0:
        sys.exit("--min-text-words must be >= 0")
    if args.min_alpha_ratio < 0:
        sys.exit("--min-alpha-ratio must be >= 0")
    if args.min_unique_word_ratio < 0:
        sys.exit("--min-unique-word-ratio must be >= 0")
    if args.max_short_token_ratio < 0 or args.max_short_token_ratio > 1:
        sys.exit("--max-short-token-ratio must be between 0 and 1")
    if args.min_avg_word_length < 0:
        sys.exit("--min-avg-word-length must be >= 0")
    if args.min_long_word_count < 0:
        sys.exit("--min-long-word-count must be >= 0")
    if args.max_repeated_char_run < 1:
        sys.exit("--max-repeated-char-run must be >= 1")
    if args.image_max_pages < 1:
        sys.exit("--image-max-pages must be >= 1")
    if args.pdf_pages_per_image < 1:
        sys.exit("--pdf-pages-per-image must be >= 1")
    if args.pdf_part_pages < 0:
        sys.exit("--pdf-part-pages must be >= 0")
    if args.image_render_dpi < 72:
        sys.exit("--image-render-dpi must be >= 72")
    if args.image_jpeg_quality < 1 or args.image_jpeg_quality > 95:
        sys.exit("--image-jpeg-quality must be between 1 and 95")
    if args.image_max_side < 0:
        sys.exit("--image-max-side must be >= 0")
    if args.workload_estimate_sample_size < 1:
        sys.exit("--workload-estimate-sample-size must be >= 1")
    if args.max_output_tokens < 1:
        sys.exit("--max-output-tokens must be >= 1")
    for price_name in (
        "input_price_per_1m",
        "output_price_per_1m",
        "cache_read_price_per_1m",
        "cache_write_price_per_1m",
    ):
        price_value = getattr(args, price_name, None)
        if price_value is not None and price_value < 0:
            sys.exit(f"--{price_name.replace('_', '-')} must be >= 0")
    if active_processing_mode == "image" and args.api_format == "chat":
        sys.exit(
            "Image mode is not supported with --api-format chat. "
            "Use --api-format openai or --api-format auto."
        )
    local_fallback_enabled = (
        args.local_fallback_on_content_filter
        or args.local_fallback_on_model_output_error
    )
    if (
        local_fallback_enabled
        and active_processing_mode == "image"
        and args.local_fallback_api_format == "chat"
    ):
        sys.exit(
            "Image mode local fallback cannot use --local-fallback-api-format chat. "
            "Use openai or auto."
        )
    if "openrouter.ai" in args.endpoint and not args.api_key:
        sys.exit(
            "OpenRouter endpoint detected but --api-key is missing. "
            "Provide --api-key or set it via your config."
        )
    if args.openrouter_provider and "openrouter.ai" not in args.endpoint:
        print(
            "WARNING: --openrouter-provider is set but endpoint is not openrouter.ai; "
            "provider routing preferences will be ignored.",
            flush=True,
        )
    if args.openrouter_no_fallbacks and not args.openrouter_provider:
        print(
            "WARNING: --openrouter-no-fallbacks was set without --openrouter-provider; "
            "no provider preference will be applied.",
            flush=True,
        )
    if args.start_row < 1:
        sys.exit("--start-row must be >= 1")
    if args.end_row is not None and args.end_row < args.start_row:
        sys.exit("--end-row must be greater than or equal to --start-row")
    if args.start_pdf < 1:
        sys.exit("--start-pdf must be >= 1")
    if args.end_pdf is not None and args.end_pdf < args.start_pdf:
        sys.exit("--end-pdf must be greater than or equal to --start-pdf")
    if args.start_pdf != 1 or args.end_pdf is not None:
        if not args.input.is_dir():
            sys.exit("--start-pdf/--end-pdf require --input to be a directory")
        if active_processing_mode != "image":
            sys.exit("--start-pdf/--end-pdf require image mode (set --processing-mode image)")
    if args.only_source_ids_file and not args.only_source_ids_file.exists():
        sys.exit(f"--only-source-ids-file not found: {args.only_source_ids_file}")
    if args.chunk_size <= 0:
        if (
            args.output.exists()
            and not args.resume
            and not args.overwrite_output
        ):
            sys.exit(
                f"Output file {args.output} already exists. "
                "Use --resume to append/skip or --overwrite-output to replace."
            )
        if (
            args.json_output.exists()
            and not args.resume
            and not args.overwrite_output
        ):
            sys.exit(
                f"JSON output file {args.json_output} already exists. "
                "Use --resume to append/skip or --overwrite-output to replace."
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    if args.checkpoint:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if args.debug_image_dir:
        args.debug_image_dir.mkdir(parents=True, exist_ok=True)

    selected_source_ids = load_source_id_filter(args.only_source_ids_file)
    if selected_source_ids is not None:
        print(
            f"Source-id filter enabled: {len(selected_source_ids):,} id(s) loaded from "
            f"{args.only_source_ids_file}",
            flush=True,
        )
        if not selected_source_ids:
            print("Source-id filter is empty. No rows to process.", flush=True)
            return
    excluded_source_ids = load_source_id_filter(args.exclude_source_ids_file)
    if excluded_source_ids is not None:
        print(
            f"Source-id exclusion enabled: {len(excluded_source_ids):,} id(s) loaded from "
            f"{args.exclude_source_ids_file}",
            flush=True,
        )
    if args.content_filter_blocklist:
        args.content_filter_blocklist.parent.mkdir(parents=True, exist_ok=True)
        print(f"Content-filter blocklist enabled: {args.content_filter_blocklist}", flush=True)
    if args.failure_log:
        args.failure_log.parent.mkdir(parents=True, exist_ok=True)
        print(f"Failure log enabled: {args.failure_log}", flush=True)

    if active_processing_mode == "image":
        if args.skip_low_quality:
            print("Image mode detected: skipping text-quality prefilter checks.", flush=True)
        print(
            f"Processing mode: image | glob: {args.input_glob} | "
            f"PDF pages per file: {args.image_max_pages} | "
            f"pages per image: {args.pdf_pages_per_image} | "
            f"split part pages: {args.pdf_part_pages if args.pdf_part_pages > 0 else 'disabled'} | "
            f"render DPI: {args.image_render_dpi} | "
            f"format: {args.image_output_format} | "
            f"jpeg quality: {args.image_jpeg_quality} | "
            f"max side: {args.image_max_side if args.image_max_side > 0 else 'disabled'}",
            flush=True,
        )
        if args.pdf_part_pages > 0:
            print(
                "PDF split mode enabled: each part is processed as an independent record "
                f"using windows of up to {args.pdf_part_pages} page(s).",
                flush=True,
            )
        if (
            "vl" not in args.model.lower()
            and "vision" not in args.model.lower()
            and "4.6v" not in args.model.lower()
        ):
            print(
                f"WARNING: image mode is enabled but model '{args.model}' may be non-vision.",
                flush=True,
            )
        if args.debug_image_dir:
            print(
                f"Image debug artifacts enabled: {args.debug_image_dir}",
                flush=True,
            )
        if args.start_pdf != 1 or args.end_pdf is not None:
            print(
                "PDF file range: "
                f"{args.start_pdf}-{args.end_pdf if args.end_pdf is not None else 'end'} "
                "(1-based, pre-split).",
                flush=True,
            )
    else:
        print(f"Processing mode: text | glob: {args.input_glob}", flush=True)
    print(
        f"Endpoint: {args.endpoint} | API format: {args.api_format} | Model: {args.model}",
        flush=True,
    )
    if "openrouter.ai" in args.endpoint:
        if args.openrouter_provider:
            print(
                "OpenRouter provider routing: "
                f"provider={args.openrouter_provider} | "
                f"fallbacks={'disabled' if args.openrouter_no_fallbacks else 'enabled'}",
                flush=True,
            )
        if args.input_price_per_1m is not None or args.output_price_per_1m is not None:
            print(
                "API token pricing enabled for cost tracking: "
                f"input={args.input_price_per_1m}, output={args.output_price_per_1m}, "
                f"cache_read={args.cache_read_price_per_1m}, cache_write={args.cache_write_price_per_1m}",
                flush=True,
            )
        else:
            print(
                "OpenRouter endpoint detected with no explicit token pricing; "
                "cost tracking will use provider-reported cost only (if present).",
                flush=True,
            )
    if local_fallback_enabled:
        trigger_labels: List[str] = []
        if args.local_fallback_on_content_filter:
            trigger_labels.append("provider_content_filter")
        if args.local_fallback_on_model_output_error:
            trigger_labels.append("model_output_error")
        print(
            "Local fallback enabled: "
            f"triggers={','.join(trigger_labels)} | "
            f"endpoint={args.local_fallback_endpoint} | "
            f"model={args.local_fallback_model} | "
            f"api_format={args.local_fallback_api_format}",
            flush=True,
        )

    fieldnames = [
        "source_id",
        "filename",
        "document_part",
        "part_index",
        "part_total",
        "part_start_page",
        "part_end_page",
        "document_total_pages",
        "source_row_index",
        "source_pdf_url",
        "processing_status",
        "skip_reason",
        "headline",
        "importance_score",
        "reason",
        "key_insights",
        "tags",
        "power_mentions",
        "agency_involvement",
        "lead_types",
        "api_prompt_tokens",
        "api_completion_tokens",
        "api_total_tokens",
        "api_cache_read_tokens",
        "api_cache_write_tokens",
        "api_cost_usd",
        "api_cost_source",
    ]
    if args.include_action_items:
        fieldnames.append("action_items")

    processed = 0
    completed_source_ids: Set[str] = load_resume_completed_ids(args)
    if completed_source_ids:
        print(f"Skipping {len(completed_source_ids)} pre-processed source ids.")
    pdf_page_count_cache: Optional[PdfPageCountCache] = None
    if active_processing_mode == "image" and args.input.is_dir():
        cache_path = args.checkpoint.parent / ".pdf_page_counts.json"
        pdf_page_count_cache = PdfPageCountCache(cache_path)
        if args.pdf_part_pages > 0:
            print(f"PDF page-count cache: {cache_path}", flush=True)

    workload_scan_mode = args.workload_scan
    if workload_scan_mode == "auto":
        if active_processing_mode == "image" and args.input.is_dir() and args.pdf_part_pages > 0:
            workload_scan_mode = "defer"
        else:
            workload_scan_mode = "full"

    workload_scanned = workload_scan_mode == "full"
    workload_stats: Dict[str, int] = {"total": 0, "already_done": 0, "workload": 0}
    total_candidates = 0
    already_done = 0
    target_total: Optional[int] = None
    target_total_is_estimate = False

    if workload_scanned:
        workload_progress_label: Optional[str] = None
        if active_processing_mode == "image" and args.input.is_dir():
            workload_progress_label = (
                "Planning workload (PDF split/page detection can take time on large volumes)"
            )
            print("Planning workload before request submission...", flush=True)

        workload_stats = calculate_workload(
            args.input,
            input_glob=args.input_glob,
            processing_mode=active_processing_mode,
            pdf_part_pages=args.pdf_part_pages,
            max_rows=args.max_rows,
            completed_filenames=completed_source_ids if args.resume else set(),
            start_row=args.start_row,
            end_row=args.end_row,
            start_pdf=args.start_pdf,
            end_pdf=args.end_pdf,
            allowed_source_ids=selected_source_ids,
            excluded_source_ids=excluded_source_ids,
            progress_label=workload_progress_label,
            pdf_page_count_cache=pdf_page_count_cache,
        )
        total_candidates = workload_stats["total"]
        already_done = workload_stats["already_done"] if args.resume else 0
        target_total = workload_stats["workload"]
        if target_total <= 0:
            print("No new rows to process. Exiting.")
            return
    else:
        print(
            "Workload scan deferred: starting processing immediately without upfront total-row counting.",
            flush=True,
        )
        print("Estimating workload quickly for ETA guidance...", flush=True)
        estimated_stats = estimate_workload_fast(
            args.input,
            input_glob=args.input_glob,
            processing_mode=active_processing_mode,
            pdf_part_pages=args.pdf_part_pages,
            completed_source_ids=completed_source_ids if args.resume else set(),
            start_row=args.start_row,
            end_row=args.end_row,
            start_pdf=args.start_pdf,
            end_pdf=args.end_pdf,
            allowed_source_ids=selected_source_ids,
            excluded_source_ids=excluded_source_ids,
            max_rows=args.max_rows,
            sample_size=args.workload_estimate_sample_size,
            pdf_page_count_cache=pdf_page_count_cache,
        )
        if estimated_stats:
            target_total = int(estimated_stats["estimated_workload"])
            target_total_is_estimate = True
            total_candidates = int(estimated_stats["estimated_total_in_range"])
            already_done = int(estimated_stats["estimated_already_done"])
            workload_stats = {
                "total": total_candidates,
                "already_done": already_done,
                "workload": target_total,
            }
            avg_pages = estimated_stats.get("avg_pages_per_pdf")
            avg_pages_msg = (
                f"{avg_pages:.1f} avg pages/PDF"
                if isinstance(avg_pages, (int, float))
                else "unknown avg pages/PDF"
            )
            print(
                "Estimated workload: "
                f"~{target_total:,} pending row(s) within requested range "
                f"(~{total_candidates:,} total, ~{already_done:,} already done) | "
                f"files={estimated_stats['total_files']:,}, pdfs={estimated_stats['pdf_files']:,}, "
                f"sampled={estimated_stats['sampled_pdf_count']:,}, "
                f"~{estimated_stats['avg_parts_per_pdf']:.2f} parts/PDF ({avg_pages_msg}).",
                flush=True,
            )

    row_range_desc = f"rows {args.start_row}-{args.end_row if args.end_row else 'end'}"
    if args.start_pdf != 1 or args.end_pdf is not None:
        range_desc = (
            f"{row_range_desc} | "
            f"PDF files {args.start_pdf}-{args.end_pdf if args.end_pdf is not None else 'end'}"
        )
    else:
        range_desc = row_range_desc
    if selected_source_ids is not None:
        range_desc = f"{range_desc} | source-id filter={len(selected_source_ids):,}"
    if excluded_source_ids is not None:
        range_desc = f"{range_desc} | source-id exclusions={len(excluded_source_ids):,}"
    if target_total is not None:
        if target_total_is_estimate:
            print(
                f"Processing ~{target_total} estimated new rows within {range_desc} "
                f"(~{already_done} already completed, ~{total_candidates} total considered).",
                flush=True,
            )
        else:
            print(
                f"Processing {target_total} new rows within {range_desc} "
                f"(skipping {already_done} already completed, total considered {total_candidates})."
            )
    else:
        print(
            f"Processing {range_desc} with deferred workload totals "
            "(ETA will use completed-row rate only).",
            flush=True,
        )

    output_router = OutputRouter(args, fieldnames)
    total_dataset_rows: Optional[int] = None
    can_reuse_workload_total = workload_scanned and (
        args.start_row == 1
        and args.end_row is None
        and args.start_pdf == 1
        and args.end_pdf is None
        and selected_source_ids is None
        and excluded_source_ids is None
        and args.max_rows is None
    )

    # Count total rows in dataset for manifest metadata
    if output_router.mode == "chunk" and workload_scan_mode == "defer":
        print(
            "Dataset total-row counting deferred in chunk mode; manifest metadata will use 'unknown'.",
            flush=True,
        )
    elif output_router.mode == "chunk" and can_reuse_workload_total:
        output_router.total_dataset_rows = total_candidates
        total_dataset_rows = output_router.total_dataset_rows
        print(
            f"Total dataset: {output_router.total_dataset_rows:,} rows (reused from workload scan)",
            flush=True,
        )
    elif output_router.mode == "chunk":
        print("Counting total rows in dataset...")
        output_router.total_dataset_rows = count_total_rows(
            args.input,
            input_glob=args.input_glob,
            processing_mode=active_processing_mode,
            pdf_part_pages=args.pdf_part_pages,
            start_pdf=args.start_pdf,
            end_pdf=args.end_pdf,
            allowed_source_ids=selected_source_ids,
            excluded_source_ids=excluded_source_ids,
            progress_label=(
                "Counting total rows (cached page counts)"
                if active_processing_mode == "image" and args.input.is_dir()
                else None
            ),
            pdf_page_count_cache=pdf_page_count_cache,
        )
        total_dataset_rows = output_router.total_dataset_rows
        print(f"Total dataset: {output_router.total_dataset_rows:,} rows")
    elif workload_scan_mode == "defer" and args.run_metadata_file:
        total_dataset_rows = None
    elif can_reuse_workload_total and args.run_metadata_file:
        total_dataset_rows = total_candidates
    elif args.run_metadata_file:
        # Single-file mode still records total row count in run metadata.
        if active_processing_mode == "image" and args.input.is_dir():
            print("Counting total rows in dataset for run metadata...", flush=True)
        total_dataset_rows = count_total_rows(
            args.input,
            input_glob=args.input_glob,
            processing_mode=active_processing_mode,
            pdf_part_pages=args.pdf_part_pages,
            start_pdf=args.start_pdf,
            end_pdf=args.end_pdf,
            allowed_source_ids=selected_source_ids,
            excluded_source_ids=excluded_source_ids,
            progress_label=(
                "Counting total rows for metadata (cached page counts)"
                if active_processing_mode == "image" and args.input.is_dir()
                else None
            ),
            pdf_page_count_cache=pdf_page_count_cache,
        )

    checkpoint_handle = (
        args.checkpoint.open("a", encoding="utf-8") if args.checkpoint else None
    )

    # Build config metadata to include in requests and outputs
    config_metadata = build_config_metadata(args, prompt_source)
    try:
        dataset_profile = load_dataset_profile(args.dataset_metadata_file)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        sys.exit(str(exc))
    write_run_metadata(
        args=args,
        prompt_source=prompt_source,
        config_metadata=config_metadata,
        workload_stats=workload_stats,
        total_dataset_rows=total_dataset_rows,
        dataset_profile=dataset_profile,
    )

    start_time = time.monotonic()
    parallel_scheduling = args.parallel_scheduling
    if parallel_scheduling == "auto":
        parallel_scheduling = "batch" if active_processing_mode == "image" else "window"
    if parallel_scheduling == "batch":
        max_in_flight_tasks = args.max_parallel_requests
    else:
        max_in_flight_tasks = args.max_parallel_requests + (
            args.image_prefetch if active_processing_mode == "image" else 0
        )
    print(
        "Parallel scheduling: "
        f"{parallel_scheduling} | request limit: {args.max_parallel_requests}"
        + (f" | task queue: {max_in_flight_tasks}" if max_in_flight_tasks != args.max_parallel_requests else "")
    )
    if parallel_scheduling == "batch" and args.image_prefetch > 0:
        print("Image prefetch ignored in batch scheduling mode.")
    elif active_processing_mode == "image" and args.image_prefetch > 0:
        print(
            "Image prefetch enabled: "
            f"{args.max_parallel_requests} concurrent request(s) + "
            f"{args.image_prefetch} queued prefetch task(s)."
        )
    if args.flow_logs:
        print("Flow logs enabled: emitting queue/prep/request timing per row.", flush=True)
    emit_order: Deque[int] = deque()
    pending_results: Dict[int, Dict[str, Any]] = {}
    in_flight: Dict[concurrent.futures.Future[Dict[str, Any]], Dict[str, Any]] = {}
    scheduled = 0
    skipped = 0
    failed = 0
    failure_log_records = 0
    failure_source_ids: Set[str] = set()
    content_filter_blocked_written = 0
    content_filter_blocked_seen: Set[str] = set()
    local_fallback_processed = 0
    local_fallback_processed_by_category: Dict[str, int] = {}
    model_scored = 0
    api_rows_with_usage = 0
    api_prompt_tokens_total = 0
    api_completion_tokens_total = 0
    api_total_tokens_total = 0
    api_cache_read_tokens_total = 0
    api_cache_write_tokens_total = 0
    api_cost_usd_total = 0.0
    api_rows_with_cost = 0
    eta_completed = 0
    eta_completion_times: Deque[float] = deque(maxlen=max(32, args.max_parallel_requests * 20))
    interrupted = False
    request_semaphore: Optional[threading.Semaphore] = None
    if max_in_flight_tasks > 1:
        request_semaphore = threading.BoundedSemaphore(args.max_parallel_requests)
    executor = (
        concurrent.futures.ThreadPoolExecutor(max_workers=max_in_flight_tasks)
        if max_in_flight_tasks > 1
        else None
    )

    def record_completion_for_eta(result: Optional[Dict[str, Any]] = None) -> None:
        nonlocal eta_completed
        nonlocal api_rows_with_usage, api_prompt_tokens_total, api_completion_tokens_total
        nonlocal api_total_tokens_total, api_cache_read_tokens_total, api_cache_write_tokens_total
        nonlocal api_cost_usd_total, api_rows_with_cost
        eta_completed += 1
        eta_completion_times.append(time.monotonic())
        if not isinstance(result, dict):
            return
        request_meta = (
            result.get("_request_meta")
            if isinstance(result.get("_request_meta"), dict)
            else {}
        )
        usage = normalize_token_usage(request_meta.get("usage"))
        if usage:
            api_rows_with_usage += 1
            api_prompt_tokens_total += usage.get("prompt_tokens", 0)
            api_completion_tokens_total += usage.get("completion_tokens", 0)
            api_total_tokens_total += usage.get("total_tokens", 0)
            api_cache_read_tokens_total += usage.get("cache_read_tokens", 0)
            api_cache_write_tokens_total += usage.get("cache_write_tokens", 0)
        model_cost = request_meta.get("model_cost")
        if isinstance(model_cost, dict):
            row_cost_usd = _as_float(model_cost.get("total_cost_usd"))
            if row_cost_usd is not None:
                api_rows_with_cost += 1
                api_cost_usd_total += row_cost_usd

    def mark_content_filter_blocked(source_id_value: str) -> None:
        nonlocal content_filter_blocked_written
        if not args.content_filter_blocklist:
            return
        source_id_normalized = str(source_id_value).strip()
        if not source_id_normalized or source_id_normalized in content_filter_blocked_seen:
            return
        args.content_filter_blocklist.parent.mkdir(parents=True, exist_ok=True)
        with args.content_filter_blocklist.open("a", encoding="utf-8") as handle:
            handle.write(source_id_normalized + "\n")
        content_filter_blocked_seen.add(source_id_normalized)
        content_filter_blocked_written += 1

    def execute_request_with_local_fallback(
        *,
        request_kwargs: Dict[str, Any],
        row_idx: int,
        source_id: str,
    ) -> Dict[str, Any]:
        try:
            return call_model(**request_kwargs)
        except Exception as primary_exc:  # noqa: BLE001
            primary_error = str(primary_exc)
            primary_category = classify_failure_reason(primary_error)
            trigger_enabled = (
                (
                    args.local_fallback_on_content_filter
                    and primary_category == "provider_content_filter"
                )
                or (
                    args.local_fallback_on_model_output_error
                    and primary_category == "model_output_error"
                )
            )
            if not trigger_enabled:
                raise
            if args.flow_logs:
                print(
                    f"[Row {row_idx}] [fallback] {source_id} | "
                    f"trigger={primary_category} | endpoint={args.local_fallback_endpoint}",
                    flush=True,
                )
            fallback_kwargs = dict(request_kwargs)
            fallback_kwargs.update(
                {
                    "endpoint": args.local_fallback_endpoint,
                    "api_format": args.local_fallback_api_format,
                    "model": args.local_fallback_model,
                    "api_key": None,
                    "http_referer": None,
                    "x_title": None,
                    "openrouter_provider": None,
                    "openrouter_allow_fallbacks": None,
                }
            )
            try:
                fallback_result = call_model(**fallback_kwargs)
            except Exception as fallback_exc:  # noqa: BLE001
                raise RuntimeError(
                    f"{primary_error} | Local fallback failed via "
                    f"{args.local_fallback_endpoint}: {fallback_exc}"
                ) from fallback_exc
            request_meta = (
                fallback_result.get("_request_meta")
                if isinstance(fallback_result.get("_request_meta"), dict)
                else {}
            )
            request_meta["local_fallback"] = {
                "trigger": primary_category,
                "primary_endpoint": request_kwargs.get("endpoint"),
                "primary_model": request_kwargs.get("model"),
                "primary_error": primary_error,
            }
            fallback_result["_request_meta"] = request_meta
            return fallback_result

    def flush_ready() -> None:
        nonlocal processed, skipped, failed, model_scored, failure_log_records, failure_source_ids
        nonlocal content_filter_blocked_written, content_filter_blocked_seen
        nonlocal local_fallback_processed, local_fallback_processed_by_category
        while emit_order and emit_order[0] in pending_results:
            row_idx = emit_order.popleft()
            outcome = pending_results.pop(row_idx)
            if outcome["type"] == "error":
                failed += 1
                row_ref = row_source_id(outcome["row"]) or outcome["row"].get("filename", "")
                error_text = str(outcome["error"])
                error_category = outcome.get("error_category") or classify_failure_reason(error_text)
                failure_source_ids.add(row_ref)
                if (
                    args.content_filter_blocklist
                    and error_category == "provider_content_filter"
                ):
                    mark_content_filter_blocked(row_ref)
                if args.failure_log:
                    append_failure_log(
                        args.failure_log,
                        {
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "source_row_index": row_idx,
                            "source_id": row_ref,
                            "filename": outcome["row"].get("filename", ""),
                            "document_part": outcome["row"].get("document_part", ""),
                            "part_index": outcome["row"].get("part_index"),
                            "part_total": outcome["row"].get("part_total"),
                            "source_pdf_index": outcome["row"].get("source_pdf_index"),
                            "error_category": error_category,
                            "error": error_text,
                            "endpoint": args.endpoint,
                            "model": args.model,
                            "openrouter_provider": args.openrouter_provider,
                        },
                    )
                    failure_log_records += 1
                print(
                    f"  ! Failed to analyze {row_ref}: {error_text}",
                    file=sys.stderr,
                )
                continue

            csv_row, json_record = build_output_records(
                idx=row_idx,
                source_row=outcome["row"],
                result=outcome["result"],
                args=args,
                config_metadata=config_metadata,
                quality=outcome["quality"],
                processing_status=outcome["processing_status"],
                skip_reason=outcome["skip_reason"],
            )
            output_router.write(row_idx, csv_row, json_record)

            row_key = row_source_id(outcome["row"])
            request_meta = (
                outcome["result"].get("_request_meta")
                if isinstance(outcome["result"], dict)
                and isinstance(outcome["result"].get("_request_meta"), dict)
                else {}
            )
            local_fallback_meta = (
                request_meta.get("local_fallback")
                if isinstance(request_meta.get("local_fallback"), dict)
                else {}
            )
            if local_fallback_meta.get("trigger") == "provider_content_filter":
                mark_content_filter_blocked(row_key)
            trigger = local_fallback_meta.get("trigger")
            if isinstance(trigger, str) and trigger:
                local_fallback_processed += 1
                local_fallback_processed_by_category[trigger] = (
                    local_fallback_processed_by_category.get(trigger, 0) + 1
                )
            if checkpoint_handle:
                checkpoint_handle.write(row_key + "\n")
                checkpoint_handle.flush()
            completed_source_ids.add(row_key)
            processed += 1
            if outcome["processing_status"] == "skipped":
                skipped += 1
            else:
                model_scored += 1

    def harvest_completed(*, block: bool, wait_for_all: bool = False) -> None:
        if not in_flight:
            return
        timeout = None if block else 0
        done, _ = concurrent.futures.wait(
            set(in_flight.keys()),
            timeout=timeout,
            return_when=(
                concurrent.futures.ALL_COMPLETED
                if wait_for_all
                else concurrent.futures.FIRST_COMPLETED
            ),
        )
        if not done:
            return
        for future in done:
            context = in_flight.pop(future)
            row_idx = context["idx"]
            row = context["row"]
            quality = context["quality"]
            started_at = context.get("submitted_at")
            wall_seconds = (
                max(0.0, time.monotonic() - started_at)
                if isinstance(started_at, (int, float))
                else None
            )
            try:
                result = future.result()
                attach_request_usage_and_cost(result, args)
                record_completion_for_eta(result)
                if args.flow_logs:
                    print(
                        f"[Row {row_idx}] [done] {row_source_id(row)} | "
                        f"{format_request_flow_details(result, wall_seconds=wall_seconds)} | "
                        f"in_flight={len(in_flight)}",
                        flush=True,
                    )
                pending_results[row_idx] = {
                    "type": "record",
                    "row": row,
                    "result": result,
                    "quality": quality,
                    "processing_status": "processed",
                    "skip_reason": "",
                }
            except Exception as exc:  # noqa: BLE001
                record_completion_for_eta()
                if args.flow_logs:
                    duration = f"{wall_seconds:.2f}s" if wall_seconds is not None else "n/a"
                    print(
                        f"[Row {row_idx}] [error] {row_source_id(row)} | wall={duration} | "
                        f"in_flight={len(in_flight)} | {exc}",
                        flush=True,
                    )
                pending_results[row_idx] = {
                    "type": "error",
                    "row": row,
                    "error": str(exc),
                    "error_category": classify_failure_reason(str(exc)),
                }
        flush_ready()

    resume_skipped_rows = 0
    resume_skipped_rows_reported = 0
    resume_skip_last_idx = 0
    resume_skip_last_report_at = time.monotonic()

    def report_resume_skip_progress(*, force: bool = False) -> None:
        nonlocal resume_skipped_rows, resume_skipped_rows_reported
        nonlocal resume_skip_last_idx, resume_skip_last_report_at
        if resume_skipped_rows <= resume_skipped_rows_reported:
            return
        newly_skipped = resume_skipped_rows - resume_skipped_rows_reported
        now = time.monotonic()
        if not force and newly_skipped < 200 and (now - resume_skip_last_report_at) < 5:
            return
        print(
            f"[resume] skipped {resume_skipped_rows:,} already-processed row(s) "
            f"(latest row {resume_skip_last_idx:,}).",
            flush=True,
        )
        resume_skipped_rows_reported = resume_skipped_rows
        resume_skip_last_report_at = now

    try:
        for idx, row in enumerate(
            iter_rows(
                args.input,
                input_glob=args.input_glob,
                include_text=True,
                processing_mode=active_processing_mode,
                pdf_part_pages=args.pdf_part_pages,
                pdf_page_count_cache=pdf_page_count_cache,
            ),
            start=1,
        ):
            if idx < args.start_row:
                continue
            if args.end_row is not None and idx > args.end_row:
                break
            if row_is_after_pdf_range(row, end_pdf=args.end_pdf):
                break
            if not row_matches_pdf_range(
                row,
                start_pdf=args.start_pdf,
                end_pdf=args.end_pdf,
            ):
                continue
            if args.max_rows is not None and scheduled >= args.max_rows:
                break

            filename = row["filename"]
            source_id = row_source_id(row)
            text = row["text"]
            input_kind = row.get("input_kind", "text")
            source_path = Path(row["source_path"]) if row.get("source_path") else None

            if selected_source_ids is not None and source_id not in selected_source_ids:
                continue
            if excluded_source_ids is not None and source_id in excluded_source_ids:
                continue
            if source_id in completed_source_ids:
                resume_skipped_rows += 1
                resume_skip_last_idx = idx
                report_resume_skip_progress(force=False)
                continue

            report_resume_skip_progress(force=True)
            scheduled += 1
            emit_order.append(idx)

            if input_kind == "text":
                quality = assess_text_quality(text)
                skip_reason = build_skip_reason(quality, args) if args.skip_low_quality else None
            else:
                quality = {
                    "mode": "image",
                    "char_count": 0,
                    "word_count": 0,
                    "avg_word_length": 0,
                    "alpha_ratio": 0,
                    "unique_word_ratio": 0,
                    "short_token_ratio": 0,
                    "long_word_count": 0,
                    "max_repeated_char_run": 0,
                }
                skip_reason = None
            if skip_reason:
                print(f"[Row {idx}] [skip] {source_id}: {skip_reason}", flush=True)
                record_completion_for_eta()
                pending_results[idx] = {
                    "type": "record",
                    "row": row,
                    "result": build_skipped_model_result(skip_reason),
                    "quality": quality,
                    "processing_status": "skipped",
                    "skip_reason": skip_reason,
                }
                flush_ready()
                continue

            # Show source row index and submission progress.
            # In parallel mode, completed-row counters lag while requests are in flight.
            if target_total:
                if target_total_is_estimate:
                    progress_prefix = f"[Row {idx}] [{scheduled}/~{target_total} est.]"
                else:
                    progress_prefix = f"[Row {idx}] [{scheduled}/{target_total} new]"
            else:
                progress_prefix = f"[Row {idx}] [{scheduled}]"

            eta_text = format_eta(
                start_time,
                eta_completed,
                target_total or 0,
                args.power_watts,
                args.electric_rate,
                api_cost_usd_total if api_rows_with_cost > 0 else None,
                eta_completion_times,
            )
            display_name = row.get("document_part") or filename
            print(f"{progress_prefix} Processing {display_name}... {eta_text}", flush=True)

            part_start_page = row.get("part_start_page")
            part_end_page = row.get("part_end_page")
            if (
                isinstance(part_start_page, int)
                and isinstance(part_end_page, int)
                and part_end_page >= part_start_page
            ):
                effective_image_max_pages = part_end_page - part_start_page + 1
            else:
                effective_image_max_pages = args.image_max_pages

            request_kwargs = {
                "endpoint": args.endpoint,
                "api_format": args.api_format,
                "model": args.model,
                "filename": row.get("analysis_filename") or filename,
                "text": text,
                "input_kind": input_kind,
                "image_path": source_path if input_kind == "image" else None,
                "image_max_pages": effective_image_max_pages,
                "pdf_pages_per_image": args.pdf_pages_per_image,
                "image_output_format": args.image_output_format,
                "image_jpeg_quality": args.image_jpeg_quality,
                "image_max_side": args.image_max_side,
                "debug_image_dir": args.debug_image_dir,
                "image_start_page": int(row.get("part_start_page") or 1),
                "image_part_index": (
                    int(row["part_index"]) if row.get("part_index") is not None else None
                ),
                "image_part_total": (
                    int(row["part_total"]) if row.get("part_total") is not None else None
                ),
                "image_render_dpi": args.image_render_dpi,
                "system_prompt": system_prompt,
                "api_key": args.api_key,
                "timeout": args.timeout,
                "max_retries": args.max_retries,
                "retry_backoff": args.retry_backoff,
                "temperature": args.temperature,
                "max_output_tokens": args.max_output_tokens,
                "reasoning_effort": args.reasoning_effort,
                "image_detail": args.image_detail,
                "request_semaphore": request_semaphore,
                "http_referer": args.http_referer,
                "x_title": args.x_title,
                "openrouter_provider": args.openrouter_provider,
                "openrouter_allow_fallbacks": (
                    None
                    if not args.openrouter_provider
                    else (not args.openrouter_no_fallbacks)
                ),
                "config_metadata": config_metadata,
            }

            if executor is None:
                try:
                    sync_started = time.monotonic()
                    result = execute_request_with_local_fallback(
                        request_kwargs=request_kwargs,
                        row_idx=idx,
                        source_id=source_id,
                    )
                    attach_request_usage_and_cost(result, args)
                    record_completion_for_eta(result)
                    if args.flow_logs:
                        print(
                            f"[Row {idx}] [done] {source_id} | "
                            f"{format_request_flow_details(result, wall_seconds=time.monotonic() - sync_started)} | "
                            "in_flight=0",
                            flush=True,
                        )
                    pending_results[idx] = {
                        "type": "record",
                        "row": row,
                        "result": result,
                        "quality": quality,
                        "processing_status": "processed",
                        "skip_reason": "",
                    }
                except Exception as exc:  # noqa: BLE001
                    record_completion_for_eta()
                    pending_results[idx] = {
                        "type": "error",
                        "row": row,
                        "error": str(exc),
                        "error_category": classify_failure_reason(str(exc)),
                    }
                flush_ready()
            else:
                if parallel_scheduling == "window":
                    while len(in_flight) >= max_in_flight_tasks:
                        harvest_completed(block=True)
                if args.flow_logs:
                    print(
                        f"[Row {idx}] [queue] submit {source_id} | "
                        f"slot={len(in_flight)+1}/{max_in_flight_tasks}",
                        flush=True,
                    )
                future = executor.submit(
                    execute_request_with_local_fallback,
                    request_kwargs=request_kwargs,
                    row_idx=idx,
                    source_id=source_id,
                )
                in_flight[future] = {
                    "idx": idx,
                    "row": row,
                    "quality": quality,
                    "submitted_at": time.monotonic(),
                }
                if parallel_scheduling == "window":
                    harvest_completed(block=False)
                elif len(in_flight) >= args.max_parallel_requests:
                    harvest_completed(block=True, wait_for_all=True)

            if args.sleep:
                time.sleep(args.sleep)

        report_resume_skip_progress(force=True)
        while in_flight:
            harvest_completed(block=True, wait_for_all=(parallel_scheduling == "batch"))

        flush_ready()
    except KeyboardInterrupt:
        interrupted = True
        print(
            "\nPause requested (Ctrl+C). Flushing completed rows and preserving checkpoint state...",
            file=sys.stderr,
        )
        harvest_completed(block=False)
        flush_ready()
    finally:
        if executor:
            if interrupted:
                for future in list(in_flight.keys()):
                    future.cancel()
                in_flight.clear()
                executor.shutdown(wait=False, cancel_futures=True)
            else:
                executor.shutdown(wait=True)
        if checkpoint_handle:
            checkpoint_handle.close()
        output_router.close()
        if pdf_page_count_cache is not None:
            pdf_page_count_cache.flush()

    elapsed = time.monotonic() - start_time
    elapsed_hours = elapsed / 3600
    cost_summary = format_cost_summary(args.power_watts, args.electric_rate, elapsed_hours, args.run_hours)
    status_verb = "Paused after" if interrupted else "Completed"
    if args.chunk_size > 0:
        complete_msg = (
            f"{status_verb} {processed} new rows in {format_duration(elapsed)} "
            f"({model_scored} modeled, {skipped} skipped, {failed} failed). "
            f"Chunks saved under {args.chunk_dir} | manifest: {args.chunk_manifest}"
        )
    else:
        complete_msg = (
            f"{status_verb} {processed} new rows in {format_duration(elapsed)} "
            f"({model_scored} modeled, {skipped} skipped, {failed} failed). "
            f"CSV saved to {args.output} | JSONL appended to {args.json_output}"
        )
    if interrupted:
        complete_msg = (
            f"{complete_msg}\nResume by re-running with --resume "
            f"(checkpoint: {args.checkpoint})."
        )
    if api_rows_with_usage > 0:
        token_summary = (
            "Token usage "
            f"(rows with usage: {api_rows_with_usage}): "
            f"in={api_prompt_tokens_total}, out={api_completion_tokens_total}, total={api_total_tokens_total}"
        )
        if api_cache_read_tokens_total > 0 or api_cache_write_tokens_total > 0:
            token_summary += (
                f", cache_read={api_cache_read_tokens_total}, "
                f"cache_write={api_cache_write_tokens_total}"
            )
        complete_msg = f"{complete_msg}\n{token_summary}"
    if api_rows_with_cost > 0:
        avg_cost = api_cost_usd_total / max(api_rows_with_cost, 1)
        complete_msg = (
            f"{complete_msg}\nModel API cost: ${api_cost_usd_total:.6f} "
            f"(rows with cost: {api_rows_with_cost}, avg ${avg_cost:.6f}/row)"
        )
    if args.failure_log:
        complete_msg = (
            f"{complete_msg}\nFailure log: {args.failure_log} "
            f"(records written: {failure_log_records}, unique source ids: {len(failure_source_ids)})"
        )
    if args.content_filter_blocklist:
        complete_msg = (
            f"{complete_msg}\nContent-filter blocklist: {args.content_filter_blocklist} "
            f"(new source ids appended this run: {content_filter_blocked_written})"
        )
    if local_fallback_enabled:
        category_summary = ", ".join(
            f"{key}={value}"
            for key, value in sorted(local_fallback_processed_by_category.items())
        ) or "none"
        complete_msg = (
            f"{complete_msg}\nLocal fallbacks completed this run: "
            f"{local_fallback_processed} ({category_summary})"
        )
    if cost_summary:
        complete_msg = f"{complete_msg}\n{cost_summary}"
    print(complete_msg)


if __name__ == "__main__":
    main()
