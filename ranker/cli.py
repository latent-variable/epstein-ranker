from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

from .constants import DEFAULT_JUSTICE_FILES_BASE_URL


def explicit_cli_destinations(argv: List[str]) -> Set[str]:
    explicit: Set[str] = set()
    for token in argv:
        if token == "--":
            break
        if not token.startswith("--"):
            continue
        flag = token[2:]
        if "=" in flag:
            flag = flag.split("=", 1)[0]
        if flag.startswith("no-"):
            flag = flag[3:]
        explicit.add(flag.replace("-", "_"))
    return explicit


def sanitize_dataset_tag(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "dataset"


def infer_dataset_tag(input_path: Path) -> str:
    if input_path.name:
        stem = input_path.stem if input_path.is_file() else input_path.name
        if stem:
            return sanitize_dataset_tag(stem)
    return "dataset"


def apply_dataset_workspace_defaults(
    args: argparse.Namespace,
    *,
    cli_explicit: Optional[Set[str]] = None,
) -> None:
    explicit = cli_explicit or set()
    if not args.dataset_workspace_root:
        return

    workspace_root = Path(args.dataset_workspace_root)
    tag = args.dataset_tag or infer_dataset_tag(args.input)
    tag = sanitize_dataset_tag(tag)
    args.dataset_tag = tag
    base = workspace_root / tag

    workspace_defaults = {
        "output": base / "results" / "epstein_ranked.csv",
        "json_output": base / "results" / "epstein_ranked.jsonl",
        "checkpoint": base / "state" / ".epstein_checkpoint",
        "chunk_dir": base / "chunks",
        "chunk_manifest": base / "metadata" / "chunks.json",
        "run_metadata_file": base / "metadata" / "run_metadata.json",
    }
    for key, value in workspace_defaults.items():
        if key not in explicit:
            setattr(args, key, value)

    if "known_json" not in explicit:
        args.known_json = []


def apply_config_defaults(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    cli_explicit: Optional[Set[str]] = None,
) -> None:
    explicit = cli_explicit or set()
    config_path: Path = args.config  # type: ignore[assignment]
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)
    for key, value in data.items():
        if not hasattr(args, key):
            continue
        if key in explicit:
            continue
        current = getattr(args, key)
        default = parser.get_default(key)
        if current == default:
            if isinstance(default, Path):
                setattr(args, key, Path(value))
            else:
                setattr(args, key, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Call a local model gateway (OpenAI-style or /chat style) to extract "
            "useful information and rank each source row."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional TOML config file to supply defaults (see ranker_config.example.toml).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data") / "EPS_FILES_20K_NOV2026.csv",
        help="Path to source CSV (filename/text) or a directory of text/image/PDF files.",
    )
    parser.add_argument(
        "--input-glob",
        default="*.txt",
        help="Glob used when --input points to a directory (searched recursively).",
    )
    parser.add_argument(
        "--processing-mode",
        choices=["auto", "text", "image"],
        default="auto",
        help="Process textual input, image/PDF input, or auto-detect by extension.",
    )
    parser.add_argument(
        "--source-files-base-url",
        default=None,
        help=(
            "Optional base URL for local source files. If omitted, source links are "
            "derived from absolute paths under /data as /data/..."
        ),
    )
    parser.add_argument(
        "--dataset-workspace-root",
        type=Path,
        default=None,
        help=(
            "If set, isolate outputs/checkpoint/chunks under "
            "<dataset-workspace-root>/<dataset-tag>/ to avoid mixing corpora."
        ),
    )
    parser.add_argument(
        "--dataset-tag",
        default=None,
        help="Dataset identifier used with --dataset-workspace-root (defaults to input name).",
    )
    parser.add_argument(
        "--dataset-source-label",
        default=None,
        help="Optional human-readable source label for provenance metadata.",
    )
    parser.add_argument(
        "--dataset-source-url",
        default=None,
        help="Optional source URL for provenance metadata.",
    )
    parser.add_argument(
        "--dataset-metadata-file",
        type=Path,
        default=None,
        help="Optional JSON file containing dataset provenance/stats to attach to run metadata.",
    )
    parser.add_argument(
        "--run-metadata-file",
        type=Path,
        default=None,
        help="Optional JSON sidecar path for run metadata (auto-set in workspace mode).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "epstein_ranked.csv",
        help="Path to write the ranked CSV results.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("data") / "epstein_ranked.jsonl",
        help="Path to append newline-delimited JSON records for each row.",
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:5555/v1",
        help="Base URL of the local model gateway.",
    )
    parser.add_argument(
        "--api-format",
        choices=["auto", "openai", "chat"],
        default="auto",
        help=(
            "Request format to use: openai (/chat/completions), chat (/chat), "
            "or auto-detect/fallback."
        ),
    )
    parser.add_argument(
        "--model",
        default="qwen/qwen3-vl-30b-a3b-thinking",
        help="Model identifier exposed by the server (check via --list-models).",
    )
    parser.add_argument(
        "--justice-files-base-url",
        default=DEFAULT_JUSTICE_FILES_BASE_URL,
        help="Base URL for DOJ Epstein PDF files used to derive source document links.",
    )
    parser.add_argument(
        "--max-parallel-requests",
        type=int,
        default=4,
        help="Maximum concurrent model requests.",
    )
    parser.add_argument(
        "--parallel-scheduling",
        choices=["auto", "window", "batch"],
        default="auto",
        help=(
            "Scheduler style for concurrent work: window keeps slots full continuously; "
            "batch waits for a full group to finish before submitting next group."
        ),
    )
    parser.add_argument(
        "--image-prefetch",
        type=int,
        default=0,
        help=(
            "Extra queued image tasks (render/prep) beyond --max-parallel-requests "
            "to keep local pipelines saturated."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for model responses (0.0 = deterministic, higher = more random).",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=900,
        help="Upper bound for model completion tokens per request.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Path to a text file containing the system prompt (overrides default).",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Inline system prompt string (overrides --prompt-file and default).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional API key for servers that require authentication.",
    )
    parser.add_argument(
        "--http-referer",
        default=None,
        help="Optional HTTP-Referer header (recommended by OpenRouter).",
    )
    parser.add_argument(
        "--x-title",
        default=None,
        help="Optional X-Title header (recommended by OpenRouter).",
    )
    parser.add_argument(
        "--openrouter-provider",
        default=None,
        help=(
            "Optional OpenRouter provider slug/name for routing preferences "
            "(example: alibaba)."
        ),
    )
    parser.add_argument(
        "--openrouter-no-fallbacks",
        action="store_true",
        default=False,
        help=(
            "Disable OpenRouter provider fallback routing (fail fast if the selected "
            "provider cannot serve the request)."
        ),
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        help="If provided, passes reasoning effort hints supported by some models.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip rows already present in the checkpoint or JSON output.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Allow truncating existing output/JSONL files (use with caution).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("data") / ".epstein_checkpoint",
        help="Plain-text file storing processed filenames (used with --resume).",
    )
    parser.add_argument(
        "--known-json",
        action="append",
        default=[],
        help="Additional JSONL files containing already-processed rows to skip.",
    )
    parser.add_argument(
        "--only-source-ids-file",
        type=Path,
        default=None,
        help=(
            "Optional newline/JSON/JSONL file listing source_id values to process. "
            "When set, rows not in this list are skipped."
        ),
    )
    parser.add_argument(
        "--failure-log",
        type=Path,
        default=None,
        help=(
            "Optional JSONL path for recording failed rows (source_id, row index, "
            "error, and lightweight metadata) for targeted retries."
        ),
    )
    parser.add_argument(
        "--exclude-source-ids-file",
        type=Path,
        default=None,
        help=(
            "Optional newline/JSON/JSONL file listing source_id values to skip. "
            "Rows in this set are never submitted to the model."
        ),
    )
    parser.add_argument(
        "--content-filter-blocklist",
        type=Path,
        default=None,
        help=(
            "Optional path where provider content-filter failures are appended as "
            "source_id entries (one per line)."
        ),
    )
    parser.add_argument(
        "--local-fallback-on-content-filter",
        action="store_true",
        default=False,
        help=(
            "If set, provider content-filter failures on hosted endpoints are retried "
            "immediately against a local endpoint/model."
        ),
    )
    parser.add_argument(
        "--local-fallback-on-model-output-error",
        action="store_true",
        default=False,
        help=(
            "If set, model output errors (for example empty/invalid JSON responses) "
            "on hosted endpoints are retried immediately against a local endpoint/model."
        ),
    )
    parser.add_argument(
        "--local-fallback-endpoint",
        default="http://localhost:5555/v1",
        help="Endpoint used for immediate local fallback requests.",
    )
    parser.add_argument(
        "--local-fallback-model",
        default="qwen/qwen3-vl-30b-a3b-thinking",
        help="Model ID used for immediate local fallback requests.",
    )
    parser.add_argument(
        "--local-fallback-api-format",
        choices=["auto", "openai", "chat"],
        default="openai",
        help="API format used for immediate local fallback requests.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="If >0, rotate outputs every N source rows and store chunk files in --chunk-dir.",
    )
    parser.add_argument(
        "--chunk-dir",
        type=Path,
        default=Path("contrib"),
        help="Directory to store chunked outputs when --chunk-size > 0.",
    )
    parser.add_argument(
        "--chunk-manifest",
        type=Path,
        default=Path("data") / "chunks.json",
        help="Manifest JSON file listing generated chunks (used by the viewer).",
    )
    parser.add_argument(
        "--include-action-items",
        action="store_true",
        help="Request action items from the model and include them in outputs.",
    )
    parser.add_argument(
        "--start-row",
        type=int,
        default=1,
        help="1-based row index to start processing (useful for collaborative chunking).",
    )
    parser.add_argument(
        "--end-row",
        type=int,
        default=None,
        help="1-based row index to stop processing (inclusive).",
    )
    parser.add_argument(
        "--start-pdf",
        type=int,
        default=1,
        help=(
            "1-based PDF file index to start processing (directory image mode only; "
            "applies before per-PDF part splitting)."
        ),
    )
    parser.add_argument(
        "--end-pdf",
        type=int,
        default=None,
        help=(
            "1-based PDF file index to stop processing (inclusive; directory image "
            "mode only, before per-PDF part splitting)."
        ),
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Limit processing to the first N rows (useful for smoke-tests).",
    )
    parser.add_argument(
        "--workload-scan",
        choices=["auto", "full", "defer"],
        default="auto",
        help=(
            "Control upfront workload counting: full scans all rows before processing; "
            "defer starts processing immediately without exact upfront totals; "
            "auto uses defer for image directory + PDF split runs, otherwise full."
        ),
    )
    parser.add_argument(
        "--workload-estimate-sample-size",
        type=int,
        default=64,
        help=(
            "When workload scanning is deferred, sample up to N PDFs to estimate "
            "split-part totals and ETA."
        ),
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between requests to avoid overwhelming the server.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="HTTP request timeout in seconds (default: 600 = 10 minutes).",
    )
    parser.add_argument(
        "--image-max-pages",
        type=int,
        default=1,
        help="Maximum PDF pages to render and send per document in image mode.",
    )
    parser.add_argument(
        "--pdf-pages-per-image",
        type=int,
        default=1,
        help=(
            "When >1, pack this many PDF pages into one tiled image before model input "
            "(example: 4 = 2x2 collage per image block)."
        ),
    )
    parser.add_argument(
        "--pdf-part-pages",
        type=int,
        default=0,
        help=(
            "If >0, split each PDF into independent part records of this many pages "
            "(example: 24 yields part_001, part_002, ...)."
        ),
    )
    parser.add_argument(
        "--image-render-dpi",
        type=int,
        default=180,
        help="DPI used for PDF page rendering before multimodal inference.",
    )
    parser.add_argument(
        "--image-detail",
        choices=["auto", "low", "high"],
        default="low",
        help="Vision detail hint passed to image_url blocks (lower detail reduces token usage).",
    )
    parser.add_argument(
        "--image-output-format",
        choices=["png", "jpeg"],
        default="jpeg",
        help="Output format for rendered/packed images before request upload.",
    )
    parser.add_argument(
        "--image-jpeg-quality",
        type=int,
        default=75,
        help="JPEG quality (1-95) when --image-output-format=jpeg.",
    )
    parser.add_argument(
        "--image-max-side",
        type=int,
        default=1024,
        help="If >0, downscale prepared images so the longest side is at most this many pixels.",
    )
    parser.add_argument(
        "--debug-image-dir",
        type=Path,
        default=None,
        help="Optional directory to save intermediate rendered/packed images and prep timing summaries.",
    )
    parser.add_argument(
        "--flow-logs",
        action="store_true",
        default=False,
        help=(
            "Emit per-row flow logs (queue/prep/request timings) to help confirm parallel behavior "
            "and identify preprocessing bottlenecks."
        ),
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum request attempts for transient endpoint/model failures.",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=1.5,
        help="Base seconds for exponential retry backoff (attempt_n = backoff * 2^(n-1)).",
    )
    parser.add_argument(
        "--skip-low-quality",
        dest="skip_low_quality",
        action="store_true",
        default=True,
        help="Skip low-signal rows (empty/too-short/noisy) before model calls.",
    )
    parser.add_argument(
        "--no-skip-low-quality",
        dest="skip_low_quality",
        action="store_false",
        help="Disable low-quality row skipping.",
    )
    parser.add_argument(
        "--min-text-chars",
        type=int,
        default=60,
        help="Minimum non-whitespace character count required to process a row.",
    )
    parser.add_argument(
        "--min-text-words",
        type=int,
        default=12,
        help="Minimum token count required to process a row.",
    )
    parser.add_argument(
        "--min-alpha-ratio",
        type=float,
        default=0.25,
        help="Minimum alphabetic-character ratio required to process a row.",
    )
    parser.add_argument(
        "--min-unique-word-ratio",
        type=float,
        default=0.15,
        help="Minimum unique-word ratio required to process a row.",
    )
    parser.add_argument(
        "--max-short-token-ratio",
        type=float,
        default=0.6,
        help="Maximum fraction of tokens with length <= 2 before a row is considered noisy.",
    )
    parser.add_argument(
        "--min-avg-word-length",
        type=float,
        default=3.0,
        help="Minimum average token length required to process a row.",
    )
    parser.add_argument(
        "--min-long-word-count",
        type=int,
        default=4,
        help="Minimum number of tokens with length >= 4 required to process a row.",
    )
    parser.add_argument(
        "--max-repeated-char-run",
        type=int,
        default=40,
        help="Maximum repeated-character run before a row is considered noisy.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List models exposed by the endpoint and exit.",
    )
    parser.add_argument(
        "--rebuild-manifest",
        action="store_true",
        help="Scan chunk files and rebuild the manifest (data/chunks.json), then exit.",
    )
    parser.add_argument(
        "--power-watts",
        type=float,
        default=None,
        help="If provided, estimate energy usage (average watts).",
    )
    parser.add_argument(
        "--electric-rate",
        type=float,
        default=None,
        help="Electricity cost in USD per kWh for cost estimation.",
    )
    parser.add_argument(
        "--run-hours",
        type=float,
        default=None,
        help="Override elapsed hours for cost estimate (otherwise uses wall time).",
    )
    parser.add_argument(
        "--input-price-per-1m",
        type=float,
        default=None,
        help=(
            "Optional API input token price in USD per 1M tokens "
            "(used for model cost tracking)."
        ),
    )
    parser.add_argument(
        "--output-price-per-1m",
        type=float,
        default=None,
        help=(
            "Optional API output token price in USD per 1M tokens "
            "(used for model cost tracking)."
        ),
    )
    parser.add_argument(
        "--cache-read-price-per-1m",
        type=float,
        default=None,
        help=(
            "Optional API cache-read token price in USD per 1M tokens. "
            "When set, cached input tokens are priced separately from normal input tokens."
        ),
    )
    parser.add_argument(
        "--cache-write-price-per-1m",
        type=float,
        default=None,
        help=(
            "Optional API cache-write token price in USD per 1M tokens. "
            "When set, cache creation tokens are priced separately from normal input tokens."
        ),
    )
    args = parser.parse_args()
    cli_explicit = explicit_cli_destinations(sys.argv[1:])
    config_path = None
    if args.config:
        config_path = Path(args.config)
    else:
        for candidate in (Path("ranker_config.toml"), Path("ranker_config.example.toml")):
            if candidate.exists():
                config_path = candidate
                break
    if config_path:
        args.config = config_path
        apply_config_defaults(parser, args, cli_explicit=cli_explicit)
    if (
        args.processing_mode == "image"
        and "input_glob" not in cli_explicit
        and args.input_glob == parser.get_default("input_glob")
    ):
        args.input_glob = "*.pdf"
    apply_dataset_workspace_defaults(args, cli_explicit=cli_explicit)
    return args


def load_system_prompt(args: argparse.Namespace) -> Tuple[str, str]:
    if args.system_prompt:
        return args.system_prompt, "inline (--system-prompt)"

    prompt_file = args.prompt_file
    if not prompt_file:
        prompt_file = Path("prompts") / "default_system_prompt.txt"

    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    with prompt_file.open("r", encoding="utf-8") as f:
        prompt = f.read().strip()

    if not prompt:
        raise ValueError(f"Prompt file is empty: {prompt_file}")

    return prompt, str(prompt_file)
