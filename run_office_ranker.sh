#!/usr/bin/env bash
# run_office_ranker.sh — Process Office documents for a given DOJ volume.
#
# Converts .docx, .xlsx, .doc, .xls, .csv, .ppt, .pptx files to PDF using
# LibreOffice, then runs the same VLM pipeline as the PDF ranker.
#
# Usage:
#   ./run_office_ranker.sh --volume 8 [options]
#   ./run_office_ranker.sh --volume 9 --dry-run
#   ./run_office_ranker.sh --volume 8 --max-files 3       # smoke test
#   ./run_office_ranker.sh --volume 8 --resume            # skip already done
#
# Environment:
#   OPENROUTER_API_KEY     — required if not in .env.openrouter
#   OFFICE_MODEL           — override model (default: qwen/qwen3-vl-30b-a3b-thinking)
#   OFFICE_PROVIDER        — override provider routing hint (default: alibaba)
#   OFFICE_ENDPOINT        — override API endpoint
#   OFFICE_PARALLEL        — max concurrent requests (default: 2)
#   OFFICE_MAX_PAGES       — pages per office file to render (default: 8)
#   OFFICE_DPI             — render DPI (default: 150)
#
# Prerequisites:
#   LibreOffice (soffice):  brew install --cask libreoffice   (macOS)
#                            apt-get install libreoffice       (Linux)
#                            or download from https://www.libreoffice.org/download/
#   pdftoppm / pdfinfo:     brew install poppler
#                            apt-get install poppler-utils

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load .env.openrouter if present
if [[ -f ".env.openrouter" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env.openrouter"
  set +a
fi

VOLUME=""
DRY_RUN=0
EXTRA_ARGS=()

usage() {
  cat <<'USAGE'
Usage:
  ./run_office_ranker.sh --volume N [options]

Required:
  --volume N             DOJ volume number (example: 8 or 9)

Pass-through options (forwarded to office_ranker.py):
  --max-files N          Limit to N files (smoke test)
  --resume               Skip already-processed files
  --dry-run              Print what would be processed
  --max-pages N          Pages per office file to render (default: 8)
  --max-parallel N       Concurrent workers (default: 2)
  --image-render-dpi N   Render DPI (default: 150)
  --model ID             Override model ID
  --endpoint URL         Override API endpoint
  --openrouter-provider  Override provider routing hint
  -h, --help             Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --volume)
      VOLUME="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      EXTRA_ARGS+=("--dry-run")
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$VOLUME" ]]; then
  echo "Missing required --volume" >&2
  usage >&2
  exit 1
fi

if [[ ! "$VOLUME" =~ ^[0-9]+$ ]]; then
  echo "Invalid --volume value: $VOLUME" >&2
  exit 1
fi

MODEL="${OFFICE_MODEL:-${OPENROUTER_MODEL:-qwen/qwen3-vl-30b-a3b-thinking}}"
PROVIDER="${OFFICE_PROVIDER:-${OPENROUTER_PROVIDER:-alibaba}}"
ENDPOINT="${OFFICE_ENDPOINT:-${OPENROUTER_ENDPOINT:-https://openrouter.ai/api/v1}}"
PARALLEL="${OFFICE_PARALLEL:-2}"
MAX_PAGES="${OFFICE_MAX_PAGES:-8}"
DPI="${OFFICE_DPI:-150}"

echo "[config] volume=$VOLUME | model=$MODEL | provider=${PROVIDER:-auto}"
echo "[config] parallel=$PARALLEL | max_pages=$MAX_PAGES | dpi=$DPI"
echo "[config] endpoint=$ENDPOINT"
echo ""

CMD=(
  python3 office_ranker.py
  --volume "$VOLUME"
  --model "$MODEL"
  --endpoint "$ENDPOINT"
  --max-parallel "$PARALLEL"
  --max-pages "$MAX_PAGES"
  --image-render-dpi "$DPI"
  --resume
)

if [[ -n "$PROVIDER" ]]; then
  CMD+=(--openrouter-provider "$PROVIDER")
fi

if [[ -n "${OPENROUTER_REFERER:-}" ]]; then
  CMD+=(--http-referer "$OPENROUTER_REFERER")
fi
if [[ -n "${OPENROUTER_TITLE:-}" ]]; then
  CMD+=(--x-title "$OPENROUTER_TITLE")
fi

CMD+=("${EXTRA_ARGS[@]}")

printf '[cmd] '
printf '%q ' "${CMD[@]}"
printf '\n\n'

"${CMD[@]}"
