#!/usr/bin/env bash
# run_av_ranker.sh — Process audio/video files for a given DOJ volume.
#
# Wraps av_ranker.py with sane defaults and .env.openrouter loading.
#
# Usage:
#   ./run_av_ranker.sh --volume 10 [options]
#   ./run_av_ranker.sh --volume 10 --dry-run
#   ./run_av_ranker.sh --volume 10 --max-files 3       # smoke test
#   ./run_av_ranker.sh --volume 10 --no-transcription  # frames only
#
# Environment:
#   OPENROUTER_API_KEY   — required if not in .env.openrouter
#   OPENROUTER_MODEL     — override model (default: nvidia/nemotron-nano-12b-v2-vl:free)
#   OPENROUTER_PROVIDER  — override provider routing hint (default: nvidia)
#   AV_PARALLEL          — max concurrent requests (default: 4)
#   AV_FRAMES            — frames per video (default: 8)
#   AV_WHISPER_MODEL     — whisper model size (default: base)

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
  ./run_av_ranker.sh --volume N [options]

Required:
  --volume N             DOJ volume number (example: 10)

Pass-through options (forwarded to av_ranker.py):
  --max-files N          Limit to N files (smoke test)
  --no-transcription     Disable Whisper transcription
  --resume               Skip already-processed files
  --dry-run              Print what would be processed
  --file-types TYPE      all | video | audio (default: all)
  --max-parallel N       Concurrent workers (default: 4)
  --num-frames N         Frames per video (default: 8)
  --whisper-model SIZE   tiny | base | small | medium | large (default: base)
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

MODEL="${AV_MODEL:-nvidia/nemotron-nano-12b-v2-vl:free}"
# Do NOT inherit OPENROUTER_PROVIDER here — that's set to 'alibaba' for the PDF pipeline.
# For the AV pipeline (Nemotron), leave provider empty to let OpenRouter choose.
PROVIDER="${AV_PROVIDER:-}"
ENDPOINT="${AV_ENDPOINT:-${OPENROUTER_ENDPOINT:-https://openrouter.ai/api/v1}}"
PARALLEL="${AV_PARALLEL:-4}"
FRAMES="${AV_FRAMES:-8}"
WHISPER="${AV_WHISPER_MODEL:-base}"

echo "[config] volume=$VOLUME | model=$MODEL | provider=${PROVIDER:-auto}"
echo "[config] parallel=$PARALLEL | frames=$FRAMES | whisper=$WHISPER"
echo "[config] endpoint=$ENDPOINT"
echo ""

CMD=(
  python3 av_ranker.py
  --volume "$VOLUME"
  --model "$MODEL"
  --endpoint "$ENDPOINT"
  --max-parallel "$PARALLEL"
  --num-frames "$FRAMES"
  --whisper-model "$WHISPER"
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
