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
#   OPENROUTER_MODEL     — override model (default: qwen/qwen3-vl-30b-a3b-thinking)
#   OPENROUTER_PROVIDER  — override provider routing hint (default: nvidia)
#   AV_PARALLEL          — max concurrent requests (default: 4)
#   AV_FPS               — frames per second to extract from video (default: 1.0)
#   AV_MAX_FRAMES        — hard cap on frames per video (default: 120)
#   AV_GRID_COLS         — grid compositing columns (default: 2)
#   AV_GRID_ROWS         — grid compositing rows (default: 2)
#   AV_WHISPER_MODEL     — whisper model size (default: small)

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
  --seconds-per-frame N  Seconds between extracted frames (default: 2.0)
  --max-frames N         Hard cap on total frames per video (default: 120)
  --grid-cols N          Grid compositing columns (default: 2)
  --grid-rows N          Grid compositing rows (default: 2)
  --no-grid              Disable frame grid compositing
  --whisper-model SIZE   tiny | base | small | medium | large (default: small)
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

MODEL="${AV_MODEL:-qwen/qwen3-vl-30b-a3b-thinking}"
# Do NOT inherit OPENROUTER_PROVIDER here — that's set to 'alibaba' for the PDF pipeline.
# For the AV pipeline (Qwen3 VL), leave provider empty to let OpenRouter choose.
PROVIDER="${AV_PROVIDER:-}"
ENDPOINT="${AV_ENDPOINT:-${OPENROUTER_ENDPOINT:-https://openrouter.ai/api/v1}}"
PARALLEL="${AV_PARALLEL:-4}"
SECONDS_PER_FRAME="${AV_SECONDS_PER_FRAME:-2.0}"
MAX_FRAMES="${AV_MAX_FRAMES:-120}"
GRID_COLS="${AV_GRID_COLS:-2}"
GRID_ROWS="${AV_GRID_ROWS:-2}"
WHISPER="${AV_WHISPER_MODEL:-small}"

echo "[config] volume=$VOLUME | model=$MODEL | provider=${PROVIDER:-auto}"
echo "[config] parallel=$PARALLEL | seconds_per_frame=$SECONDS_PER_FRAME | max_frames=$MAX_FRAMES | grid=${GRID_COLS}x${GRID_ROWS} | whisper=$WHISPER"
echo "[config] endpoint=$ENDPOINT"
echo ""

CMD=(
  python3 av_ranker.py
  --volume "$VOLUME"
  --model "$MODEL"
  --endpoint "$ENDPOINT"
  --max-parallel "$PARALLEL"
  --seconds-per-frame "$SECONDS_PER_FRAME"
  --max-frames "$MAX_FRAMES"
  --grid-cols "$GRID_COLS"
  --grid-rows "$GRID_ROWS"
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

CMD+=("${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}")

printf '[cmd] '
printf '%q ' "${CMD[@]}"
printf '\n\n'

"${CMD[@]}"
