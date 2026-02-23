#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VOLUME=""
START_PDF=""
END_PDF=""
DRY_RUN=0
SKIP_LOCAL_RETRY=0
CONCURRENT=0
INTERRUPTED=0

trap 'INTERRUPTED=1' INT

OPENROUTER_MODEL="${OPENROUTER_MODEL:-qwen/qwen3-vl-30b-a3b-thinking}"
OPENROUTER_PROVIDER="${OPENROUTER_PROVIDER:-alibaba}"
OPENROUTER_ENDPOINT="${OPENROUTER_ENDPOINT:-https://openrouter.ai/api/v1}"
LOCAL_ENDPOINT="${LOCAL_ENDPOINT:-http://localhost:5555/v1}"
LOCAL_MODEL="${LOCAL_MODEL:-qwen/qwen3-vl-30b}"
LOCAL_API_FORMAT="${LOCAL_API_FORMAT:-openai}"
CLOUD_PARALLEL_SCHEDULING="${CLOUD_PARALLEL_SCHEDULING:-window}"
LOCAL_PARALLEL_SCHEDULING="${LOCAL_PARALLEL_SCHEDULING:-batch}"
CLOUD_PARALLEL="${CLOUD_PARALLEL:-}"
LOCAL_PARALLEL="${LOCAL_PARALLEL:-}"
INLINE_LOCAL_FALLBACK=1
INLINE_FALLBACK_MODEL_OUTPUT_ERROR=1

usage() {
  cat <<'USAGE'
Usage:
  ./run_hybrid_volume.sh --volume N [options]

Required:
  --volume N                 DOJ volume number (example: 11)

Options:
  --start-pdf N              Start PDF index (1-based, pre-split)
  --end-pdf N                End PDF index (inclusive, pre-split)
  --openrouter-endpoint URL  Cloud endpoint (default: https://openrouter.ai/api/v1)
  --openrouter-model ID      Cloud model (default: qwen/qwen3-vl-30b-a3b-thinking)
  --openrouter-provider ID   OpenRouter provider (default: alibaba)
  --local-endpoint URL       Local endpoint (default: http://localhost:5555/v1)
  --local-model ID           Local model (default: qwen/qwen3-vl-30b)
  --local-api-format FMT     Local API format: auto | openai | chat (default: openai)
  --inline-local-fallback    Enable immediate cloud->local fallback (default)
  --no-inline-local-fallback Disable immediate cloud->local fallback
  --inline-fallback-model-output-error
                             Enable inline fallback for model_output_error (default)
  --no-inline-fallback-model-output-error
                             Disable inline fallback for model_output_error
  --cloud-parallel N         Max concurrent cloud requests (default: run_ranker.sh default of 8)
  --local-parallel N         Max concurrent local requests (default: run_ranker.sh default of 8)
  --cloud-scheduling MODE    window | batch (default: window)
  --local-scheduling MODE    window | batch (default: batch)
  --skip-local-retry         Do cloud pass only (sequential mode)
  --concurrent               Run cloud and local simultaneously (both process full workload)
  --dry-run                  Print commands only
  -h, --help                 Show help

What it does:
  Sequential mode (default):
  1) Runs OpenRouter for the target volume and logs unresolved failures to:
     data/workspaces/standardworks_epstein_files_volXXXXX/metadata/failed_requests_openrouter.jsonl
  2) If that log has entries, reruns only unresolved failed source IDs locally.

  Concurrent mode (--concurrent):
  1) Launches cloud and local rankers simultaneously as background processes.
  2) Both process the full workload independently; whichever writes a result first wins.
  3) Skips the sequential failure-log retry phase.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --volume)
      VOLUME="${2:-}"
      shift 2
      ;;
    --start-pdf)
      START_PDF="${2:-}"
      shift 2
      ;;
    --end-pdf)
      END_PDF="${2:-}"
      shift 2
      ;;
    --openrouter-model)
      OPENROUTER_MODEL="${2:-}"
      shift 2
      ;;
    --openrouter-endpoint)
      OPENROUTER_ENDPOINT="${2:-}"
      shift 2
      ;;
    --openrouter-provider)
      OPENROUTER_PROVIDER="${2:-}"
      shift 2
      ;;
    --local-endpoint)
      LOCAL_ENDPOINT="${2:-}"
      shift 2
      ;;
    --local-model)
      LOCAL_MODEL="${2:-}"
      shift 2
      ;;
    --local-api-format)
      LOCAL_API_FORMAT="${2:-}"
      shift 2
      ;;
    --inline-local-fallback)
      INLINE_LOCAL_FALLBACK=1
      shift
      ;;
    --no-inline-local-fallback)
      INLINE_LOCAL_FALLBACK=0
      shift
      ;;
    --inline-fallback-model-output-error)
      INLINE_FALLBACK_MODEL_OUTPUT_ERROR=1
      shift
      ;;
    --no-inline-fallback-model-output-error)
      INLINE_FALLBACK_MODEL_OUTPUT_ERROR=0
      shift
      ;;
    --cloud-parallel)
      CLOUD_PARALLEL="${2:-}"
      shift 2
      ;;
    --local-parallel)
      LOCAL_PARALLEL="${2:-}"
      shift 2
      ;;
    --cloud-scheduling)
      CLOUD_PARALLEL_SCHEDULING="${2:-}"
      shift 2
      ;;
    --local-scheduling)
      LOCAL_PARALLEL_SCHEDULING="${2:-}"
      shift 2
      ;;
    --skip-local-retry)
      SKIP_LOCAL_RETRY=1
      shift
      ;;
    --concurrent)
      CONCURRENT=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
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

case "$LOCAL_API_FORMAT" in
  auto|openai|chat) ;;
  *)
    echo "Invalid --local-api-format: $LOCAL_API_FORMAT (expected: auto|openai|chat)" >&2
    exit 1
    ;;
esac

VOL_TAG="$(printf "standardworks_epstein_files_vol%05d" "$VOLUME")"
FAIL_LOG="data/workspaces/${VOL_TAG}/metadata/failed_requests_openrouter.jsonl"

mkdir -p "$(dirname "$FAIL_LOG")"
# Keep cumulative failure history so interrupted runs do not lose retry candidates.
touch "$FAIL_LOG"

CLOUD_CMD=(
  ./run_ranker.sh
  --volumes "$VOLUME"
  --provider openrouter
  --endpoint "$OPENROUTER_ENDPOINT"
  --openrouter-model "$OPENROUTER_MODEL"
  --openrouter-provider "$OPENROUTER_PROVIDER"
  --parallel-scheduling "$CLOUD_PARALLEL_SCHEDULING"
  --failure-log "$FAIL_LOG"
)
if (( INLINE_LOCAL_FALLBACK )); then
  CLOUD_CMD+=(
    --local-fallback-on-content-filter
    --local-fallback-endpoint "$LOCAL_ENDPOINT"
    --local-fallback-model "$LOCAL_MODEL"
    --local-fallback-api-format "$LOCAL_API_FORMAT"
  )
  if (( INLINE_FALLBACK_MODEL_OUTPUT_ERROR )); then
    CLOUD_CMD+=(--local-fallback-on-model-output-error)
  fi
fi
if [[ -n "$START_PDF" ]]; then
  CLOUD_CMD+=(--start-pdf "$START_PDF")
fi
if [[ -n "$END_PDF" ]]; then
  CLOUD_CMD+=(--end-pdf "$END_PDF")
fi
if [[ -n "$CLOUD_PARALLEL" ]]; then CLOUD_CMD+=(--parallel "$CLOUD_PARALLEL"); fi
if (( DRY_RUN )); then
  CLOUD_CMD+=(--dry-run)
fi
CLOUD_CMD+=(-- --workload-scan defer)

echo "[config] cloud endpoint=$OPENROUTER_ENDPOINT | model=$OPENROUTER_MODEL | provider=$OPENROUTER_PROVIDER | scheduling=$CLOUD_PARALLEL_SCHEDULING | parallel=${CLOUD_PARALLEL:-default}"
echo "[config] local endpoint=$LOCAL_ENDPOINT | model=$LOCAL_MODEL | api_format=$LOCAL_API_FORMAT | scheduling=$LOCAL_PARALLEL_SCHEDULING | parallel=${LOCAL_PARALLEL:-default}"
if (( INLINE_LOCAL_FALLBACK )); then
  if (( INLINE_FALLBACK_MODEL_OUTPUT_ERROR )); then
    echo "[config] inline fallback=enabled | triggers=provider_content_filter,model_output_error"
  else
    echo "[config] inline fallback=enabled | triggers=provider_content_filter"
  fi
else
  echo "[config] inline fallback=disabled"
fi

LOCAL_FULL_CMD=(
  ./run_ranker.sh
  --volumes "$VOLUME"
  --provider local
  --endpoint "$LOCAL_ENDPOINT"
  --model "$LOCAL_MODEL"
  --api-format "$LOCAL_API_FORMAT"
  --parallel-scheduling "$LOCAL_PARALLEL_SCHEDULING"
)
if [[ -n "$START_PDF" ]]; then  LOCAL_FULL_CMD+=(--start-pdf "$START_PDF"); fi
if [[ -n "$END_PDF" ]]; then    LOCAL_FULL_CMD+=(--end-pdf "$END_PDF"); fi
if [[ -n "$LOCAL_PARALLEL" ]]; then LOCAL_FULL_CMD+=(--parallel "$LOCAL_PARALLEL"); fi
if (( DRY_RUN )); then          LOCAL_FULL_CMD+=(--dry-run); fi
LOCAL_FULL_CMD+=(-- --workload-scan defer)

if (( CONCURRENT )); then
  echo "[concurrent] Launching cloud and local rankers simultaneously for volume $VOLUME..."
  printf '[cloud] '; printf '%q ' "${CLOUD_CMD[@]}"; printf '\n'
  printf '[local] '; printf '%q ' "${LOCAL_FULL_CMD[@]}"; printf '\n'

  "${CLOUD_CMD[@]}" &
  CLOUD_PID=$!
  "${LOCAL_FULL_CMD[@]}" &
  LOCAL_PID=$!

  CLOUD_EXIT=0
  LOCAL_EXIT=0
  wait "$CLOUD_PID" || CLOUD_EXIT=$?
  wait "$LOCAL_PID" || LOCAL_EXIT=$?

  if (( INTERRUPTED )); then
    echo "[stop] Interrupted during concurrent run."
    exit 130
  fi

  echo "[done] Concurrent run finished. cloud_exit=$CLOUD_EXIT local_exit=$LOCAL_EXIT"
  exit 0
fi

echo "[cloud] Running volume $VOLUME on OpenRouter..."
printf '[cloud] '
printf '%q ' "${CLOUD_CMD[@]}"
printf '\n'
"${CLOUD_CMD[@]}"

if (( INTERRUPTED )); then
  echo "[stop] Interrupt received during cloud pass; skipping local-retry phase."
  exit 130
fi

if (( SKIP_LOCAL_RETRY )); then
  echo "[done] Cloud pass finished (local retry skipped by flag)."
  exit 0
fi

if [[ ! -s "$FAIL_LOG" ]]; then
  echo "[done] Cloud pass finished with no failed rows to retry locally."
  exit 0
fi

LOCAL_CMD=(
  ./run_ranker.sh
  --volumes "$VOLUME"
  --provider local
  --endpoint "$LOCAL_ENDPOINT"
  --model "$LOCAL_MODEL"
  --api-format "$LOCAL_API_FORMAT"
  --parallel-scheduling "$LOCAL_PARALLEL_SCHEDULING"
  --only-source-ids-file "$FAIL_LOG"
)
if [[ -n "$START_PDF" ]]; then  LOCAL_CMD+=(--start-pdf "$START_PDF"); fi
if [[ -n "$END_PDF" ]]; then    LOCAL_CMD+=(--end-pdf "$END_PDF"); fi
if [[ -n "$LOCAL_PARALLEL" ]]; then LOCAL_CMD+=(--parallel "$LOCAL_PARALLEL"); fi
if (( DRY_RUN )); then          LOCAL_CMD+=(--dry-run); fi
LOCAL_CMD+=(-- --workload-scan defer)

echo "[local-retry] Found failed rows; retrying locally using $FAIL_LOG"
printf '[local-retry] '
printf '%q ' "${LOCAL_CMD[@]}"
printf '\n'
"${LOCAL_CMD[@]}"

if (( INTERRUPTED )); then
  echo "[stop] Interrupt received during local-retry pass."
  exit 130
fi

echo "[done] Hybrid run finished."
