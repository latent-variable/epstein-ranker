#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VOLUME=""
START_PDF=""
END_PDF=""
DRY_RUN=0
SKIP_LOCAL_RETRY=0
INTERRUPTED=0

trap 'INTERRUPTED=1' INT

OPENROUTER_MODEL="${OPENROUTER_MODEL:-qwen/qwen3-vl-30b-a3b-thinking}"
OPENROUTER_PROVIDER="${OPENROUTER_PROVIDER:-alibaba}"
CLOUD_PARALLEL_SCHEDULING="${CLOUD_PARALLEL_SCHEDULING:-window}"
LOCAL_PARALLEL_SCHEDULING="${LOCAL_PARALLEL_SCHEDULING:-batch}"

usage() {
  cat <<'USAGE'
Usage:
  ./run_hybrid_volume.sh --volume N [options]

Required:
  --volume N                 DOJ volume number (example: 11)

Options:
  --start-pdf N              Start PDF index (1-based, pre-split)
  --end-pdf N                End PDF index (inclusive, pre-split)
  --openrouter-model ID      Cloud model (default: qwen/qwen3-vl-30b-a3b-thinking)
  --openrouter-provider ID   OpenRouter provider (default: alibaba)
  --cloud-scheduling MODE    window | batch (default: window)
  --local-scheduling MODE    window | batch (default: batch)
  --skip-local-retry         Do cloud pass only
  --dry-run                  Print commands only
  -h, --help                 Show help

What it does:
  1) Runs OpenRouter for the target volume with immediate local fallback on provider
     content-filter blocks, and logs unresolved failures to:
     data/workspaces/standardworks_epstein_files_volXXXXX/metadata/failed_requests_openrouter.jsonl
  2) If that log has entries, reruns only those unresolved failed source IDs locally.
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
    --openrouter-provider)
      OPENROUTER_PROVIDER="${2:-}"
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

VOL_TAG="$(printf "standardworks_epstein_files_vol%05d" "$VOLUME")"
FAIL_LOG="data/workspaces/${VOL_TAG}/metadata/failed_requests_openrouter.jsonl"

mkdir -p "$(dirname "$FAIL_LOG")"
# Keep cumulative failure history so interrupted runs do not lose retry candidates.
touch "$FAIL_LOG"

CLOUD_CMD=(
  ./run_ranker.sh
  --volumes "$VOLUME"
  --provider openrouter
  --openrouter-model "$OPENROUTER_MODEL"
  --openrouter-provider "$OPENROUTER_PROVIDER"
  --parallel-scheduling "$CLOUD_PARALLEL_SCHEDULING"
  --failure-log "$FAIL_LOG"
)
if [[ -n "$START_PDF" ]]; then
  CLOUD_CMD+=(--start-pdf "$START_PDF")
fi
if [[ -n "$END_PDF" ]]; then
  CLOUD_CMD+=(--end-pdf "$END_PDF")
fi
if (( DRY_RUN )); then
  CLOUD_CMD+=(--dry-run)
fi
CLOUD_CMD+=(-- --workload-scan defer)

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
  --parallel-scheduling "$LOCAL_PARALLEL_SCHEDULING"
  --only-source-ids-file "$FAIL_LOG"
)
if [[ -n "$START_PDF" ]]; then
  LOCAL_CMD+=(--start-pdf "$START_PDF")
fi
if [[ -n "$END_PDF" ]]; then
  LOCAL_CMD+=(--end-pdf "$END_PDF")
fi
if (( DRY_RUN )); then
  LOCAL_CMD+=(--dry-run)
fi
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
