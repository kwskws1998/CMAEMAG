#!/usr/bin/env bash
set -euo pipefail

REPO_ID="${1:-${HF_DATASET_REPO:-}}"
LOCAL_DIR="${2:-.}"
DATASET_PATTERN="${3:-${HF_DATASET_NAME:-*}}"

if [[ -z "$REPO_ID" ]]; then
  echo "Usage: $0 <hf_user/eyebench-processed-folds> [local_dir] [dataset_name|*]" >&2
  echo "Or set HF_DATASET_REPO." >&2
  exit 1
fi

if ! command -v hf >/dev/null 2>&1; then
  echo "hf CLI is not installed. Install with: pip install -U 'huggingface_hub[hf_xet]<1.0,>=0.24.0'" >&2
  exit 1
fi

INCLUDE_PATTERNS=(
  'README.md'
  'manifest_sizes.txt'
  'manifest_files.txt'
  "data/${DATASET_PATTERN}/processed/*"
  "data/${DATASET_PATTERN}/folds/fold_*/*"
  "data/${DATASET_PATTERN}/folds_metadata/*/*"
)

for include_pattern in "${INCLUDE_PATTERNS[@]}"; do
  hf download "$REPO_ID" \
    --repo-type dataset \
    --local-dir "$LOCAL_DIR" \
    --include "$include_pattern"
done
