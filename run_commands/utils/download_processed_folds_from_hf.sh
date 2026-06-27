#!/usr/bin/env bash
set -euo pipefail

REPO_ID="${1:-${HF_DATASET_REPO:-}}"
LOCAL_DIR="${2:-.}"

if [[ -z "$REPO_ID" ]]; then
  echo "Usage: $0 <hf_user/eyebench-processed-folds> [local_dir]" >&2
  echo "Or set HF_DATASET_REPO." >&2
  exit 1
fi

if ! command -v hf >/dev/null 2>&1; then
  echo "hf CLI is not installed. Install with: pip install -U 'huggingface_hub[cli,hf_xet]'" >&2
  exit 1
fi

hf download "$REPO_ID" \
  --repo-type dataset \
  --local-dir "$LOCAL_DIR" \
  --include 'README.md' \
  --include 'manifest_sizes.txt' \
  --include 'manifest_files.txt' \
  --include 'data/*/processed/**' \
  --include 'data/*/folds/**' \
  --include 'data/*/folds_metadata/**'
