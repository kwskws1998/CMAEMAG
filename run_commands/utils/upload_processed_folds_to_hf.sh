#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

REPO_ID="${1:-${HF_DATASET_REPO:-}}"
REPO_NAME="${HF_DATASET_REPO_NAME:-eyebench-processed-folds}"
NUM_WORKERS="${HF_UPLOAD_NUM_WORKERS:-8}"
DRY_RUN="${DRY_RUN:-0}"

if ! command -v hf >/dev/null 2>&1; then
  echo "hf CLI is not installed. Install with: pip install -U 'huggingface_hub[cli,hf_xet]'" >&2
  exit 1
fi

if [[ "$DRY_RUN" != "1" ]]; then
  if ! hf auth whoami >/dev/null 2>&1; then
    echo "Not logged in to Hugging Face. Run: hf auth login" >&2
    exit 1
  fi
fi

if [[ -z "$REPO_ID" ]]; then
  if [[ "$DRY_RUN" == "1" ]]; then
    REPO_ID="dry-run/${REPO_NAME}"
  else
    HF_USER="$(python - <<'PYHF'
from huggingface_hub import whoami
print(whoami()["name"])
PYHF
)"
    REPO_ID="${HF_USER}/${REPO_NAME}"
  fi
fi

DATASETS=(CopCo IITBHGC MECOL2 MECOL2W1 MECOL2W2 OneStop PoTeC SBSAT)
SUBDIRS=(processed folds folds_metadata)
missing=0
for dataset in "${DATASETS[@]}"; do
  for subdir in "${SUBDIRS[@]}"; do
    path="data/${dataset}/${subdir}"
    if [[ ! -d "$path" ]]; then
      echo "Missing required directory: $path" >&2
      missing=1
    fi
  done
done
if [[ "$missing" -ne 0 ]]; then
  exit 1
fi

CARD_DIR="${TMPDIR:-/tmp}/eyebench_processed_folds_hf_card"
rm -rf "$CARD_DIR"
mkdir -p "$CARD_DIR"

{
  echo "# EyeBench processed folds"
  echo
  echo "This private working dataset contains only the preprocessed EyeBench files needed to train and benchmark models without re-running raw data download or preprocessing."
  echo
  echo "Included paths:"
  echo
  echo "- data/*/processed/"
  echo "- data/*/folds/"
  echo "- data/*/folds_metadata/"
  echo
  echo "Excluded paths: raw downloads, precomputed intermediate files, feature caches, model outputs, logs, and sweep outputs."
  echo
  echo "Cloud download example:"
  echo
  echo '```bash'
  echo "pip install -U 'huggingface_hub[cli,hf_xet]'"
  echo "hf auth login"
  printf "hf download %s --repo-type dataset --local-dir . \\\n  --include 'data/*/processed/**' \\\n  --include 'data/*/folds/**' \\\n  --include 'data/*/folds_metadata/**'\n" "$REPO_ID"
  echo '```'
} > "$CARD_DIR/README.md"

du -sh data/*/processed data/*/folds data/*/folds_metadata 2>/dev/null | sort -k2 > "$CARD_DIR/manifest_sizes.txt"
find data -type f \
  \( -path '*/processed/*' -o -path '*/folds/*' -o -path '*/folds_metadata/*' \) \
  -print | sort > "$CARD_DIR/manifest_files.txt"

echo "Target HF dataset repo: $REPO_ID"
echo "Upload size estimate:"
du -ch data/*/processed data/*/folds data/*/folds_metadata 2>/dev/null | tail -n 1

echo "Manifest file count: $(wc -l < "$CARD_DIR/manifest_files.txt" | tr -d ' ')"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1, not creating repo or uploading."
  exit 0
fi

if hf repos create --help >/dev/null 2>&1; then
  hf repos create "$REPO_ID" --type dataset --private --exist-ok
else
  hf repo create "$REPO_ID" --repo-type dataset --private --exist-ok
fi

hf upload "$REPO_ID" "$CARD_DIR/README.md" README.md \
  --repo-type dataset \
  --commit-message "Add processed-folds dataset card"

hf upload "$REPO_ID" "$CARD_DIR/manifest_sizes.txt" manifest_sizes.txt \
  --repo-type dataset \
  --commit-message "Add processed-folds size manifest"

hf upload "$REPO_ID" "$CARD_DIR/manifest_files.txt" manifest_files.txt \
  --repo-type dataset \
  --commit-message "Add processed-folds file manifest"

hf upload-large-folder "$REPO_ID" . \
  --repo-type dataset \
  --num-workers "$NUM_WORKERS" \
  --include 'data/*/processed/**' \
  --include 'data/*/folds/**' \
  --include 'data/*/folds_metadata/**' \
  --exclude 'data/*/downloads/**' \
  --exclude 'data/*/precomputed_events/**' \
  --exclude 'data/*/precomputed_reading_measures/**' \
  --exclude 'data/cache/**'
