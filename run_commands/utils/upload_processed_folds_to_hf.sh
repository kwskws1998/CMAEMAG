#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

REPO_ID="${1:-${HF_DATASET_REPO:-}}"
REPO_NAME="${HF_DATASET_REPO_NAME:-eyebench-processed-folds}"
NUM_WORKERS="${HF_UPLOAD_NUM_WORKERS:-8}"
DRY_RUN="${DRY_RUN:-0}"
STAGING_DIR="${HF_UPLOAD_STAGING_DIR:-${ROOT_DIR}/../eyebench_processed_folds_hf_stage}"
REBUILD_STAGING="${HF_REBUILD_STAGING:-0}"
KEEP_STAGING="${HF_KEEP_STAGING:-1}"

if ! command -v hf >/dev/null 2>&1; then
  echo "hf CLI is not installed. Install with: pip install -U 'huggingface_hub[hf_xet]<1.0,>=0.24.0'" >&2
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

MANIFEST_FILE_COUNT="$(find data -type f \( -path '*/processed/*' -o -path '*/folds/*' -o -path '*/folds_metadata/*' \) ! -name '.DS_Store' -print | wc -l | tr -d ' ')"

echo "Target HF dataset repo: $REPO_ID"
echo "Upload size estimate:"
du -ch data/*/processed data/*/folds data/*/folds_metadata 2>/dev/null | tail -n 1
echo "Manifest file count: $MANIFEST_FILE_COUNT"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1, not creating repo or uploading."
  exit 0
fi

if [[ "$REBUILD_STAGING" == "1" || ! -d "$STAGING_DIR/data" ]]; then
  rm -rf "$STAGING_DIR"
  mkdir -p "$STAGING_DIR"
  python - "$ROOT_DIR" "$STAGING_DIR" <<'PYSTAGE'
import os
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1])
staging = Path(sys.argv[2])
datasets = ["CopCo", "IITBHGC", "MECOL2", "MECOL2W1", "MECOL2W2", "OneStop", "PoTeC", "SBSAT"]
subdirs = ["processed", "folds", "folds_metadata"]
linked = 0
copied = 0

for dataset in datasets:
    for subdir in subdirs:
        src_root = root / "data" / dataset / subdir
        dst_root = staging / "data" / dataset / subdir
        for src in src_root.rglob("*"):
            rel = src.relative_to(src_root)
            dst = dst_root / rel
            if src.is_dir():
                dst.mkdir(parents=True, exist_ok=True)
                continue
            if src.name == ".DS_Store":
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                dst.unlink()
            try:
                os.link(src, dst)
                linked += 1
            except OSError:
                shutil.copy2(src, dst)
                copied += 1

print(f"Prepared staging directory: {staging}")
print(f"Hardlinked files: {linked}")
print(f"Copied files: {copied}")
PYSTAGE
else
  echo "Reusing existing staging directory: $STAGING_DIR"
  echo "Set HF_REBUILD_STAGING=1 to rebuild it."
fi

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
  echo "Excluded paths: raw downloads, precomputed intermediate files, feature caches, model outputs, logs, sweep outputs, and .DS_Store files."
  echo
  echo "Cloud download example:"
  echo
  echo '```bash'
  echo "pip install -U 'huggingface_hub[hf_xet]<1.0,>=0.24.0'"
  echo "hf auth login"
  printf "bash run_commands/utils/download_processed_folds_from_hf.sh %s .\n" "$REPO_ID"
  echo '```'
  echo
  echo "The repository helper calls the Hugging Face download command once per include pattern for compatibility with CLI versions that do not honor repeated --include flags in a single command."
} > "$STAGING_DIR/README.md"

du -sh data/*/processed data/*/folds data/*/folds_metadata 2>/dev/null | sort -k2 > "$STAGING_DIR/manifest_sizes.txt"
find data -type f \
  \( -path '*/processed/*' -o -path '*/folds/*' -o -path '*/folds_metadata/*' \) \
  ! -name '.DS_Store' \
  -print | sort > "$STAGING_DIR/manifest_files.txt"

if hf repos create --help >/dev/null 2>&1; then
  hf repos create "$REPO_ID" --type dataset --private --exist-ok
else
  hf repo create "$REPO_ID" --repo-type dataset --private --exist-ok
fi

hf upload-large-folder "$REPO_ID" "$STAGING_DIR" \
  --repo-type dataset \
  --num-workers "$NUM_WORKERS" \
  --exclude '**/.DS_Store'

if [[ "$KEEP_STAGING" != "1" ]]; then
  rm -rf "$STAGING_DIR"
fi
