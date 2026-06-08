#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$SCRIPT_DIR}"
DATASET_SLUG="${DATASET_SLUG:-bnthanh/rukopys-dataset}"
SILVER_SLUG="${SILVER_SLUG:-}"
PHASE1_SLUG="${PHASE1_SLUG:-}"
FORCE_DOWNLOAD_RAW="${FORCE_DOWNLOAD_RAW:-0}"
USE_LOCAL_RAW="${USE_LOCAL_RAW:-0}"

cd "$ROOT"

if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  . .venv/bin/activate
fi

if ! command -v kaggle >/dev/null 2>&1; then
  python -m pip install kaggle
fi

if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
  if [ -f "$ROOT/kaggle.json" ]; then
    mkdir -p "$HOME/.kaggle"
    cp "$ROOT/kaggle.json" "$HOME/.kaggle/kaggle.json"
    echo "Copied $ROOT/kaggle.json -> $HOME/.kaggle/kaggle.json"
  else
    cat >&2 <<MSG
Missing $HOME/.kaggle/kaggle.json

Create a Kaggle API token from Kaggle Account settings, then put it at:
  $HOME/.kaggle/kaggle.json

Or place it temporarily at:
  $ROOT/kaggle.json

Then run:
  chmod 600 $HOME/.kaggle/kaggle.json
  cd $ROOT
  bash setup_kaggle_dataset.sh
MSG
    exit 2
  fi
fi

chmod 600 "$HOME/.kaggle/kaggle.json"

LOCAL_RUKOPYS="$ROOT/rukopys_raw"
KAGGLE_DATASET_DIR="$ROOT/data/kaggle_input/rukopys-dataset"
RUKOPYS_ROOT=""

has_raw_dataset() {
  local dir="$1"
  [ -f "$dir/train/metadata.jsonl" ] \
    && [ -d "$dir/train/images" ] \
    && [ -f "$dir/test/metadata.jsonl" ] \
    && [ -d "$dir/test/images" ]
}

normalize_download_root() {
  local dir="$1"
  if [ -d "$dir/rukopys_raw" ]; then
    echo "$dir/rukopys_raw"
  else
    echo "$dir"
  fi
}

if [ "$USE_LOCAL_RAW" = "1" ] && [ "$FORCE_DOWNLOAD_RAW" != "1" ] && has_raw_dataset "$LOCAL_RUKOPYS"; then
  RUKOPYS_ROOT="$LOCAL_RUKOPYS"
  echo "Using local raw dataset: $RUKOPYS_ROOT"
elif [ "$FORCE_DOWNLOAD_RAW" != "1" ] && has_raw_dataset "$(normalize_download_root "$KAGGLE_DATASET_DIR")"; then
  RUKOPYS_ROOT="$(normalize_download_root "$KAGGLE_DATASET_DIR")"
  echo "Using existing Kaggle dataset: $RUKOPYS_ROOT"
else
  mkdir -p "$KAGGLE_DATASET_DIR"
  kaggle datasets download -d "$DATASET_SLUG" \
    -p "$KAGGLE_DATASET_DIR" \
    --unzip
  RUKOPYS_ROOT="$(normalize_download_root "$KAGGLE_DATASET_DIR")"

  if [ "$USE_LOCAL_RAW" = "1" ] && [ -d "$LOCAL_RUKOPYS" ] && has_raw_dataset "$LOCAL_RUKOPYS" ]; then
    RUKOPYS_ROOT="$LOCAL_RUKOPYS"
  fi
fi

if ! has_raw_dataset "$RUKOPYS_ROOT"; then
  echo "Required raw dataset structure not found under: $RUKOPYS_ROOT" >&2
  find "$RUKOPYS_ROOT" -maxdepth 4 -type d | sort >&2 || true
  exit 3
fi

if [ ! -f "$RUKOPYS_ROOT/silver/metadata.jsonl" ] && [ -n "$SILVER_SLUG" ]; then
  mkdir -p data/kaggle_input/rukopys-silver
  kaggle datasets download -d "$SILVER_SLUG" \
    -p data/kaggle_input/rukopys-silver \
    --unzip

  SILVER_ROOT="$(normalize_download_root "$ROOT/data/kaggle_input/rukopys-silver")"
  if [ -d "$SILVER_ROOT/silver" ]; then
    SILVER_ROOT="$SILVER_ROOT/silver"
  fi

  if [ -f "$SILVER_ROOT/metadata.jsonl" ] && [ -d "$SILVER_ROOT/images" ]; then
    if [ -e "$RUKOPYS_ROOT/silver" ]; then
      echo "Silver target already exists but metadata was not found: $RUKOPYS_ROOT/silver" >&2
      echo "Move or remove that path, then rerun setup." >&2
      exit 4
    fi
    ln -s "$SILVER_ROOT" "$RUKOPYS_ROOT/silver"
    echo "Linked silver data: $RUKOPYS_ROOT/silver -> $SILVER_ROOT"
  else
    echo "Downloaded SILVER_SLUG but did not find metadata.jsonl + images under: $SILVER_ROOT" >&2
    find "$ROOT/data/kaggle_input/rukopys-silver" -maxdepth 4 -type d | sort >&2 || true
    exit 4
  fi
fi

if [ ! -f "$RUKOPYS_ROOT/silver/metadata.jsonl" ]; then
  echo "[warn] No silver/metadata.jsonl found. Silver phases will be skipped."
fi

if [ -n "$PHASE1_SLUG" ]; then
  mkdir -p data/kaggle_input/phase1-artifacts
  kaggle datasets download -d "$PHASE1_SLUG" \
    -p data/kaggle_input/phase1-artifacts \
    --unzip
fi

cat > .env.kaggle <<EOF
export PROJECT_ROOT=$ROOT
export INPUT_ROOT=$ROOT/data/kaggle_input
export WORK_ROOT=$ROOT/local_working
export HTR_ART_DIR=$ROOT/local_working/htr_artifacts
export RUKOPYS_ROOT=$RUKOPYS_ROOT
export DATA_ROOT=$RUKOPYS_ROOT
export HF_HOME=$ROOT/local_working/hf_cache
export TRANSFORMERS_CACHE=$ROOT/local_working/hf_cache
export USE_SILVER=${USE_SILVER:-1}
export MAX_SILVER=${MAX_SILVER:-80000}
export USE_SYNTH=${USE_SYNTH:-1}
export USE_HKR=${USE_HKR:-1}
export USE_PSEUDO=${USE_PSEUDO:-0}
export CURRICULUM=${CURRICULUM:-0}
export USE_FLASH_ATTN=${USE_FLASH_ATTN:-auto}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=false
export YOLO_CONFIG_DIR=$HOME/.config/Ultralytics
export PYTHONPATH=$ROOT:$ROOT/scripts:\${PYTHONPATH:-}
EOF

echo "Kaggle dataset ready."
echo "RUKOPYS_ROOT=$RUKOPYS_ROOT"
echo "train rows: $(wc -l < "$RUKOPYS_ROOT/train/metadata.jsonl")"
echo "test rows:  $(wc -l < "$RUKOPYS_ROOT/test/metadata.jsonl")"
if [ -f "$RUKOPYS_ROOT/silver/metadata.jsonl" ]; then
  echo "silver rows: $(wc -l < "$RUKOPYS_ROOT/silver/metadata.jsonl")"
else
  echo "silver rows: 0 (optional)"
fi
echo
echo "Before running scripts:"
echo "  cd $ROOT"
echo "  source .venv/bin/activate"
echo "  source .env.kaggle"
echo "  python scripts/show_run_config.py"
