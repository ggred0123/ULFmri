#!/usr/bin/env bash
# Extract/normalize downloaded ZIPs, then build Stage 1 paired slices and/or VAE
# latents from a BIDS-like NIfTI tree.
#
# The phases are intentionally separable:
#   extract : CPU + disk, Python standard library only
#   pairs   : CPU, nibabel + SimpleITK (+ torch for canvas resizing)
#   latents : CUDA GPU, torch + diffusers
#
# This makes it possible to create aligned pairs on a NIfTI-only CPU machine,
# transfer the pair directory, and encode latents on a different GPU machine.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

usage() {
  cat <<'EOF'
Usage:
  prepare_stage1_from_nifti.sh MODE [options]

MODE:
  extract    ZIP bundles or an already-unpacked download -> normalized NIfTI tree
  pairs      CPU registration, normalization and paired-slice generation
  latents    GPU VAE encoding of an existing pair directory
  all        Resolve/extract NIfTIs, then run pairs followed by latents

Required options:
  extract/pairs/all: one of --medical-root or --archive-root
  latents/all: --vae-path PATH_OR_HF_ID

Input options:
  --archive-root PATH    Directory containing the downloaded ZIPs or their
                         directly extracted archive/, kcl/, ulfenc/, webb/ trees
  --medical-root PATH    Existing normalized {kcl,ulfenc,webb} root, or extraction
                         destination when --archive-root contains ZIPs

Output options:
  --work-root PATH       Parent for default output directories
  --pairs-root PATH      Default: WORK_ROOT/data_stage1_pairs_3ori
  --latents-root PATH    Default: WORK_ROOT/data_stage1_latents_3ori
  --log-dir PATH         Default: WORK_ROOT/stage1_preprocess_logs

Runtime options:
  --prep-python PATH     Python with nibabel, SimpleITK, numpy and torch (default: python3)
  --gpu-python PATH      Python with CUDA torch and diffusers (default: python3)
  --vae-path VALUE       Local SD3 VAE directory, SD3 repository, or HF model ID
  --gpu ID               CUDA device ID exposed to the latent phase
  --batch-size N         VAE batch size (default: 4; lower this on small GPUs)

Data options:
  --datasets CSV         Default: kcl,ulfenc,webb
  --orientations CSV     Default: axial,coronal,sagittal
  --resolution N         Default: 256
  --min-fg FLOAT         Minimum target foreground fraction (default: 0.05)
  --limit N              Process at most N subjects; useful for a smoke test
  --exclude CSV          Subject IDs to skip, e.g. sub-364,sub-999

The script reuses a valid extracted NIfTI tree, and resumably extracts ZIP files
when needed. It refuses to overwrite existing pair/latent metadata; choose new
output directories when rebuilding those products.

Expected NIfTI layout (both .nii and .nii.gz are accepted):
  MEDICAL_ROOT/
    kcl/sub-200/ses-c01/anat/sub-200_ses-c01_acq-HF_T1w.nii.gz
    kcl/sub-200/ses-c01/anat/sub-200_ses-c01_acq-HF_T2w.nii.gz
    kcl/sub-200/ses-001/anat/sub-200_ses-001_acq-ULF_Axial_T1w.nii.gz
    kcl/sub-200/ses-001/anat/sub-200_ses-001_acq-ULF_Axial_T2w.nii.gz
    kcl/sub-200/ses-001/anat/..._ULF_Coronal_{T1w,T2w}.nii.gz
    kcl/sub-200/ses-001/anat/..._ULF_Sagittal_{T1w,T2w}.nii.gz

ULF-EnC and Webb use their Axial acquisition. If Webb derivatives containing
{T1,T2,FLAIR}_axi.nii.gz are present they are preferred; otherwise the raw Webb
NIfTIs are rigidly registered to the 3T T2 grid.

Examples:
  # ZIP-only download: select the correct bundles, extract, then make pairs
  bash scripts/prepare_stage1_from_nifti.sh pairs \
    --archive-root /data/new_nyu --work-root /data/ulfmri_work \
    --prep-python /opt/venvs/ulf-prep/bin/python

  # Already unpacked download: archive/, kcl/, ulfenc/, webb/ are detected and a
  # normalized no-copy view is made with symlinks before pair creation
  bash scripts/prepare_stage1_from_nifti.sh pairs \
    --archive-root /data/unpacked_download --work-root /data/ulfmri_work \
    --prep-python /opt/venvs/ulf-prep/bin/python

  # CPU machine: raw NIfTI -> paired slices
  bash scripts/prepare_stage1_from_nifti.sh pairs \
    --medical-root /data/medical --work-root /data/ulfmri_work \
    --prep-python /opt/venvs/ulf-prep/bin/python

  # GPU machine: transferred pairs -> latents (works with any CUDA GPU)
  bash scripts/prepare_stage1_from_nifti.sh latents \
    --pairs-root /data/ulfmri_work/data_stage1_pairs_3ori \
    --latents-root /data/ulfmri_work/data_stage1_latents_3ori \
    --vae-path /models/sd3/vae --gpu-python /opt/venvs/ulf-gpu/bin/python \
    --gpu 0 --batch-size 4

  # One machine with a CUDA GPU
  bash scripts/prepare_stage1_from_nifti.sh all \
    --medical-root /data/medical --work-root /data/ulfmri_work \
    --vae-path /models/sd3/vae
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

[[ $# -gt 0 ]] || { usage; exit 2; }
MODE=$1
shift
case "$MODE" in
  extract|pairs|latents|all) ;;
  -h|--help) usage; exit 0 ;;
  *) die "MODE must be one of: extract, pairs, latents, all" ;;
esac

MEDICAL_ROOT=""
ARCHIVE_ROOT=""
WORK_ROOT=""
PAIRS_ROOT=""
LATENTS_ROOT=""
LOG_DIR=""
PREP_PYTHON=python3
GPU_PYTHON=python3
VAE_PATH=""
GPU_ID=""
BATCH_SIZE=4
DATASETS_CSV=kcl,ulfenc,webb
ORIENTATIONS_CSV=axial,coronal,sagittal
RESOLUTION=256
MIN_FG=0.05
LIMIT=0
EXCLUDE_CSV=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --archive-root) [[ $# -ge 2 ]] || die "$1 needs a value"; ARCHIVE_ROOT=$2; shift 2 ;;
    --medical-root) [[ $# -ge 2 ]] || die "$1 needs a value"; MEDICAL_ROOT=$2; shift 2 ;;
    --work-root) [[ $# -ge 2 ]] || die "$1 needs a value"; WORK_ROOT=$2; shift 2 ;;
    --pairs-root) [[ $# -ge 2 ]] || die "$1 needs a value"; PAIRS_ROOT=$2; shift 2 ;;
    --latents-root) [[ $# -ge 2 ]] || die "$1 needs a value"; LATENTS_ROOT=$2; shift 2 ;;
    --log-dir) [[ $# -ge 2 ]] || die "$1 needs a value"; LOG_DIR=$2; shift 2 ;;
    --prep-python) [[ $# -ge 2 ]] || die "$1 needs a value"; PREP_PYTHON=$2; shift 2 ;;
    --gpu-python) [[ $# -ge 2 ]] || die "$1 needs a value"; GPU_PYTHON=$2; shift 2 ;;
    --vae-path) [[ $# -ge 2 ]] || die "$1 needs a value"; VAE_PATH=$2; shift 2 ;;
    --gpu) [[ $# -ge 2 ]] || die "$1 needs a value"; GPU_ID=$2; shift 2 ;;
    --batch-size) [[ $# -ge 2 ]] || die "$1 needs a value"; BATCH_SIZE=$2; shift 2 ;;
    --datasets) [[ $# -ge 2 ]] || die "$1 needs a value"; DATASETS_CSV=$2; shift 2 ;;
    --orientations) [[ $# -ge 2 ]] || die "$1 needs a value"; ORIENTATIONS_CSV=$2; shift 2 ;;
    --resolution) [[ $# -ge 2 ]] || die "$1 needs a value"; RESOLUTION=$2; shift 2 ;;
    --min-fg) [[ $# -ge 2 ]] || die "$1 needs a value"; MIN_FG=$2; shift 2 ;;
    --limit) [[ $# -ge 2 ]] || die "$1 needs a value"; LIMIT=$2; shift 2 ;;
    --exclude) [[ $# -ge 2 ]] || die "$1 needs a value"; EXCLUDE_CSV=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

if [[ -z "$WORK_ROOT" ]]; then
  if [[ -n "$PAIRS_ROOT" ]]; then
    WORK_ROOT=$(cd -- "$(dirname -- "$PAIRS_ROOT")" 2>/dev/null && pwd || dirname -- "$PAIRS_ROOT")
  elif [[ -n "$LATENTS_ROOT" ]]; then
    WORK_ROOT=$(cd -- "$(dirname -- "$LATENTS_ROOT")" 2>/dev/null && pwd || dirname -- "$LATENTS_ROOT")
  elif [[ -n "$MEDICAL_ROOT" ]]; then
    WORK_ROOT=$(cd -- "$(dirname -- "$MEDICAL_ROOT")" 2>/dev/null && pwd || dirname -- "$MEDICAL_ROOT")
  else
    die "provide --work-root, or an explicit output root"
  fi
fi

PAIRS_ROOT=${PAIRS_ROOT:-$WORK_ROOT/data_stage1_pairs_3ori}
LATENTS_ROOT=${LATENTS_ROOT:-$WORK_ROOT/data_stage1_latents_3ori}
LOG_DIR=${LOG_DIR:-$WORK_ROOT/stage1_preprocess_logs}

IFS=',' read -r -a DATASETS <<< "$DATASETS_CSV"
IFS=',' read -r -a ORIENTATIONS <<< "$ORIENTATIONS_CSV"
EXCLUDES=()
if [[ -n "$EXCLUDE_CSV" ]]; then
  IFS=',' read -r -a EXCLUDES <<< "$EXCLUDE_CSV"
fi

[[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || die "--batch-size must be a positive integer"
[[ "$RESOLUTION" =~ ^[1-9][0-9]*$ ]] || die "--resolution must be a positive integer"
[[ "$LIMIT" =~ ^[0-9]+$ ]] || die "--limit must be a non-negative integer"

mkdir -p -- "$LOG_DIR"

is_normalized_medical_root() {
  local root=$1 ds
  [[ -d "$root" ]] || return 1
  for ds in "${DATASETS[@]}"; do
    [[ -d "$root/$ds" ]] || return 1
  done
}

make_normalized_view() {
  local source_root=$1 out_root=$2
  local kcl_source ulfenc_source webb_source ds src src_abs

  # Default unzip layout contains both archive/webb (complete) and webb (partial
  # rerun). Match the known-good medical tree: named KCL, named ULF-EnC, and the
  # complete archive/Webb export.
  kcl_source="$source_root/kcl"
  [[ -d "$kcl_source" ]] || kcl_source="$source_root/archive/kcl"
  ulfenc_source="$source_root/ulfenc"
  webb_source="$source_root/archive/webb"
  [[ -d "$webb_source" ]] || webb_source="$source_root/webb"

  [[ -d "$kcl_source" && -d "$ulfenc_source" && -d "$webb_source" ]] || return 1
  if [[ -e "$out_root" ]] && find "$out_root" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    is_normalized_medical_root "$out_root" && return 0
    die "normalized medical output is non-empty but incomplete: $out_root"
  fi
  mkdir -p -- "$out_root"
  for ds in "${DATASETS[@]}"; do
    case "$ds" in
      kcl) src=$kcl_source ;;
      ulfenc) src=$ulfenc_source ;;
      webb) src=$webb_source ;;
      *) die "cannot auto-normalize unknown dataset: $ds" ;;
    esac
    src_abs=$(cd -- "$src" && pwd -P)
    ln -s -- "$src_abs" "$out_root/$ds"
  done
  echo "[extract] using already-unpacked data through: $out_root"
  return 0
}

resolve_medical_root() {
  if [[ -n "$MEDICAL_ROOT" ]] && is_normalized_medical_root "$MEDICAL_ROOT"; then
    echo "[extract] reusing normalized NIfTI tree: $MEDICAL_ROOT"
    return
  fi

  [[ -n "$ARCHIVE_ROOT" ]] || \
    die "provide an existing --medical-root or a ZIP/unpacked --archive-root"
  [[ -d "$ARCHIVE_ROOT" ]] || die "archive root is not a directory: $ARCHIVE_ROOT"

  # Common case after somebody has already run unzip: create a lightweight,
  # normalized view without copying the tens of GiB of NIfTIs again.
  local normalized=${MEDICAL_ROOT:-$WORK_ROOT/medical_from_archives}
  if make_normalized_view "$ARCHIVE_ROOT" "$normalized"; then
    MEDICAL_ROOT=$normalized
    return
  fi

  compgen -G "$ARCHIVE_ROOT/*.zip" >/dev/null || \
    die "no usable extracted dataset tree or ZIP files found in: $ARCHIVE_ROOT"
  MEDICAL_ROOT=$normalized
  command -v "$PREP_PYTHON" >/dev/null 2>&1 || [[ -x "$PREP_PYTHON" ]] || \
    die "prep Python is not executable: $PREP_PYTHON"
  local log="$LOG_DIR/extract_$(date +%Y%m%d_%H%M%S).log"
  echo "[extract] ZIP source: $ARCHIVE_ROOT"
  echo "[extract] output:     $MEDICAL_ROOT"
  echo "[extract] log:        $log"
  "$PREP_PYTHON" "$REPO_ROOT/data/extract_stage1_archives.py" \
    --archive_root "$ARCHIVE_ROOT" --out_root "$MEDICAL_ROOT" 2>&1 | tee "$log"
  is_normalized_medical_root "$MEDICAL_ROOT" || \
    die "archive extraction did not produce kcl/ulfenc/webb under: $MEDICAL_ROOT"
}

run_pairs() {
  resolve_medical_root
  [[ -d "$MEDICAL_ROOT" ]] || die "medical root is not a directory: $MEDICAL_ROOT"
  command -v "$PREP_PYTHON" >/dev/null 2>&1 || [[ -x "$PREP_PYTHON" ]] || \
    die "prep Python is not executable: $PREP_PYTHON"
  [[ ! -e "$PAIRS_ROOT/metadata.jsonl" ]] || \
    die "pair metadata already exists; choose a new --pairs-root: $PAIRS_ROOT"

  "$PREP_PYTHON" -c 'import nibabel, SimpleITK, numpy, torch' >/dev/null || \
    die "prep Python needs nibabel, SimpleITK, numpy and torch: $PREP_PYTHON"

  local cmd=(
    "$PREP_PYTHON" "$REPO_ROOT/data/preprocess_pairs.py"
    --medical_root "$MEDICAL_ROOT"
    --out_root "$PAIRS_ROOT"
    --datasets "${DATASETS[@]}"
    --orientations "${ORIENTATIONS[@]}"
    --res "$RESOLUTION"
    --min_fg "$MIN_FG"
    --limit "$LIMIT"
  )
  if [[ ${#EXCLUDES[@]} -gt 0 ]]; then
    cmd+=(--exclude "${EXCLUDES[@]}")
  fi

  local log="$LOG_DIR/pairs_$(date +%Y%m%d_%H%M%S).log"
  echo "[pairs] output: $PAIRS_ROOT"
  echo "[pairs] log:    $log"
  "${cmd[@]}" 2>&1 | tee "$log"

  [[ -s "$PAIRS_ROOT/metadata.jsonl" ]] || die "pair phase produced no metadata rows"
  local rows files
  rows=$(wc -l < "$PAIRS_ROOT/metadata.jsonl")
  files=$(find "$PAIRS_ROOT" -type f -name '*.npz' | wc -l)
  [[ "$rows" -eq "$files" ]] || \
    die "pair integrity check failed: metadata=$rows, npz=$files"
  echo "[pairs] verified $rows paired slices"
}

run_latents() {
  [[ -s "$PAIRS_ROOT/metadata.jsonl" ]] || \
    die "pair metadata not found: $PAIRS_ROOT/metadata.jsonl"
  [[ -n "$VAE_PATH" ]] || die "--vae-path is required for MODE=$MODE"
  command -v "$GPU_PYTHON" >/dev/null 2>&1 || [[ -x "$GPU_PYTHON" ]] || \
    die "GPU Python is not executable: $GPU_PYTHON"
  [[ ! -e "$LATENTS_ROOT/metadata_latents.jsonl" ]] || \
    die "latent metadata already exists; choose a new --latents-root: $LATENTS_ROOT"

  if [[ -n "$GPU_ID" ]]; then
    export CUDA_VISIBLE_DEVICES=$GPU_ID
  fi
  "$GPU_PYTHON" -c \
    'import torch, diffusers; assert torch.cuda.is_available(), "CUDA is unavailable"' >/dev/null || \
    die "GPU Python needs CUDA-enabled torch and diffusers: $GPU_PYTHON"

  local log="$LOG_DIR/latents_$(date +%Y%m%d_%H%M%S).log"
  echo "[latents] output: $LATENTS_ROOT"
  echo "[latents] log:    $log"
  "$GPU_PYTHON" "$REPO_ROOT/data/precompute_latents_pairs.py" \
    --data_root "$PAIRS_ROOT" \
    --out_root "$LATENTS_ROOT" \
    --vae_path "$VAE_PATH" \
    --batch_size "$BATCH_SIZE" 2>&1 | tee "$log"

  [[ -s "$LATENTS_ROOT/metadata_latents.jsonl" ]] || \
    die "latent phase produced no metadata rows"
  local rows files expected
  expected=$(wc -l < "$PAIRS_ROOT/metadata.jsonl")
  rows=$(wc -l < "$LATENTS_ROOT/metadata_latents.jsonl")
  files=$(find "$LATENTS_ROOT" -type f -name '*.npz' | wc -l)
  [[ "$rows" -eq "$expected" && "$files" -eq "$expected" ]] || \
    die "latent integrity check failed: expected=$expected, metadata=$rows, npz=$files"
  echo "[latents] verified $rows latent pairs"
}

case "$MODE" in
  extract) resolve_medical_root ;;
  pairs) run_pairs ;;
  latents) run_latents ;;
  all) run_pairs; run_latents ;;
esac

echo "Done."
if [[ "$MODE" == "extract" ]]; then
  echo "  medical: $MEDICAL_ROOT"
  exit 0
fi
echo "  pairs:   $PAIRS_ROOT"
if [[ "$MODE" != "pairs" ]]; then
  echo "  latents: $LATENTS_ROOT"
fi
