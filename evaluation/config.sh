# Settings shared by the scripts in run/, which source this file. Export a variable first to override it.
export EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PRESET_DATA="${PRESET_DATA:-$EVAL/data}"           # raw data, layout in data/README.md
export PRESET_RESULTS="${PRESET_RESULTS:-$EVAL/results}"  # every evaluation writes here, every figure and table script reads here
export PRESET_OUTPUT="${PRESET_OUTPUT:-$EVAL/output}"     # figures and LaTeX tables written by paper/*.py
export PYTHON="${PYTHON:-python}"
export TXDE_RSCRIPT="${TXDE_RSCRIPT:-Rscript}"            # R 4.4 with the packages of install_R_packages.R (R_LIBS may point to their library)
# Paper setting for every comparison: Bioconductor DESeq2, no shared gene pre-filter (each method applies its own filter).
export TXDE_DESEQ2="${TXDE_DESEQ2:-R}" TXDE_PREFILTER="${TXDE_PREFILTER:-none}"
# One BLAS/OpenMP thread per worker; parallelism comes from the --jobs workers of each script (JOBS below).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-}"   # the evaluations ran on CPU (--gpu -1)
export JOBS="${JOBS:-16}"                                 # parallel workers per script (the paper used 8-40)
# Released weights (preset/weights/preset_v1.0.pt), used for every PRESET result in the paper.
export CKPT="${CKPT:-$($PYTHON -c 'from preset.weights import resolve_weights; print(resolve_weights(None))')}"
mkdir -p "$PRESET_RESULTS" "$PRESET_OUTPUT"

# run <name> <category/script.py> [args...]: runs analysis/<category/script.py> with --output_dir $PRESET_RESULTS/<name>, log in <name>/${LOG:-run}.log
run() {
  local name=$1; shift; local out="$PRESET_RESULTS/$name"; local log="$out/${LOG:-run}.log"; mkdir -p "$out"
  echo "[$(date +%H:%M)] $name: $*"
  (cd "$EVAL/analysis" && $PYTHON -u "$@" --output_dir "$out") > "$log" 2>&1 || { echo "  FAILED, see $log"; return 1; }
}
# py <log name> <category/script.py> [args...]: runs analysis/<category/script.py> as given (no --output_dir), log in $PRESET_RESULTS/<log name>.log
py() { local log="$PRESET_RESULTS/$1.log"; shift; echo "[$(date +%H:%M)] $*"; (cd "$EVAL/analysis" && $PYTHON -u "$@") > "$log" 2>&1 || { echo "  FAILED, see $log"; return 1; }; }
