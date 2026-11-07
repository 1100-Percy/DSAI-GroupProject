#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0 PYTHONUTF8=1 TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
export PYTORCH_ENABLE_MPS_FALLBACK=1
export HF_HOME="$PWD/.cache/huggingface" GENSIM_DATA_DIR="$PWD/gensim-data"
export NLTK_DATA="$PWD/.nltk_data" HF_HUB_DISABLE_XET=1
export XDG_CACHE_HOME="$PWD/.cache" MPLCONFIGDIR="$PWD/.cache/matplotlib"
export IPYTHONDIR="$PWD/.cache/ipython" PIP_CACHE_DIR="$PWD/.cache/pip"
mkdir -p "$MPLCONFIGDIR" "$IPYTHONDIR" "$PIP_CACHE_DIR"
exec .venv/bin/python -m stage1.run "$@"
