#!/bin/bash
#SBATCH -t 4:00:00
#SBATCH --gpus-per-node=a100:2
#SBATCH --mem=500G
#SBATCH -c 16
#SBATCH -p gpu,uri-gpu,gpu-preempt
#SBATCH -o gpu_test.%j.out
#SBATCH -e gpu_test.%j.err
#SBATCH --mail-user=samuel.clyne@uri.edu
#SBATCH --mail-type=ALL

set -euo pipefail

export TMP=/dev/shm
export TMPDIR=/dev/shm
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

module purge
module load uri/main
module load GSL/2.7-GCC-11.3.0
module load GCC/11.3.0
module load CUDA/12.1.1
module load FFTW/3.3.10-GCC-11.3.0
module load Python/3.10.4-GCCcore-11.3.0
module load conda/latest

conda activate lisa_310_build
export PYTHONPATH=/work/pi_mpuerrer_uri_edu/Sam/BBHx${PYTHONPATH:+:$PYTHONPATH}

git -C /work/pi_mpuerrer_uri_edu/Sam/dingo checkout LISA-dev
git -C /work/pi_mpuerrer_uri_edu/Sam/dingo pull --ff-only
cd /work/pi_mpuerrer_uri_edu/Sam/dingo

python - <<'PY'
import bbhx
import torch

print("BBHx:", bbhx.__file__)
print("CUDA available:", torch.cuda.is_available(), "count=", torch.cuda.device_count())
PY
