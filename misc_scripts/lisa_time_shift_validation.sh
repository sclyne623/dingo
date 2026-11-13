#!/bin/bash -l
# Cluster validation of the LISA coalescence-time shift (DINGO branch
# claude/lisa-free-time). CPU only, a few minutes.
#
# The branch is imported from its own worktree via PYTHONPATH, so the shared
# checkout (/work/pi_mpuerrer_uri_edu/Sam/dingo) is NOT switched and running
# training/IS jobs are unaffected. One-time setup, once the branch is on Unity:
#   git -C /work/pi_mpuerrer_uri_edu/Sam/dingo worktree add \
#       /work/pi_mpuerrer_uri_edu/Sam/dingo-lisa-time claude/lisa-free-time
#
# Usage:
#   sbatch lisa_time_shift_validation.sh
#   sbatch lisa_time_shift_validation.sh --n-draws 10 --model <path/to/model.pt>
# Exit code 0 = all checks passed; the JSON report path is printed at the end.

#SBATCH -t 1:00:00
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH --constraint=avx512
#SBATCH -p cpu,cpu-preempt
#SBATCH -o lisa_time_shift_validation.%j.out
#SBATCH -e lisa_time_shift_validation.%j.err

export TMP=/dev/shm
export TMPDIR=/dev/shm

module load uri/main GSL/2.7-GCC-11.3.0 GCC/11.3.0 FFTW/3.3.10-GCC-11.3.0 Python/3.10.4-GCCcore-11.3.0-bare
source /work/pi_mpuerrer_uri_edu/Sam/lisabeta_env/bin/activate

DINGO_WT=${DINGO_WT:-/work/pi_mpuerrer_uri_edu/Sam/dingo-lisa-time}
if [ ! -f "$DINGO_WT/dingo/gw/lisa_time.py" ]; then
    echo "No dingo/gw/lisa_time.py under $DINGO_WT; create the worktree first (see header)." >&2
    exit 2
fi
export PYTHONPATH="$DINGO_WT${PYTHONPATH:+:$PYTHONPATH}"
echo "dingo worktree: $DINGO_WT @ $(git -C "$DINGO_WT" log --oneline -1)"

cd /work/pi_mpuerrer_uri_edu/projects/LISA_npe
srun python "$DINGO_WT/misc_scripts/lisa_time_shift_validation.py" "$@"
