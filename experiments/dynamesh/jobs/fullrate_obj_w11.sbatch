#!/bin/bash
# Full-rate (all 150 frames) video-space temporal metrics for ONE object.
#
# CPU ONLY -- NO --gres. The work is 2,550 PNG reads over NFS per object; the
# measured cost is ~1.2 s/frame of read latency against ~0 s of compute. A GPU
# would sit idle and would burn one of the scarce a40/L40S slots the trellis2
# training fleet needs. --cpus-per-task=8 plus a 16-thread loader inside the
# script is what actually moves this: the parallelism has to be in flight I/O.
#
# WHY THIS EXISTS. eval_video_metrics.py resamples all seven methods onto
# SV4D2's 21 instants, because SV4D2 only ever produced 21 frames. Flicker is a
# per-step rate, so that stride-7 clock inflates every number AND inflates ours
# more than frozen (measured 5.4x vs 4.2x) -- it erases exactly the frame-to-frame
# smoothness MCFM buys. This job measures the four methods that natively run 150
# frames on their own clock. SV4D2 (21) and DG4D (32) cannot join and stay in the
# 21-instant table.
#
# SELF-VALIDATING. Each object also recomputes the 21-instant numbers from the
# same loaded frames; jobs/fullrate_table.py diffs those against
# baselines4d/results/video_metrics_all.json and refuses to emit a table if they
# disagree. A silent regression in the loader cannot reach the paper.
#SBATCH --job-name=frw
#SBATCH --partition=threedle-contrib,threedle-own,general
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/fr_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/fr_%j.log
set -eo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
PY2=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
: "${OBJ:?}"
mkdir -p "${E}/out/FULLRATE"
OUTJ="${E}/out/FULLRATE_W11/${OBJ}.json"
if [ -s "$OUTJ" ]; then echo "ALREADY DONE: $OUTJ"; exit 0; fi
echo "=== $(date) host=$(hostname) OBJ=${OBJ} cpus=${SLURM_CPUS_PER_TASK} ==="
${PY2} -u "${E}/jobs/fullrate_metrics_w11.py" --obj "${OBJ}" --out "${OUTJ}"
[ -s "$OUTJ" ] || { echo "FAILED: no json written"; exit 1; }
echo "=== Done $(date) ==="
