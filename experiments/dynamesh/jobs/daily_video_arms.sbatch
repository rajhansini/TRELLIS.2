#!/bin/bash
# One object's four daily panel videos with an explicit arm list (batch F/G tags).
# CPU only; one job per object, never a loop over objects inside one job.
# PATH puts richards_distillation's ffmpeg first: it is the only one on shared storage
# with libx264 (trellis2's has openh264 only and exits 8 after a 20-minute composite).
# Pass OBJ, optionally ARMS (default r19,r27,r27mcfm).
#SBATCH --job-name=dva
#SBATCH --partition=general,threedle-contrib,threedle-own
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/dva_%x_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/dva_%x_%j.log
set -o pipefail   # NOT -e: one failed view must not kill the other three
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
PY2=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
export PATH="/net/projects/ranalab/rajhansini/conda_envs/richards_distillation/bin:$PATH"
cd "$E"
: "${OBJ:?}"
echo "=== $(date) host=$(hostname) OBJ=${OBJ} ARMS=${ARMS:-r19,r27,r27mcfm} ==="
HAVE_X264=$(ffmpeg -hide_banner -encoders 2>/dev/null | grep -c libx264 || true)
echo "ffmpeg=$(command -v ffmpeg)  libx264 entries=${HAVE_X264:-0}"
[ "${HAVE_X264:-0}" -ge 1 ] || { echo "FAILED: ffmpeg on PATH has no libx264"; exit 1; }
${PY2} -u "$E/jobs/build_daily_video_arms.py" "${OBJ}" --arms "${ARMS:-r19,r27,r27mcfm}" 2>&1 \
  | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn"
echo "=== done $(date) ==="
ls "$E/out/DAILY/${OBJ}"_*.mp4 2>/dev/null | wc -l | xargs -I{} echo "{}/4 daily videos present for ${OBJ}"
