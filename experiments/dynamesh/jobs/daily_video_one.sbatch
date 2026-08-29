#!/bin/bash
# daily_video_one.sbatch -- one object's four daily panel videos (train + 3 diagonals).
# CPU only; one job per object, never a loop over objects inside one job.
#
# FFMPEG. build_daily_video.py calls a bare `ffmpeg`. Three exist on this cluster and
# only two can encode h264:
#   /usr/bin/ffmpeg                        libx264 YES -- LOGIN NODE ONLY
#   conda_envs/trellis2/bin/ffmpeg         libx264 NO  -- openh264 only, exits 8
#   conda_envs/richards_distillation/bin   libx264 YES -- shared storage, all nodes
# So the third is put on PATH ahead of everything. Without this the composite runs for
# 20 minutes and dies at the encode step, which is exactly how the batch-C panel jobs
# failed five times.
#
# NO_R27=1 -> three-column daily (GT | frozen | rung27+MCFM), for objects trained on
# the MCFM arm only. Pass OBJ, optionally NO_R27.
#SBATCH --job-name=dv
#SBATCH --partition=general,threedle-contrib,threedle-own
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/dv_%x_%j.log
#SBATCH --error=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/out/dv_%x_%j.log
set -o pipefail   # NOT -e: one failed view must not kill the other three
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
PY2=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
export PATH="/net/projects/ranalab/rajhansini/conda_envs/richards_distillation/bin:$PATH"
cd "$E"
echo "=== $(date) host=$(hostname) OBJ=${OBJ} NO_R27=${NO_R27:-0} ==="
# GATE-encoder. NOT `| grep -q libx264`: under `set -o pipefail` grep -q exits on the
# first match and closes the pipe, ffmpeg dies of SIGPIPE, and pipefail reports the
# PIPELINE as failed even though the match succeeded -- so a perfectly good ffmpeg is
# rejected. grep -c drains its input, so there is no SIGPIPE to misread. Same class of
# bug as the `ls glob | wc -l` note in render_view.sbatch.
HAVE_X264=$(ffmpeg -hide_banner -encoders 2>/dev/null | grep -c libx264 || true)
echo "ffmpeg=$(command -v ffmpeg)  libx264 entries=${HAVE_X264:-0}"
[ "${HAVE_X264:-0}" -ge 1 ] || { echo "FAILED: ffmpeg on PATH has no libx264"; exit 1; }

${PY2} -u "$E/jobs/build_daily_video.py" "${OBJ}" ${NO_R27:+--no-r27} 2>&1 \
  | stdbuf -oL grep --line-buffered -vE "FutureWarning|UserWarning|warnings.warn"

echo "=== done $(date) ==="
ls "$E/out/DAILY/${OBJ}"_*.mp4 2>/dev/null | wc -l | xargs -I{} echo "{}/4 daily videos present for ${OBJ}"
