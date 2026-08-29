#!/bin/bash
# submit_on_targets.sh — submit an object's rung ladder the moment its 2D-copy
# targets land, and never twice.
#
# WHY A MARKER FILE. The watcher re-runs every poll. Without out/SUBMITTED_<obj>
# a slow squeue or a restarted watcher would submit the same 14 cells again, and
# 154 duplicate GPU jobs is not a mistake you notice until fairshare collapses.
#
# WHY GATE-align GATES THIS. gt_targets.json is written ONLY after build_targets_hero
# passes GATE-align, so its existence already means the mesh is correctly posed.
# A mis-posed object never reaches this script.
#
# 32 and 34 take no MCFM: --context-window and --mcfm are mutually exclusive.
set -uo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
T2=/net/projects/ranalab/rajhansini/TRELLIS.2
cd "$E"
PYBIN=/net/projects/ranalab/rajhansini/conda_envs/trellis2/bin/python
OBJS="animal_blob_crack animal_blob_orange_crack chair_real_wooden_crack chair_ice chair_moss napolean_teapot_crack napolean_waves octopus_tar octopus_sparkle octopus_rainbow octopus_rust octocat_clay octocat_shine sheep_soil sheep_mud dragon_mush dragon_mush2 tie_fighter_bw penguin_ice penguin_orange_mush penguin_real fish_glitter fish_ink"

# Two runners (the nightwatch slurm job and the detached login-node loop) both
# call this. The marker file alone is not enough: it is written AFTER the 14
# sbatches, so both could pass the check and submit 28 jobs. mkdir is atomic on
# this filesystem, so it is the lock.
for o in $OBJS; do
  TG="$E/out/gt_targets_$o/gt_targets.json"
  [ -f "$TG" ] || continue
  [ -f "$E/out/SUBMITTED_$o" ] && continue
  mkdir "$E/out/.lock_$o" 2>/dev/null || continue      # another runner has it
  trap 'rmdir "$E/out/.lock_'"$o"'" 2>/dev/null' EXIT
  if [ -f "$E/out/SUBMITTED_$o" ]; then                 # won the race, lost the check
    rmdir "$E/out/.lock_$o" 2>/dev/null; continue
  fi
  MESH="$T2/data/$o/mesh/${o}_render_frame.obj"
  GTDIR="$E/out/gt_targets_$o/frames"
  NFR=$(ls "$T2/data/$o/frames_from_video"/*.png 2>/dev/null | wc -l)
  if [ ! -f "$MESH" ] || [ ! -d "$GTDIR" ] || [ "$NFR" -lt 1 ]; then
    echo "SKIP $o — mesh/gtdir/frames missing"
    rmdir "$E/out/.lock_$o" 2>/dev/null; continue
  fi
  n=0
  for r in 27 28 29 30 31 32 33 34; do
    # 32 and 34 are wide-context: MCFM is mutually exclusive there
    if [ "$r" = "32" ] || [ "$r" = "34" ]; then MODES=""; else MODES=" v2_D"; fi
    for m in "" $MODES; do
      # FULL object name, never truncated. ${o:0:9} collapsed
      # animal_blob_crack/animal_blob_orange_crack to one name, and would also have
      # collapsed napolean_teapot_crack/napolean_waves and octopus_rainbow/rust.
      # The watchdog keys its live-guard and retry counter on the job name, so two
      # cells sharing a name are treated as one and a real failure goes unretried.
      [ -n "$m" ] && nm="${r}_${m}_${o}" || nm="${r}_${o}"
      jid=$(sbatch --parsable --job-name="$nm" \
            --export=ALL,OBJ=$o,MODE=$m,NFR=$NFR,MESH=$MESH,GTDIR=$GTDIR \
            jobs/fig${r}.sbatch 2>&1)
      if [[ "$jid" =~ ^[0-9]+$ ]]; then
        n=$((n+1))
        # Register with the watchdog AT SUBMIT TIME. The fig log header is
        # 'OBJ=.. ARM=.. MODE=.. nfr=..', which watchdog.parse_log cannot parse --
        # it would recover OBJ alone and resubmit without MODE/MESH/GTDIR, training
        # the wrong thing successfully. Manifest first, log only as a fallback.
        EXPORT="OBJ=$o,MODE=$m,NFR=$NFR,MESH=$MESH,GTDIR=$GTDIR" \
        JID="$jid" NAME="$nm" SCRIPT="jobs/fig${r}.sbatch" \
        "$PYBIN" - <<'PYEOF'
import json, os
from pathlib import Path
f = Path('out/job_manifest.json')
try: m = json.loads(f.read_text())
except Exception: m = {}
m[os.environ['JID']] = {'name': os.environ['NAME'],
                        'script': os.environ['SCRIPT'],
                        'export': os.environ['EXPORT']}
f.write_text(json.dumps(m, indent=1))
PYEOF
        printf '%s\t%s\t%s\tfig%s\tMODE=%s\tout/FIGRUNS/%s_%s.log\tpass=[FINAL] rung\n' \
          "$(date +'%F %T')" "$jid" "$nm" "$r" "${m:-none}" "$nm" "$jid" >> JOBLOG.tsv
      else
        echo "SUBMIT FAILED $o rung$r mode=${m:-none}: ${jid:0:120}"
      fi
    done
  done
  date +"%F %T submitted $n cells" > "$E/out/SUBMITTED_$o"
  rmdir "$E/out/.lock_$o" 2>/dev/null
  echo "SUBMITTED $o — $n cells"
done
