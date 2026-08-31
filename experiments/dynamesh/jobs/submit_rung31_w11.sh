#!/bin/bash
# submit_rung31_w11.sh -- spatio-temporal attention over an 11-FRAME window, all 42
# objects. This is Itai's "trellis cross attention weights with lora applied to the
# tokens of 11 frames": rung31's dual branch, LoRA on the cross-attention weights,
# context stacked over W frames.
#
# WHY IT DID NOT EXIST. Every temporal-attention run on disk (rung31, rung31+MCFM,
# rung33, rung37) is temporal_window=3. W=11 was not merely unrun, it was UNRUNNABLE:
# rung31_dual_attn_lora.py declared `--temporal-window choices=[0, 3, 5]`, so 11 died
# at argument parsing. The stacking itself was always width-generic (_half = W//2,
# offsets -half..+half, same end-clamping as mcfm_blend.blend_conds), verified on CPU
# at W=11 before this script was written: 1029 -> 11,319 tokens, centre slice equals
# frame f byte for byte, both ends clamp instead of wrapping.
#
# THE RISK HERE IS VRAM, NOT CORRECTNESS. W=3 already peaks at 24 GiB of a 48 GiB
# card, and W=11 is 3.67x the context tokens. That is why CANARY=1 exists and why the
# fan-out must wait on a measured peak from a real job. MCFM's W=13/15 was safe to
# fan out early because its blend is precomputed once; this one is not.
#
# Run dirs land as rung31_l1_lp_tw11b_all_qkvo+sa_r4_s42_<hash>, distinct from tw3b,
# so nothing can resume across windows.
#
#   ONLY=obj1,obj2   restrict objects
#   LIMIT=n          submit at most n (CANARY)
#   TWIN=11          window width (default 11)
#   DRY=1            print the work list, submit nothing
set -eo pipefail
T2=/net/projects/ranalab/rajhansini/TRELLIS.2
E=$T2/experiments/dynamesh
MAN=$E/out/job_manifest.json
JL=/net/projects/ranalab/rajhansini/joblog.sh
TSV=$E/jobs/rung37_objects.tsv
TW=${TWIN:-11}
cd "$E"
[ -s "$TSV" ] || { echo "missing $TSV"; exit 1; }
mkdir -p "$E/out/FIGRUNS"

# The trainer must accept this width before 42 jobs go looking for it.
"$T2/../conda_envs/trellis2/bin/python" - "$TW" <<'PY' || exit 1
import sys, re
w = int(sys.argv[1])
src = open('/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/'
           'rung31_dual_attn_lora.py').read()
m = re.search(r"--temporal-window'.*?choices=\[([0-9, ]+)\]", src, re.S)
ok = m and w in [int(x) for x in m.group(1).split(',')]
print(f'trainer accepts --temporal-window {w}: {bool(ok)}')
sys.exit(0 if ok else 1)
PY

note() {
  python3 - "$MAN" "$1" "$2" "$3" "$4" <<'PY'
import json,sys,os
man,jid,name,exp,script=sys.argv[1:6]
m=json.load(open(man)) if os.path.exists(man) else {}
m[jid]={"script":script,"name":name,"export":exp}
json.dump(m,open(man,"w"),indent=1)
PY
  $JL add "$1" "$2" "$5"
}

finished() {   # obj -> 'yes' if a 30-epoch tw<W>b run already wrote final_eval.json
  python3 - "$E" "$1" "$TW" <<'PY'
import json,sys,glob,os
E,obj,tw=sys.argv[1:4]
for c in glob.glob(f'{E}/runs/rung31_l1_lp_tw{tw}b_all_qkvo+sa_r4_s42_*/config.json'):
    try: d=json.load(open(c))
    except Exception: continue
    if d.get('epochs')!=30 or d.get('temporal_window')!=int(tw): continue
    if str(d.get('mcfm'))!='None': continue
    if f'/data/{obj}/' not in (d.get('gt_dir') or ''): continue
    if os.path.exists(os.path.join(os.path.dirname(c),'final_eval.json')):
        print('yes'); break
PY
}

squeue -u "$USER" -h -o "%j" > /tmp/_r31w.$$ || true
N=0; S=0; SKIP=0
while IFS=$'\t' read -r B OBJ NFR MESH GTDIR GTREN; do
  [ "$B" = "batch" ] && continue
  [ -z "$OBJ" ] && continue
  [ -n "${ONLY:-}" ] && [[ ",$ONLY," != *",$OBJ,"* ]] && continue
  N=$((N+1))
  [ -n "$(finished "$OBJ")" ] && { SKIP=$((SKIP+1)); continue; }
  JN="31_w${TW}_${OBJ}"        # starts with '31_', so watchdog.py already watches it
  grep -qx "$JN" /tmp/_r31w.$$ && { SKIP=$((SKIP+1)); continue; }
  # fig31.sbatch feeds GTDIR to --gt-render-dir, which needs the BACK-PROJECTED
  # targets (col 6), not the raw video frames (col 5). Passing col 5 is what sent an
  # earlier batch into "GATE-align FAILED: render_mask.npy is missing".
  [ -f "$MESH" ]  || { echo "   MISSING MESH        $OBJ"; continue; }
  [ -d "$GTREN" ] || { echo "   MISSING GTREN       $OBJ"; continue; }
  [ -f "$(dirname "$GTREN")/render_mask.npy" ] || { echo "   MISSING render_mask $OBJ"; continue; }
  [ -n "${DRY:-}" ] && { printf '   %-1s %-24s tw=%-3s nfr=%s\n' "$B" "$OBJ" "$TW" "$NFR"; continue; }
  [ -n "${LIMIT:-}" ] && [ "$S" -ge "$LIMIT" ] && continue
  EXP="OBJ=${OBJ},NFR=${NFR},MESH=${MESH},GTDIR=${GTREN},TW=${TW}"
  J=$(sbatch --parsable ${NICE:+--nice="$NICE"} --job-name="$JN" \
       --export=ALL,"$EXP" jobs/fig31.sbatch)
  note "$J" "$JN" "$EXP" "jobs/fig31.sbatch" \
    "SPATIO-TEMPORAL W=${TW}: rung31 dual-branch, LoRA on cross-attn, context stacked over ${TW} frames (1029 -> $((1029*TW)) tokens), ${OBJ} batch ${B}, ${NFR}fr, seed42; log out/FIGRUNS/${JN}_${J}.log; TIMEOUT at the 4h wall is expected and resumes from ckpts; PASS: log has '[FINAL] rung' and the run's final_eval.json exists"
  echo "  $J  $JN"
  S=$((S+1))
done < "$TSV"
rm -f /tmp/_r31w.$$
echo; echo "cells considered ${N}; already done or queued ${SKIP}; submitted ${S}"
exit 0
