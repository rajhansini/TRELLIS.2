#!/bin/bash
# submit_rung37_w11.sh -- rung37 (PER-POSITION temporal LoRA) at an 11-FRAME window,
# all 42 objects. Companion to submit_rung31_w11.sh, which is the JOINT
# spatio-temporal arm at the same width. Together they separate the two axes:
#
#   rung31 tw11 : ot = inner(x, context)  -> 11,319 keys, joint space x time
#   rung37 tw11 : ot = inner(x, c_mix)    ->  1,029 keys, time only at each position
#
# WHY IT DID NOT EXIST. rung37_perpos_temporal_lora.py declared
# `--temporal-window choices=[0, 3, 5]`, so 11 died at argument parsing. The mixer
# itself was always width-generic (fold to (W,N,D), softmax over W), and the cross-
# attention context stays 1029 tokens after the collapse, so W=11 is STRICTLY
# CHEAPER than rung31 tw11 -- which already ran 42 objects on a 32G/48GiB cell.
# That domination is why there is no canary gate here.
#
# Run dirs land as rung37_l1_lp_tw11bpp_all_qkvo+sa_r4_s42_<hash>, distinct from
# tw3bpp and from every rung31 dir, so nothing can resume across windows or arms.
#
#   ONLY=obj1,obj2   restrict objects
#   LIMIT=n          submit at most n (canary)
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
           'rung37_perpos_temporal_lora.py').read()
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

finished() {   # obj -> 'yes' if a 30-epoch tw<W>bpp run already wrote final_eval.json
  python3 - "$E" "$1" "$TW" <<'PY'
import json,sys,glob,os
E,obj,tw=sys.argv[1:4]
for c in glob.glob(f'{E}/runs/rung37_l1_lp_tw{tw}bpp_all_qkvo+sa_r4_s42_*/config.json'):
    try: d=json.load(open(c))
    except Exception: continue
    if d.get('epochs')!=30 or d.get('temporal_window')!=int(tw): continue
    if str(d.get('mcfm'))!='None': continue
    if f'/data/{obj}/' not in (d.get('gt_dir') or ''): continue
    if os.path.exists(os.path.join(os.path.dirname(c),'final_eval.json')):
        print('yes'); break
PY
}

squeue -u "$USER" -h -o "%j" > /tmp/_r37w.$$ || true
N=0; S=0; SKIP=0
while IFS=$'\t' read -r B OBJ NFR MESH GTDIR GTREN; do
  [ "$B" = "batch" ] && continue
  [ -z "$OBJ" ] && continue
  [ -n "${ONLY:-}" ] && [[ ",$ONLY," != *",$OBJ,"* ]] && continue
  N=$((N+1))
  [ -n "$(finished "$OBJ")" ] && { echo "   skip $OBJ (done)"; SKIP=$((SKIP+1)); continue; }
  JN="37_w${TW}_${OBJ}"        # '37_' is registered in watchdog.py KIND -> fig37.sbatch
  grep -qx "$JN" /tmp/_r37w.$$ && { echo "   skip $OBJ (queued)"; SKIP=$((SKIP+1)); continue; }
  # --gt-render-dir needs the BACK-PROJECTED targets (col 6), not the raw video
  # frames (col 5). Passing col 5 is what sent an earlier batch into
  # "GATE-align FAILED: render_mask.npy is missing".
  [ -f "$MESH" ]  || { echo "   MISSING MESH        $OBJ"; continue; }
  [ -d "$GTREN" ] || { echo "   MISSING GTREN       $OBJ"; continue; }
  [ -f "$(dirname "$GTREN")/render_mask.npy" ] || { echo "   MISSING render_mask $OBJ"; continue; }
  [ -n "${DRY:-}" ] && { printf '   %-1s %-24s tw=%-3s nfr=%s\n' "$B" "$OBJ" "$TW" "$NFR"; continue; }
  [ -n "${LIMIT:-}" ] && [ "$S" -ge "$LIMIT" ] && continue
  EXP="OBJ=${OBJ},NFR=${NFR},MESH=${MESH},GTDIR=${GTREN},TW=${TW}"
  J=$(sbatch --parsable ${NICE:+--nice="$NICE"} --job-name="$JN" \
       --export=ALL,"$EXP" jobs/fig37.sbatch)
  note "$J" "$JN" "$EXP" "jobs/fig37.sbatch" \
    "TEMPORAL-ONLY W=${TW}: rung37 per-position temporal LoRA (fold to (W,N,D), softmax over W at a fixed spatial index; cross-attn context stays 1029 tokens), branch both, gate 0, r4 s42 30ep, ${OBJ} batch ${B} ${NFR}fr; log out/FIGRUNS/${JN}_${J}.log; TIMEOUT at the 4h wall is expected and resumes from ckpts; PASS: log has '=== Done' and runs/rung37_l1_lp_tw${TW}bpp_all_qkvo+sa_r4_s42_*/final_eval.json exists for ${OBJ}"
  echo "  $J  $JN"
  S=$((S+1))
done < "$TSV"
rm -f /tmp/_r37w.$$
echo; echo "cells considered ${N}; already done or queued ${SKIP}; submitted ${S}"
exit 0
