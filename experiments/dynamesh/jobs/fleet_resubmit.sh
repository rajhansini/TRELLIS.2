#!/bin/bash
# fleet_resubmit.sh -- one pass: resubmit any m3_/texdep_/tex_ job of THIS fleet that
# reached a terminal bad state and is not already back in the queue.
#
# WHY THIS EXISTS ALONGSIDE watchdog.py. The watchdog is the general mechanism, but it
# was not running when the first TIMEOUT landed: 2208501/2208502 exited at 22:50 and
# their chained successors sat PENDING on (Priority) behind the fleet. TIMEOUT is the
# NORMAL outcome here -- 30 epochs takes 3:00-4:00 against a 4:00 wall -- so a gap in
# watchdog coverage silently drops ablation cells.
#
# Parameters come from out/job_manifest.json, never from the log: rung27_mcfm_mode.sbatch
# echoes OBJ and NFR but not MESH/GTDIR, so a log-derived resubmit would hand the guan
# objects hero paths. Constraint and memory now live in the sbatch script itself, so a
# resubmit carrying only --job-name/--export still lands on a40|L40S at 32G.
#
# Retries are capped per job NAME (not id) at 3, tracked in out/fleet_retries.json.
# CANCELLED is deliberately NOT retried: every CANCELLED in this fleet so far was a
# deliberate scancel (premature measurement jobs, and one external bulk cancel).
set -eo pipefail
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
MAN=$E/out/job_manifest.json
RET=$E/out/fleet_retries.json
JL=/net/projects/ranalab/rajhansini/joblog.sh
SINCE=${SINCE:-2026-08-24T20:45}
MAXTRY=${MAXTRY:-3}
cd "$E"

python3 - "$MAN" "$RET" "$SINCE" "$MAXTRY" <<'PY' > /tmp/_fleet_resub.$$
import json, os, subprocess, sys
man_p, ret_p, since, maxtry = sys.argv[1:5]
maxtry = int(maxtry)
man = json.load(open(man_p)) if os.path.exists(man_p) else {}
ret = json.load(open(ret_p)) if os.path.exists(ret_p) else {}

# A terminal-bad job whose WORK later succeeded must not be resubmitted. sacct keeps the
# TIMEOUT row forever, and the name-is-live check only covers the window while a retry is
# still queued -- once that retry COMPLETES, the old TIMEOUT row looks actionable again and
# the cell gets retrained from scratch for another 4h. This is the same class of mistake as
# gating on ckpts/lora_best.pt: asking "is a job present" instead of "is the work done".
ROSTER = [l.split('\t')[0] for l in
          open('out/texel_r19_params.tsv').read().strip().split('\n')[1:]]
MODES = {'v2_E': 'w5', 'st_D': 'stD', 'v3_D': 'v3D', 'v2_D': 'r27mcfm'}

_trained = None
def trained_cells():
    """(object, mcfm) pairs whose rung27 training wrote final_eval.json."""
    global _trained
    if _trained is None:
        import glob
        _trained = set()
        for d in glob.glob('runs/rung27_*'):
            c = os.path.join(d, 'config.json')
            if not os.path.exists(c):
                continue
            try:
                cfg = json.load(open(c))
            except Exception:
                continue
            if cfg.get('context_window'):
                continue
            if not os.path.exists(os.path.join(d, 'final_eval.json')):
                continue
            gt = cfg.get('gt_dir') or ''
            o = next((x for x in ROSTER if f'/data/{x}/' in gt), None)
            if o:
                _trained.add((o, str(cfg.get('mcfm'))))
    return _trained

def satisfied(name):
    if name.startswith('m3_'):
        body = name[3:]
        m = next((k for k in MODES if body.endswith('_' + k)), None)
        if not m:
            return False
        return (body[:-(len(m) + 1)], m) in trained_cells()
    if name.startswith('bd_'):
        return False   # batch D is not in ROSTER; resubmit-on-BAD only, capped by MAXTRY
    if name.startswith(('texdep_', 'tex_')):
        tag = name.split('_', 1)[1]
        return os.path.exists(f'out/TEXEL/{tag}.json')
    return False

live = set()
for l in subprocess.run(['squeue', '-u', 'rajhansini', '-h', '-o', '%j'],
                        capture_output=True, text=True).stdout.split('\n'):
    if l.strip():
        live.add(l.strip())

BAD = {'FAILED', 'TIMEOUT', 'NODE_FAIL', 'PREEMPTED', 'OUT_OF_MEMORY', 'BOOT_FAIL'}
rows = subprocess.run(['sacct', '-u', 'rajhansini', '-S', since, '-n', '-X',
                       '-o', 'JobID,JobName%40,State'],
                      capture_output=True, text=True).stdout
out = []
for line in rows.split('\n'):
    f = line.split()
    if len(f) < 3:
        continue
    jid, name, state = f[0], f[1], f[2].rstrip('+')
    if not name.startswith(('m3_', 'texdep_', 'tex_', 'bd_')):
        continue
    if state not in BAD:
        continue
    if name in live:                      # already back in the queue
        continue
    if satisfied(name):                   # the WORK is already done -- see below
        print(f'SKIP {jid} {name} {state} -- cell already complete', file=sys.stderr)
        continue
    ent = man.get(jid)
    if not ent:
        print(f'SKIP {jid} {name} {state} -- no manifest entry', file=sys.stderr)
        continue
    n = ret.get(name, 0)
    if n >= maxtry:
        print(f'SKIP {jid} {name} {state} -- {n} retries already', file=sys.stderr)
        continue
    ret[name] = n + 1
    live.add(name)                        # do not queue the same name twice in one pass
    out.append('\t'.join([jid, name, state, ent['export'], ent['script'], str(n + 1)]))
json.dump(ret, open(ret_p, 'w'), indent=1)
print('\n'.join(out))
PY

N=0
while IFS=$'\t' read -r JID NAME STATE EXPORT SCRIPT TRY; do
  [ -z "$NAME" ] && continue
  J=$(sbatch --parsable --job-name="$NAME" --export=ALL,"$EXPORT" "$E/$SCRIPT")
  python3 - "$MAN" "$J" "$NAME" "$EXPORT" "$SCRIPT" <<'PY'
import json,sys,os
man,jid,name,exp,script=sys.argv[1:6]
m=json.load(open(man)); m[jid]={"script":script,"name":name,"export":exp}
json.dump(m,open(man,"w"),indent=1)
PY
  $JL add "$J" "$NAME" "AUTO-RESUBMIT by fleet_resubmit.sh: ${JID} ended ${STATE} (attempt ${TRY}/${MAXTRY}); params from out/job_manifest.json, resumes from lora_best.pt if a checkpoint exists; out=$E/out/*_${J}.log; PASS: the cell's run dir gains final_eval.json"
  echo "RESUBMIT ${JID} ${NAME} ${STATE} -> ${J} (attempt ${TRY}/${MAXTRY})"
  N=$((N+1))
done < /tmp/_fleet_resub.$$
rm -f /tmp/_fleet_resub.$$
[ "$N" = "0" ] && echo "nothing to resubmit"
exit 0
