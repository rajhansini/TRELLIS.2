"""
watchdog.py — resubmit jobs that died for infrastructure reasons, and refuse to
resubmit jobs that died for real reasons.

WHY THE DISTINCTION IS THE WHOLE POINT
  A node failure, a preemption, a wall-clock timeout: resubmitting is correct and
  the job will probably succeed. A GATE failure is the opposite -- alien_glow's
  targets failed three times tonight on a mesh in the wrong pose, and resubmitting
  it unchanged would have looped forever while looking busy. So a job whose log
  contains a GATE failure is recorded as NEEDS-ATTENTION and never retried.

  Same for a job cancelled by hand: that was a decision, not an accident.

HOW IT KNOWS WHAT TO RESUBMIT
  Every sbatch here echoes its parameters into its own log ("OBJ=... nfr=...",
  "RUN=... TAG=... yaw0=... elev=... turns=..."). The watchdog reads them back out
  of the log rather than keeping a separate manifest that could drift out of sync
  with what actually ran.

RETRY CAP
  Three attempts per logical unit of work, counted in a state file that survives
  restarts. Past that it is not a transient fault and a human should look.

IT ALSO NURSES THE QUEUE
  Held renders are released a few at a time while fewer than --keep-running of my
  jobs are running. Submitting all of them at once is what drove fairshare to
  0.087 and made everything crawl.
"""
import argparse, json, re, subprocess, time
from datetime import datetime
from pathlib import Path

E = Path('/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh')
STATE = E / 'out' / 'watchdog_state.json'
WLOG = E / 'out' / 'WATCHDOG.log'

ap = argparse.ArgumentParser()
ap.add_argument('--since', default='2026-08-17T05:00')
ap.add_argument('--every', type=int, default=300)
ap.add_argument('--minutes', type=float, default=225, help='exit after this long, for a chained job')
ap.add_argument('--keep-running', type=int, default=8)
ap.add_argument('--release-batch', type=int, default=4)
ap.add_argument('--max-retries', type=int, default=3)
ap.add_argument('--dry-run', action='store_true')
A = ap.parse_args()

LIVE = set()
BAD = {'FAILED', 'TIMEOUT', 'NODE_FAIL', 'PREEMPTED', 'OUT_OF_MEMORY', 'BOOT_FAIL'}
# jobname prefix -> (sbatch script, log prefix)
KIND = {'tgt_': ('jobs/targets_hero.sbatch', 'tgt'),
        # training-timestep bias experiment (rung27_fastprefix.py copy).
        'kbias_': ('jobs/kbias_hand.sbatch', 'kbias'),
        # sequential vs frame-batched inference benchmark.
        'benchpar_': ('jobs/bench_parallel.sbatch', 'benchpar'),
        # corrected frozen TRELLIS.2 baseline renders (mesh-mask conditioning,
        # vanilla tokens). Registered at submit time so it is never an
        # unwatched prefix.
        'frzfix_': ('jobs/frozen_condfix.sbatch', 'frzfix'),
        'r27_': ('jobs/rung27_hero.sbatch', 'r27hg'),
        'mcf_': ('jobs/rung27_mcfm_hero.sbatch', 'r27hm'),
        'v_':   ('jobs/render_view.sbatch', 'rview'),
        'g_':   ('jobs/render_view.sbatch', 'rview'),
        # added after five pan_ jobs and seven m3_ jobs sat failed/at-risk with the
        # watchdog blind to them: a prefix missing from this table is not watched.
        'm3_':  ('jobs/rung27_mcfm_mode.sbatch', 'r27m3'),
        # full-rate Table B fleet; CPU-only, no gres -- added at submit time so it
        # is never one of the unwatched prefixes that hid pan_ and m3_.
        'fr_':  ('jobs/fullrate_obj.sbatch', 'fr'),
        # reversed-ladder temporal+cross-attn fleet (rung26 --targets qkvo).
        'r26ca_': ('jobs/rung26_mcfm_ca.sbatch', 'r26ca'),
        # texel for the reversed ladder's temporal+CA row.
        'tx19_': ('jobs/texel_after_train.sbatch', 'texdep'),
        'pan_': ('jobs/panels_one.sbatch', 'pan'),
        'ta2_': ('jobs/ta_t2.sbatch', 'ta2'),
        # The figure matrix. Names are '<rung>_<mode>_<obj>' (e.g. 28_v2_D_spot_lava),
        # which shares no prefix with anything above -- so before this entry existed
        # kind_of() returned None and the whole 42-cell matrix ran unwatched.
        **{f'{r}_': (f'jobs/fig{r}.sbatch', 'FIG') for r in range(27, 35)},
        # Row (b) of the ablation table: cross-attention-only LoRA. Job names are
        # '19_<obj>'. range(27,35) above does not cover 19, and a prefix missing
        # from this table is not watched -- the same blind spot that hid pan_/m3_.
        '19_': ('jobs/fig19.sbatch', 'FIG'),
        'rarm_': ('jobs/render_arm.sbatch', 'RENDER'),
        # batch F/G pixel renders. kind_of() matches on startswith and these are
        # named 'rfg_<obj>_<arm>_<view>', which shares no prefix with 'rarm_' --
        # registered at submit time so they are never an unwatched fleet.
        'rfg_': ('jobs/render_arm.sbatch', 'RENDER'),
        # The batch-C render fleet. kind_of() matches on startswith and
        # 'rarm4c_' does NOT start with 'rarm_', so all 392 of these ran
        # unwatched -- the same blind spot that hid pan_ and m3_.
        'rarm4c_': ('jobs/render_arm.sbatch', 'RENDER'),
        'rarm7c_': ('jobs/render_arm.sbatch', 'RENDER'),
        # GATE-align rescue for the penguins. A prefix missing from this
        # table is not watched -- the same gap that left pan_ and m3_ blind.
        'fit_': ('jobs/fit_align.sbatch', 'fit'),
        # Texel temporal metrics. Job names are 'tex_<obj>_<arm>' and the sbatch
        # writes out/tex_<jid>.log, so the log prefix is 'tex'. A prefix missing
        # from this table is not watched -- and texel_any.sbatch has a 2h wall
        # clock that has already timed out once (2204282, r31mcfm).
        'tex_': ('jobs/texel_any.sbatch', 'tex'),
        # Window ablation (W = 1/3/5) and the video-space comparison metrics.
        # 'texdep_' resolves its run directory at RUNTIME from (RUNG, MODE, OBJ),
        # so unlike 'tex_' its manifest export carries no RUN= and must not be
        # rewritten into one. A prefix missing from this table is not watched --
        # the same blind spot that hid pan_ and m3_.
        'texdep_': ('jobs/texel_after_train.sbatch', 'texdep'),
        'fxv_': ('jobs/fixview_arm.sbatch', 'fxv'),
        # Dailies renders. submit_dailies_de.sh names them 'dl_<obj>_<arm>_<view>'
        # and submits jobs/render_arm.sbatch, whose log prefix is RENDER. 'dl_' shares
        # no prefix with 'rarm_', so before this entry every dailies render ran
        # unwatched -- the same blind spot that hid pan_ and m3_.
        'dl_': ('jobs/render_arm.sbatch', 'RENDER'),
        # Daily panel composites, one job per object, named 'dv_<obj>'; the sbatch
        # writes out/dv_<name>_<jid>.log. 'dv_' shares no prefix with 'dl_' or 'pan_',
        # and a prefix missing from this table is not watched -- the same blind spot
        # that hid pan_ and m3_.
        'dv_': ('jobs/daily_video_one.sbatch', 'dv'),
        # batch F/G dailies: arm-generic composite (view_<obj>_<arm>_<view> tags),
        # submitted as 'dva_<obj>'. Logs land as out/dva_dva_<obj>_<jid>.log because
        # the sbatch --output pattern is dva_%x_%j and %x is already 'dva_<obj>'.
        'dva_': ('jobs/daily_video_arms.sbatch', 'dva'),
        # Target building. build_targets_one.sbatch is named 'mkt_one' and
        # build_targets_obj.sbatch 'mkt_obj', and BOTH start with 'mkt_' -- neither
        # was in this table, so every target build has run unwatched, the same
        # blind spot that hid pan_ and m3_. The generalised script is the resubmit
        # target because it reads OBJ/ORIENT/SUFFIX from --export, which is the
        # only thing a watchdog resubmit carries forward.
        'mkt_': ('jobs/build_targets_obj.sbatch', 'mkt'),
        # Fixed-camera panel renders. gate_then_submit_r32.sh names these 'pv_*'
        # and they write out/pview_<jid>.log, but 'pv_' was never in this table --
        # so every panel render has been unwatched. 'pan_' does NOT cover them:
        # kind_of() matches on startswith and 'pv_' shares no prefix with 'pan_'.
        'pv_': ('jobs/panel_view.sbatch', 'pview'),
        # rung37 per-position temporal LoRA, submitted as 'r37_<obj>' by
        # jobs/submit_rung37.sh with log prefix out/r37_<jid>.log. The first pass
        # ran at a 4h wall and 21 of 42 hit TIMEOUT unnoticed, because 'r37_' was
        # missing here and a prefix missing from this table is not watched.
        'r37_': ('jobs/rung37_perpos.sbatch', 'r37'),
        # rung37 at a WIDER window, submitted as '37_w<W>_<obj>' by
        # jobs/submit_rung37_w11.sh against jobs/fig37.sbatch, logging to
        # out/FIGRUNS/%x_%j.log like the rung31 w11 arm. range(27,35) above
        # stops at 34, so '37_' is NOT covered by it and must be listed here or
        # a TIMEOUT passes unnoticed -- the exact failure 'r37_' already cost.
        '37_': ('jobs/fig37.sbatch', 'FIG'),
        # W=11 pixel renders for the headline-window switch, submitted as
        # 'w11_<obj>_<view>' by jobs/submit_w11_renders.sh; log prefix RENDER.
        'w11_': ('jobs/render_arm.sbatch', 'RENDER'),
        # W=5 / W=7 pixel renders, submitted as 'w5_<obj>_<view>' and
        # 'w7_<obj>_<view>' by jobs/submit_w5w7_renders.sh. Same sbatch and log
        # prefix as w11_; registered here at submit time so the arm is watched
        # from the first job rather than after a silent TIMEOUT sweep.
        'w5_': ('jobs/render_arm.sbatch', 'RENDER'),
        'w7_': ('jobs/render_arm.sbatch', 'RENDER'),
        # VIDEO-SPACE renders of the r19 / r37 / r31 arms, submitted as
        # '<arm>v_<obj>_<view>' by jobs/submit_arm_renders.sh; same sbatch and log
        # prefix as w11_. Registered HERE AT SUBMIT TIME, before the fleet lands,
        # rather than after a silent TIMEOUT sweep -- that is what r37_ cost.
        # All four are listed: kind_of() matches on startswith, and 'r31mv_' does
        # NOT start with 'r31v_', so one entry would leave the mcfm arm unwatched
        # the same way 'rarm4c_' slipped past 'rarm_'.
        'r19v_':  ('jobs/render_arm.sbatch', 'RENDER'),
        'r37v_':  ('jobs/render_arm.sbatch', 'RENDER'),
        'r31v_':  ('jobs/render_arm.sbatch', 'RENDER'),
        'r31mv_': ('jobs/render_arm.sbatch', 'RENDER'),
        # W=11 full-rate pixel metrics, jobs/fullrate_obj_w11.sbatch, log prefix fr.
        'frw_': ('jobs/fullrate_obj_w11.sbatch', 'fr'),
        # Window-arm full-rate pixel metrics (W=5 / W=7 / W=11 on the 2D-copy
        # reference), submitted as 'fw5_<obj>' / 'fw7_<obj>' / 'fw11_<obj>' by
        # jobs/submit_window_fullrate.sh. CPU-only, log prefix fr. Registered at
        # submit time so a TIMEOUT cannot pass unnoticed the way r37_ did.
        'fw5_':  ('jobs/fullrate_obj_arm.sbatch', 'fr'),
        'fw7_':  ('jobs/fullrate_obj_arm.sbatch', 'fr'),
        'fw11_': ('jobs/fullrate_obj_arm.sbatch', 'fr'),
        # Video-space metrics for the ABLATION-LADDER arms, submitted as
        # 'fr19_<obj>' / 'fr37_<obj>' / 'fr31_<obj>' / 'fr31m_<obj>' by
        # jobs/submit_ladder_fullrate.sh. Same sbatch and 'fr' log prefix as the fw*
        # window arms. Registered at SUBMIT time, which is the whole lesson of the
        # r37_ entry above. All four spelled out: kind_of() matches on startswith and
        # 'fr31m_' does NOT start with 'fr31_', the same way 'rarm4c_' slipped past
        # 'rarm_'. None of them collides with the existing 'frw_'.
        # Batch F renders (doorknob_spinodal, doorknob_bz, bunny_shine) at the four
        # fixed cameras, submitted as 'bf_<obj>_<view>'. Registered at submit time.
        # 'bfr_' is the CORRECTED batch F resubmit. The first 'bf_' round carried a RUN
        # path missing its 'runs/' component, so all 12 died on render_arm.sbatch's
        # preflight and the watchdog faithfully retried each 3 times -- 48 failures from
        # one wrong string. The manifest export is what a retry replays, so a bad export
        # is retried verbatim; the preflight is what stopped it rendering a wrong arm.
        'bfr_':   ('jobs/render_arm.sbatch', 'RENDER'),
        'bf_':    ('jobs/render_arm.sbatch', 'RENDER'),
        'fr19_':  ('jobs/fullrate_obj_arm.sbatch', 'fr'),
        'fr37_':  ('jobs/fullrate_obj_arm.sbatch', 'fr'),
        'fr31_':  ('jobs/fullrate_obj_arm.sbatch', 'fr'),
        'fr31m_': ('jobs/fullrate_obj_arm.sbatch', 'fr'),
        # 'fr27_' is the +self-attention / temporal-OFF row of the video-space
        # ladder: arm 27, the non-MCFM rung27 run. It is the only ladder row that
        # had no FULLRATE dir. Spelled out for the same startswith reason as above;
        # 'fr27_' does not collide with 'fr31_', 'fr37_' or 'frw_'.
        'fr27_':  ('jobs/fullrate_obj_arm.sbatch', 'fr'),
        # 'r27v_' re-renders the arm-27 (MCFM off) panels. The Aug-22 frames for
        # spot_lava disagreed with their own 30-epoch checkpoint by 4.6 dB and no
        # log accounts for them. Registered at submit time like the rest.
        'r27v_': ('jobs/render_arm.sbatch', 'RENDER'),
        # BATCH D, the 7-arm supplementary ladder. Registered at SUBMIT time, not
        # after the first failure, so it is never one of the unwatched prefixes
        # that hid pan_, m3_, mkt_ and pv_. batch_d_arm.sbatch reads OBJ / NFR /
        # TARGETS / MODE / MESH / GTDIR from --export, and a resubmit carries only
        # --job-name and --export, so the manifest entry is what makes it usable;
        # constraint (a40|L40S) and mem live in the script for the same reason.
        # 'bd_' DISABLED 2026-08-26 at Raj's request: batch-D objects are not in the
        # 24-object roster and feed neither the ablation nor the comparison table. It is
        # listed here so the watchdog does not resubmit them while the submission fixes
        # (fr_*, rarm_27m_spot_lava) hold the GPU slots. Re-enable by restoring this line.
        # 'bd_': ('jobs/batch_d_arm.sbatch', 'bd'),
        }

# jobid -> {script, name, export}. Written at SUBMIT time by whoever launched the
# job. The log header is not always enough: rung27_mcfm_mode.sbatch echoes only
# OBJ and nfr, so rebuilding its command from the log would silently drop MODE,
# MESH and GTDIR -- and hand the guan-pipeline objects hero paths. Manifest first,
# log only as a fallback.
MANIFEST = E / 'out' / 'job_manifest.json'


def manifest_entry(jid):
    try:
        return json.loads(MANIFEST.read_text()).get(str(jid))
    except Exception:
        return None


def sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True).stdout


def note(msg):
    line = f'{datetime.now():%Y-%m-%d %H:%M}  {msg}'
    with open(WLOG, 'a') as f:
        f.write(line + '\n')
    print(line, flush=True)


def load():
    if STATE.exists():
        try:
            return json.loads(STATE.read_text())
        except Exception:
            pass
    return {'tries': {}, 'seen': [], 'attention': [], 'epochs': {}}


def kind_of(name):
    for p in KIND:
        if name.startswith(p):
            return p
    return None


def last_epoch(logp):
    """Highest epoch this attempt reached, from the trainer's own '[EPOCH n/N]' line.

    A 30-epoch run at a 4h wall does NOT fit in one slice -- rung37 at W=11 was at
    epoch 12 after 2h11 -- so a TIMEOUT is the normal way these jobs end, not a
    failure. The trainer resumes from ckpts/lora_e*.pt, so each slice adds epochs.
    Counting those against --max-retries retires a healthy run after 3 slices and
    leaves it at epoch ~20 forever, which is exactly the silent stall this function
    exists to prevent. Returns None for jobs that print no epoch line (renders,
    target builds), and those keep the old counting unchanged.
    """
    if not logp.exists():
        return None
    try:
        eps = re.findall(r'\[EPOCH (\d+)/(\d+)\]', logp.read_text(errors='replace'))
    except Exception:
        return None
    if not eps:
        return None
    return max(int(a) for a, _ in eps), int(eps[-1][1])


def parse_log(logp):
    """Recover the sbatch --export from the job's own header line."""
    if not logp.exists():
        return None
    head = logp.read_text(errors='replace')[:4000]
    m = re.search(r'RUN=(\S+) TAG=(\S+) yaw0=(\S+) elev=(\S+) turns=(\S+)', head)
    if m:
        run, tag, y, e, t = m.groups()
        res = '1024' if tag.startswith('guan_') else '518'
        return f'RUN={run},TAG={tag},YAW0={y},ELEV={e},TURNS={t},RES={res}', tag
    m = re.search(r'OBJ=(\S+) nfr=(\S+)', head)
    if m:
        return f'OBJ={m.group(1)},NFR={m.group(2)}', m.group(1)
    m = re.search(r'OBJ=(\S+) ', head)
    if m:
        return f'OBJ={m.group(1)}', m.group(1)
    return None


NFRAMES = {'ancient_lady': 135}          # everything else is 150


def satisfied(k, unit):
    """Has this work ALREADY been done, by a manual resubmit or an earlier retry?

    Without this the watchdog re-runs every historical failure it can see. Tonight's
    60 renders died on a path bug, were fixed and resubmitted by hand, and their
    outputs now exist -- resubmitting them again would waste hours of GPU and, worse,
    look like progress.
    """
    if k == 'v_' or k == 'g_':
        d = E / 'out' / unit / 'frames'                     # unit is the TAG here
        if not d.is_dir():
            return False
        obj = next((o for o in NFRAMES if o in unit), None)
        want = NFRAMES.get(obj, 150)
        return len(list(d.glob('*.png'))) >= want
    if k == 'tgt_':
        return (E / 'out' / f'gt_targets_{unit}' / 'gt_targets.json').exists()
    if k in {'rarm_', 'rarm4c_', 'rarm7c_'}:
        # unit is the TAG, e.g. view_pumpkin_rot_29_train -- pull the object name
        # back out and check against ITS OWN frame count, not a fixed constant.
        m = re.match(r'view_(.+)_(?:\d+m?)_(?:train|diagA|diagB|diagC|u\d+)$', unit)
        obj = m.group(1) if m else None
        want = 150
        if obj:
            vids = list(Path('/net/projects/ranalab/rajhansini/TRELLIS.2/data', obj, 'frames_from_video').glob('*.png'))
            if vids: want = len(vids)
        d = E / 'out' / unit / 'frames'
        return d.is_dir() and len(list(d.glob('*.png'))) >= want
    if k in {f'{r}_' for r in range(27, 35)} | {'19_'}:
        d = E / 'out' / 'FIGRUNS'
        if d.is_dir():
            for q in d.glob(f'{unit}_*.log'):
                if '[FINAL] rung' in q.read_text(errors='replace'):
                    return True
        return False
    if k in ('r27_', 'mcf_'):
        pre = 'r27hg' if k == 'r27_' else 'r27hm'
        for p in (E / 'out').glob(f'{pre}_*.log'):
            t = p.read_text(errors='replace')
            if f'OBJ={unit} ' in t and '[FINAL] rung' in t:
                return True
    return False


def live_names():
    """Job names currently PENDING or RUNNING. A failed job whose name is back in
    the queue has already been requeued -- by a human or by an earlier cycle -- and
    resubmitting it again just burns a GPU on a duplicate."""
    out = sh("squeue -u rajhansini -h -O 'Name:64,State:12'")
    names = set()
    for l in out.splitlines():
        p = l.split()
        if len(p) >= 2 and p[1] in ('PENDING', 'RUNNING'):
            names.add(p[0])
    return names


def gated(logp):
    """True when the job died on a GATE — a data problem, not a transient one."""
    if not logp.exists():
        return False
    t = logp.read_text(errors='replace')
    return bool(re.search(r'GATE-\S+ FAILED', t)) or 'GATE-align FAILED' in t


def cycle(st):
    global LIVE
    LIVE = live_names()
    rows = sh(f"sacct -S {A.since} -X -n -P --format=JobID,JobName,State").strip().splitlines()
    for row in rows:
        parts = row.split('|')
        if len(parts) < 3:
            continue
        jid, name, state = parts[0], parts[1], parts[2].split()[0]
        if state not in BAD or jid in st['seen']:
            continue
        k = kind_of(name)
        if k is None:
            continue
        st['seen'].append(jid)
        script, pre = KIND[k]
        if pre == 'FIG':
            hits = sorted((E / 'out' / 'FIGRUNS').glob(f'*_{jid}.log'))
            logp = hits[-1] if hits else E / 'out' / 'FIGRUNS' / f'{name}_{jid}.log'
        elif pre == 'RENDER':
            hits = sorted((E / 'out' / 'RENDERS').glob(f'*_{jid}.log'))
            logp = hits[-1] if hits else E / 'out' / 'RENDERS' / f'{name}_{jid}.log'
        else:
            logp = E / 'out' / f'{pre}_{jid}.log'
        if gated(logp):
            note(f'{jid} {name} {state} — GATE failure, NOT resubmitting (needs a fix)')
            if name not in st['attention']:
                st['attention'].append(f'{jid} {name} gate')
            continue
        if name in LIVE:
            note(f'{jid} {name} {state} — already queued/running again, not resubmitting')
            continue
        ent = manifest_entry(jid)
        if ent:
            # PROGRESS IS NOT A RETRY. See last_epoch(): a resuming trainer ends every
            # slice but the last in TIMEOUT, and each slice adds epochs. Charge the
            # budget only when the attempt gained nothing.
            prog = last_epoch(logp) if state == 'TIMEOUT' else None
            free = False
            if prog:
                ep, tot = prog
                seen_ep = st.setdefault('epochs', {}).get(ent['name'], 0)
                if ep > seen_ep:
                    st['epochs'][ent['name']] = ep
                    free = True
                    note(f'{jid} {name} TIMEOUT at epoch {ep}/{tot} (was {seen_ep}) '
                         f'— progress made, not charged against --max-retries')
            n = st['tries'].get(ent['name'], 0)
            if not free and n >= A.max_retries:
                note(f'{jid} {name} {state} — {n} retries already, giving up')
                st['attention'].append(f'{ent["name"]} exhausted'); continue
            if not free:
                st['tries'][ent['name']] = n + 1
            if A.dry_run:
                note(f'[dry-run] would resubmit {ent["name"]} from manifest'); continue
            out = sh(f'cd {E} && sbatch --parsable --job-name={ent["name"]} '
                     f'--export=ALL,{ent["export"]} {ent["script"]}').strip()
            if out.isdigit():
                # carry the manifest entry onto the NEW id, or a second timeout
                # would fall back to the log and lose the parameters after all
                try:
                    m = json.loads(MANIFEST.read_text()); m[out] = ent
                    MANIFEST.write_text(json.dumps(m, indent=1))
                except Exception:
                    pass
                note(f'{jid} {name} {state} -> resubmitted as {out} FROM MANIFEST '
                     f'(attempt {n+1}/{A.max_retries})')
            else:
                note(f'{jid} {name} {state} -> MANIFEST RESUBMIT FAILED: {out[:160]}')
            continue

        got = parse_log(logp)
        if got is None:
            note(f'{jid} {name} {state} — cannot recover params from {logp.name}, skipping')
            st['attention'].append(f'{jid} {name} unparseable')
            continue
        exp, unit = got
        if satisfied(k, unit):
            note(f'{jid} {name} {state} — output already complete, not resubmitting')
            continue
        key = name if k in {f'{r}_' for r in range(27, 35)} else f'{k}{unit}'
        n = st['tries'].get(key, 0)
        if n >= A.max_retries:
            note(f'{jid} {name} {state} — {n} retries already, giving up on {key}')
            if key not in st['attention']:
                st['attention'].append(f'{key} exhausted')
            continue
        st['tries'][key] = n + 1
        if A.dry_run:
            note(f'[dry-run] would resubmit {name} ({key}) attempt {n+1}')
            continue
        out = sh(f'cd {E} && sbatch --parsable --job-name={name} --export=ALL,{exp} {script}').strip()
        if out.isdigit():
            note(f'{jid} {name} {state} -> resubmitted as {out} (attempt {n+1}/{A.max_retries})')
        else:
            note(f'{jid} {name} {state} -> RESUBMIT FAILED: {out[:200]}')


def release():
    try:
        nrun = len([l for l in sh("squeue -u rajhansini -h -t R -o '%j'").splitlines()
                    if not l.startswith('nlp_')])
        if nrun >= A.keep_running:
            return
        ids = [l.split()[0] for l in
               sh("squeue -u rajhansini -h -t PD -o '%i %r'").splitlines()
               if 'JobHeldUser' in l][:A.release_batch]
        if ids:
            subprocess.run(['scontrol', 'release'] + ids, capture_output=True)
            note(f'released {len(ids)} held renders (running was {nrun})')
    except Exception as ex:
        note(f'release error: {ex}')


def main():
    stop = time.time() + A.minutes * 60
    note(f'watchdog up (every {A.every}s, until {datetime.fromtimestamp(stop):%H:%M})')
    while time.time() < stop:
        st = load()
        try:
            cycle(st)
            release()
        except Exception as ex:
            note(f'cycle error: {ex}')
        STATE.write_text(json.dumps(st, indent=1))
        time.sleep(A.every)
    note('watchdog exiting (chained job should take over)')


if __name__ == '__main__':
    main()
