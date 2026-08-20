# ablation_jobs — the CVPR ablation registry

`ablation_jobs.tsv` is the machine-readable table; this file is how to use it.
One row per (ablation, rung, object). Append a row when you submit; fill `psnr`
and `delta` when it finishes. `JOBLOG.tsv` stays the chronological log — this is
the experiment-shaped view of the same work.

## Columns
    ablation   which ablation from ABLATIONS_CVPR.md (A-baseline, F-temporal-lora, ...)
    rung       rung number the script stamps into config.json
    object     spot_lava | skull_lava | pumpkin_rot | hand_rorschach | ...
    job_ids    slurm ids, comma or range
    state      QUEUED | RUNNING | DONE | FAILED
    run_dir    directory under runs/ — the run's identity, not the job id
    psnr       final_eval.json -> final.psnr_mean
    baseline   the rung27 number this is measured against
    delta      psnr - baseline
    notes      the flags that make this arm different

## Standing rules
- **One object per job.** Never loop objects inside one sbatch.
- **Every arm's distinguishing flags must be in `_CFG`**, or two arms hash to one
  directory and the second resumes from the first's checkpoint. That voided 24
  rung30 arms and 24 rung29 arms.
- **Do not cancel and resubmit to chase idle nodes.** `PriorityWeightAge = 143`
  outranks `PriorityWeightFairShare = 120`, and age maxes at 9 h. Resubmitting
  resets age to zero and makes it worse.
- **Small asks get backfilled.** Renders need ~24G/4cpu/40min, not 96G/8cpu/2h.
- Log every submission to `/net/projects/ranalab/rajhansini/JOBLOG.tsv` too.

## Still to run (from ABLATIONS_CVPR.md)
    B  lpips on/off            6 runs
    C  lora rank               10 runs   supplementary
    D  temporal window 1/2/3/5 10 runs   NEEDS mcfm_blend.py window 'E'
    E  v2 vs v3 blend          4 runs
    F2 temporal-LoRA only      4 runs    isolates what F3 measured
    G  frame count             6 runs
    +  seeds on 27 and 31      needed — every number is seed 42 only
