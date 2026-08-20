#!/bin/bash
# Print every logged run with its CURRENT slurm state and output frame count.
F=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/temporal_attention.md
printf '%-12s %-9s %-9s %-16s %-24s %-11s %s\n' DATE JOB REPO OBJECT MODE STATE FRAMES
grep -E '^[0-9]{2}-[0-9]{2} ' "$F" 2>/dev/null | while IFS='|' read -r d j r o m out; do
  j=$(echo $j|xargs); st=$(sacct -j "$j" -X -n -P --format=State 2>/dev/null|head -1)
  out=$(echo $out|xargs); n=0; [ -d "$out" ] && n=$(find "$out" -name '*.png' 2>/dev/null|wc -l)
  printf '%-12s %-9s %-9s %-16s %-24s %-11s %s\n' "$(echo $d|xargs)" "$j" "$(echo $r|xargs)" \
    "$(echo $o|xargs)" "$(echo $m|xargs)" "${st:-?}" "$n"
done
