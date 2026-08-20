#!/bin/bash
# log_ta.sh <jobid> <repo> <object> <mode> <outdir>
# Appends one row to temporal_attention.md. State is resolved live by report_ta.sh,
# not written here -- a state written at submit time is a guess that ages badly.
F=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/temporal_attention.md
printf '%s | %s | %s | %s | %s | %s\n' "$(date '+%m-%d %H:%M')" "$1" "$2" "$3" "$4" "$5" >> "$F"
