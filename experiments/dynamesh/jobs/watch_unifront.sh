#!/bin/bash
# Failure monitor for the unicorn front-sweep fan-out.
# Exists because the first pair of these died on a bad --ckpt path and nobody noticed
# until the user hit the same artifact again. Emits one line per TERMINAL state, and
# checks the PASS condition (24 angle renders) rather than trusting COMPLETED.
B=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
O=/net/projects/ranalab/guanc/geometry-def/dynamesh_render/out
IDS=$(tr ' ' ',' < $B/out/FIGRUNS/unifront_jobids.txt | sed 's/^,//;s/,$//')
prev=""
while true; do
  cur=$(sacct -j "$IDS" -X --format=JobName%18,State%14 -n 2>/dev/null \
        | awk '$2!="RUNNING" && $2!="PENDING" && NF {print $1": "$2}' | sort)
  comm -13 <(echo "$prev") <(echo "$cur"); prev="$cur"
  [ "$(echo "$cur" | grep -c ':')" -ge 2 ] && {
    for e in 8 18; do
      n=$(ls $O/unicorn_effect1_front_e${e}_f130/*.png 2>/dev/null | wc -l)
      [ "$n" -ge 20 ] && echo "PASS e$e: $n renders" || echo "FAIL e$e: only $n renders"
    done
    echo "ALL TERMINAL"; break; }
  sleep 40
done
