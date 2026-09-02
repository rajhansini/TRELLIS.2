#!/bin/bash
# Failure monitor for the unicorn fixed-view fan-out. Checks the PASS condition on disk
# (150 renders) rather than trusting COMPLETED, and names the failing arm.
B=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
O=/net/projects/ranalab/guanc/geometry-def/dynamesh_render/out
IDS=$(tr ' ' ',' < $B/out/FIGRUNS/unifix_jobids.txt | sed 's/^,//;s/,$//;s/,,*/,/g')
prev=""
while true; do
  cur=$(sacct -j "$IDS" -X --format=JobName%16,State%14 -n 2>/dev/null \
        | awk '$2!="RUNNING" && $2!="PENDING" && NF {print $1": "$2}' | sort)
  comm -13 <(echo "$prev") <(echo "$cur"); prev="$cur"
  [ "$(echo "$cur" | grep -c ':')" -ge 3 ] && {
    for y in 315 340 20; do
      n=$(ls $O/unicorn_effect1_y${y}_e12/*.png 2>/dev/null | wc -l)
      [ "$n" -ge 140 ] && echo "PASS yaw $y: $n renders" || echo "FAIL yaw $y: only $n renders"
    done
    echo "ALL TERMINAL"; break; }
  sleep 45
done
