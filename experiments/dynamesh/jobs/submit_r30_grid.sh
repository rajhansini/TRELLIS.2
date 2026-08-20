#!/bin/bash
# Submit the 4x4 beta grid for ONE object, from its probe's measured ratios.
# usage: submit_r30_grid.sh <obj>
set -eo pipefail
O=$1
E=/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh
T2=/net/projects/ranalab/rajhansini/TRELLIS.2
L=$(ls -t $E/out/r30probe_${O}_*.log 2>/dev/null | head -1)
[ -n "$L" ] || { echo "no probe log for $O"; exit 1; }
R=$(grep -oE "loss/KLc=[0-9.e+-]+ +loss/KLs=[0-9.e+-]+" "$L" | tail -1)
[ -n "$R" ] || { echo "$O: probe has no ratio line yet"; exit 1; }
RC=$(echo "$R" | grep -oE "KLc=[0-9.e+-]+" | cut -d= -f2)
RS=$(echo "$R" | grep -oE "KLs=[0-9.e+-]+" | cut -d= -f2)
echo "$O  loss/KLc=$RC  loss/KLs=$RS"
case $O in
  spot_lava)      M=$T2/data/spot_lava/mesh/spot_render_frame.obj; NF=150; TG=$E/out/gt_targets_spot_lava/frames;;
  teapot_lava2)   M=/net/projects/ranalab/rajhansini/TRELLIS/render/out/f0075/frozen_f0075.ply; NF=150; TG=$E/out/gt_targets_teapot_lava2/frames;;
  # The regenerated porcelain video, prompt fixed so the body stays grey instead
  # of turning white. Needs its own case: 150 frames not 121, and neither the
  # mesh nor the targets carry the _guan suffix the default branch assumes, so
  # falling through would point at paths that do not exist.
  # These two carry no _guan suffix on either the mesh or the targets, and run at
  # 150 frames. The default branch below would point at gt_targets_<obj>_guan/ and
  # <obj>_render_frame_guan.obj, neither of which exists for them.
  monster_rainbow|skull_lava)
                  M=$T2/data/$O/mesh/${O}_render_frame.obj; NF=150; TG=$E/out/gt_targets_${O}/frames;;
  teapot_porcelain_correct|teapot_ceramic_crack_correct)
                  M=$T2/data/$O/mesh/${O}_render_frame.obj; NF=150; TG=$E/out/gt_targets_${O}/frames;;
  *)              M=$T2/data/$O/mesh/${O}_render_frame_guan.obj; NF=121; TG=$E/out/gt_targets_${O}_guan/frames;;
esac
# Fail loudly rather than launching 16 jobs at a path that does not exist.
[ -f "$M" ]  || { echo "$O: mesh missing: $M"; exit 1; }
[ -d "$TG" ] || { echo "$O: targets missing: $TG"; exit 1; }
G=$T2/data/$O/frames_from_video
for FC in 15 25 50 75; do
 for FS in 15 25 50 75; do
  BC=$(python3 -c "print(f'{$FC/100/(1-$FC/100)*$RC:.6g}')")
  BS=$(python3 -c "print(f'{$FS/100/(1-$FS/100)*$RS:.6g}')")
  F=$E/jobs/r30grid_${O}_c${FC}_s${FS}.sbatch
  bash $E/jobs/mk_r30.sh "$O" "$M" "$G" "$TG" "$NF" "r30g" \
       "--w-kl $BC --w-kl-self $BS" 30 > $F
  grep -q -- "--w-kl $BC --w-kl-self $BS" $F || { echo "PATCH FAIL $O $FC $FS"; continue; }
  grep -q -- "--kl-probe" $F && { echo "PROBE FLAG LEAKED $O $FC $FS"; continue; }
  bash -n $F || continue
  J=$(sbatch --parsable $F)
  /net/projects/ranalab/rajhansini/joblog.sh add "$J" "r30g_${O}_c${FC}_s${FS}" \
    "rung30 GRID mcfm=v2_D ${O}: w_kl=$BC (~${FC}% cross) w_kl_self=$BS (~${FS}% self), 30ep/${NF}fr | PASS: 30/30 + final_eval.json + unique run dir"
 done
done
echo "$O: 16 cells submitted"
