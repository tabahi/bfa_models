#!/usr/bin/env bash
# pb02 sweep: the ranked next steps in memory.md, one arm per run, sequential on both GPUs.
# Every arm is pb01b (ramp 2.0, soft_onset_ramp_k 0, repr_flux_weight 1.0, warm from pa07f)
# with ONE knob changed, so each is an A/B against the same curve. Runs are sequential, never
# parallel: splitting the GPUs changes the effective batch and the steps per epoch, and then
# the arms are no longer comparable to pb01b or to each other.
#
#   bash p3cupe/run_pb02.sh            # all arms
#   bash p3cupe/run_pb02.sh pb02b pb02f   # a subset, in the order given
#
# Logs: tmp/training/p3cupe/logs/<arm>.log. A failed arm is reported and the queue moves on.
set -u
cd "$(dirname "$0")/.."
PY=/root/anaconda3/bin/python
mkdir -p tmp/training/p3cupe/logs

declare -A ARM=(
  # step 3: the structural test - can the head use the change the embedding now carries?
  [pb02a]="local_delta=1"
  # step 1: ramp back above the 38.9 ms receptive-field floor (2.0 = +-10 ms, 3.0 = +-15 ms)
  [pb02b]="local_delta=0 ramp_frames=3.0"
  # control: pb01b itself for all 10 epochs (it stopped at e4); the late-epoch slope reference
  [pb02f]="local_delta=0"
  # step 2: does dip ever follow flux as the repr term gets stronger / weaker?
  [pb02c]="local_delta=0 repr_flux_weight=3.0"
  [pb02d]="local_delta=0 repr_flux_weight=0.3"
  # step 1, second point - only informative if pb02b beat pb02f
  [pb02e]="local_delta=0 ramp_frames=4.0"
  # every pb02 arm peaks at e0 and declines; is the decline lr-driven drift of the EM loop?
  [pb02g]="local_delta=0 lr=1e-5"
  # the loop's fixed point sits below the warm start (pb02g); time-warp averaging of the
  # target onsets is the one implemented signal that does not come from the model itself
  [pb02h]="local_delta=0 aug_views=2 aug_warp_range=0.05"
  # the onset bonus in the TARGETS: the decoder sweep says it is monotonically worse per phone;
  # pa03a (bonus 0) predates the EMA teacher that now does the anchoring
  [pb02i]="local_delta=0 boundary_bonus_weight=3.0"
  [pb02j]="local_delta=0 boundary_bonus_weight=0.0"
)
ORDER=(pb02a pb02b pb02f pb02c pb02d pb02e)
[ $# -gt 0 ] && ORDER=("$@")

for arm in "${ORDER[@]}"; do
  args=${ARM[$arm]:?unknown arm $arm}
  log=tmp/training/p3cupe/logs/$arm.log
  # Wait for both cards to be empty: on 2026-09-23 an outside job held 42 GB and every arm
  # after pb02a died of OOM within 20 s, emptying the whole queue in two minutes.
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0,1 \
             | awk '$1 > 2000' | wc -l)" -gt 0 ]; do
    echo "[$(date '+%F %T')] $arm waiting for free GPUs"; sleep 300
  done
  echo "[$(date '+%F %T')] start $arm: $args -> $log"
  # shellcheck disable=SC2086
  $PY p3cupe/train_cupe.py p3cupe/cupe_hp experiment=$arm "note=pb01b + $args" $args > "$log" 2>&1
  rc=$?
  echo "[$(date '+%F %T')] end   $arm: exit $rc"
  grep -h "no-bonus f1@25" "$log" | tail -1
done
