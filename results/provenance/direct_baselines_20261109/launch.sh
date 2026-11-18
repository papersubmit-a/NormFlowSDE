#!/usr/bin/env bash
# Runs all 36 tasks, 9 at a time, 2 pinned cores each (slot s -> cores 2(s-1), 2(s-1)+1). Longest tasks first.
cd "$(dirname "$0")"
for m in sde_gan latent_sde latent_nsf; do for d in lorenz96 reassemble aloha pendulum; do for s in 0 1 2; do
  echo "$m $d $s"; done; done; done |
parallel -j 9 --colsep ' ' --joblog joblog.txt './run_task.sh {1} {2} {3} $((2*({%}-1))),$((2*({%}-1)+1))'
