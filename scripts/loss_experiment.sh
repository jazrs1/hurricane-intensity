#!/usr/bin/env bash
# Loss-function experiment: L1 / Huber / MSE x ten seeds on the original nine training storms, 40 epochs,
# run round-robin by seed with rotated arm order. Design: logs/decisions.md, 2026-10-14 "Loss-function
# experiment, design" and the approval entry after it.
# Keeps only each run's validation-selected checkpoint. Never writes best_model.pt, models/baseline_9kt.pt,
# results/heldout_predictions.csv or site/.
cd "$(dirname "$0")/.." || exit 1
PY=./.venv/Scripts/python.exe
TRAIN=storms_train_baseline.txt
LOG=logs/loss; RES=results/loss; MOD=models/loss
mkdir -p "$LOG" "$RES" "$MOD"
Q="timestamp,temperature.gpu,clocks.current.sm,power.draw,utilization.gpu,clocks_event_reasons.sw_power_cap,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown"
snap() { nvidia-smi --query-gpu=$Q --format=csv,noheader | tr -d '\r'; }
sha256sum best_model.pt models/baseline_9kt.pt > "$LOG/sha256_before.txt"
ORDER=0

run() {  # loss seed
  local loss=$1 seed=$2
  ORDER=$((ORDER + 1))
  local ckpt="$MOD/$loss/seed$seed.pt" log="$LOG/${loss}_seed$seed.log"
  mkdir -p "$MOD/$loss"
  echo "RUN loss=$loss seed=$seed order=$ORDER list=$TRAIN epochs=40" > "$log"
  echo "START $(date -Is) | $(snap)" >> "$log"
  $PY -u scripts/train.py --train-list "$TRAIN" --epochs 40 --seed "$seed" --loss "$loss" --out "$ckpt" >> "$log" 2>&1 || { echo "TRAIN_FAILED" >> "$log"; echo "TRAIN_FAILED $loss $seed"; return; }
  echo "END $(date -Is) | $(snap)" >> "$log"
  for set in h3 h6; do
    local list=storms_holdout_baseline.txt; [ "$set" = h6 ] && list=storms_holdout_h6.txt
    $PY scripts/evaluate.py --checkpoint "$ckpt" --train-list "$TRAIN" --holdout-list "$list" --out "$RES/${loss}_seed${seed}_$set.csv" > "$LOG/${loss}_seed${seed}_eval_$set.log" 2>&1 || echo "EVAL_FAILED $loss $seed $set"
  done
  echo "$loss seed $seed (order $ORDER) done $(date +%T): H3 $(grep 'overall MAE' "$LOG/${loss}_seed${seed}_eval_h3.log" | tr -s ' ') | H6 $(grep 'overall MAE' "$LOG/${loss}_seed${seed}_eval_h6.log" | tr -s ' ')"
}

# Round-robin by seed, with the arm order rotated each round (L/H/M, H/M/L, M/L/H, repeating),
# so no arm always runs first in a round or always follows the same arm.
ROT=("l1 huber mse" "huber mse l1" "mse l1 huber")
for s in 0 1 2 3 4 5 6 7 8 9; do
  for loss in ${ROT[$((s % 3))]}; do
    run "$loss" "$s"
  done
done

sha256sum best_model.pt models/baseline_9kt.pt
cat "$LOG/sha256_before.txt"
echo "ALL DONE $(date -Is)"
