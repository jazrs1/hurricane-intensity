#!/usr/bin/env bash
# Rotation-augmentation experiment: masked control vs masked + rotation, ten seeds each, L1, nine training storms,
# 40 epochs. Design and approval: logs/decisions.md, 2026-10-14. Arm order alternates each round (C/R, R/C, ...).
# Keeps only each run's validation-selected checkpoint. Never writes best_model.pt, models/baseline_9kt.pt,
# results/heldout_predictions.csv or site/.
cd "$(dirname "$0")/.." || exit 1
PY=./.venv/Scripts/python.exe
TRAIN=storms_train_baseline.txt
LOG=logs/rotation; RES=results/rotation; MOD=models/rotation
mkdir -p "$LOG" "$RES" "$MOD/control" "$MOD/rotation"
Q="timestamp,temperature.gpu,clocks.current.sm,power.draw,utilization.gpu,clocks_event_reasons.sw_power_cap,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown"
snap() { nvidia-smi --query-gpu=$Q --format=csv,noheader | tr -d '\r'; }
sha256sum best_model.pt models/baseline_9kt.pt > "$LOG/sha256_before.txt"
ORDER=0

run() {  # arm seed
  local arm=$1 seed=$2 extra="--circular-mask"
  [ "$arm" = rotation ] && extra="--circular-mask --rotate"
  ORDER=$((ORDER + 1))
  local ckpt="$MOD/$arm/seed$seed.pt" log="$LOG/${arm}_seed$seed.log"
  echo "RUN arm=$arm seed=$seed order=$ORDER list=$TRAIN epochs=40 flags=$extra" > "$log"
  echo "START $(date -Is) | $(snap)" >> "$log"
  $PY -u scripts/train.py --train-list "$TRAIN" --epochs 40 --seed "$seed" --loss l1 $extra --out "$ckpt" >> "$log" 2>&1 || { echo "TRAIN_FAILED" >> "$log"; echo "TRAIN_FAILED $arm $seed"; return; }
  echo "END $(date -Is) | $(snap)" >> "$log"
  $PY scripts/evaluate.py --checkpoint "$ckpt" --train-list "$TRAIN" --holdout-list storms_holdout_h6.txt --circular-mask --out "$RES/${arm}_seed${seed}_h6.csv" > "$LOG/${arm}_seed${seed}_eval_h6.log" 2>&1 || echo "EVAL_FAILED $arm $seed"
  $PY scripts/storm_fraction.py --checkpoint "$ckpt" --circular-mask --out "$RES/${arm}_seed${seed}_storm_fraction.json" > "$LOG/${arm}_seed${seed}_storm_fraction.log" 2>&1 || echo "FRACTION_FAILED $arm $seed"
  echo "$arm seed $seed (order $ORDER) done $(date +%T): H6 $(grep 'overall MAE' "$LOG/${arm}_seed${seed}_eval_h6.log" | tr -s ' ') | $(grep 'same-storm' "$LOG/${arm}_seed${seed}_storm_fraction.log" | cut -d' ' -f1-8)"
}

ROT=("control rotation" "rotation control")
for s in 0 1 2 3 4 5 6 7 8 9; do
  for arm in ${ROT[$((s % 2))]}; do
    run "$arm" "$s"
  done
done

sha256sum best_model.pt models/baseline_9kt.pt
cat "$LOG/sha256_before.txt"
echo "ALL DONE $(date -Is)"
