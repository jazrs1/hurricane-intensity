#!/usr/bin/env bash
# More-distinct-storms experiment: control (original nine storms, 40 epochs) vs expanded (nine + ten new storms,
# 25 epochs), ten seeds each, L1, unmasked, unrotated, both arms selecting on storms_val_expanded.txt, evaluated
# on H9. Design: logs/decisions.md, 2026-10-14 "More-distinct-storms experiment, design" and the entries after it.
# Arm order alternates each round (C/E, E/C, ...). Keeps only each run's validation-selected checkpoint.
# Never writes best_model.pt, models/baseline_9kt.pt, results/heldout_predictions.csv or site/.
cd "$(dirname "$0")/.." || exit 1
PY=./.venv/Scripts/python.exe
VAL=storms_val_expanded.txt
HOLD=storms_holdout_h9.txt
LOG=logs/storms; RES=results/storms; MOD=models/storms
mkdir -p "$LOG" "$RES" "$MOD/control" "$MOD/expanded"
Q="timestamp,temperature.gpu,clocks.current.sm,power.draw,utilization.gpu,clocks_event_reasons.sw_power_cap,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown"
snap() { nvidia-smi --query-gpu=$Q --format=csv,noheader | tr -d '\r'; }
sha256sum best_model.pt models/baseline_9kt.pt > "$LOG/sha256_before.txt"
sha256sum scripts/storms_analysis.py > "$LOG/analysis_sha256_at_launch.txt"
ORDER=0

run() {  # arm seed
  local arm=$1 seed=$2 list epochs
  if [ "$arm" = control ]; then list=storms_train_control2.txt; epochs=40; else list=storms_train_expanded.txt; epochs=25; fi
  ORDER=$((ORDER + 1))
  local ckpt="$MOD/$arm/seed$seed.pt" log="$LOG/${arm}_seed$seed.log"
  echo "RUN arm=$arm seed=$seed order=$ORDER list=$list val=$VAL epochs=$epochs" > "$log"
  echo "START $(date -Is) | $(snap)" >> "$log"
  $PY -u scripts/train.py --train-list "$list" --val-list "$VAL" --epochs "$epochs" --seed "$seed" --loss l1 --out "$ckpt" >> "$log" 2>&1 || { echo "TRAIN_FAILED" >> "$log"; echo "TRAIN_FAILED $arm $seed"; return; }
  echo "END $(date -Is) | $(snap)" >> "$log"
  $PY scripts/evaluate.py --checkpoint "$ckpt" --train-list "$list" --holdout-list "$HOLD" --out "$RES/${arm}_seed${seed}_h9.csv" > "$LOG/${arm}_seed${seed}_eval_h9.log" 2>&1 || echo "EVAL_FAILED $arm $seed"
  # secondary: same-storm fraction on the original nine storms' 1307 training frames (storm_fraction.py's fixed frame set)
  $PY scripts/storm_fraction.py --checkpoint "$ckpt" --out "$RES/${arm}_seed${seed}_storm_fraction.json" > "$LOG/${arm}_seed${seed}_storm_fraction.log" 2>&1 || echo "FRACTION_FAILED $arm $seed"
  echo "$arm seed $seed (order $ORDER) done $(date +%T): H9 $(grep 'overall MAE' "$LOG/${arm}_seed${seed}_eval_h9.log" | tr -s ' ') | $(grep 'same-storm' "$LOG/${arm}_seed${seed}_storm_fraction.log" | cut -d' ' -f1-8)"
}

ROT=("control expanded" "expanded control")
for s in 0 1 2 3 4 5 6 7 8 9; do
  for arm in ${ROT[$((s % 2))]}; do
    run "$arm" "$s"
  done
done

sha256sum best_model.pt models/baseline_9kt.pt
cat "$LOG/sha256_before.txt"
sha256sum scripts/storms_analysis.py
cat "$LOG/analysis_sha256_at_launch.txt"
echo "ALL DONE $(date -Is)"
