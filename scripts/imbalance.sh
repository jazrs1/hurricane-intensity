#!/usr/bin/env bash
# Class-imbalance experiment: three arms x ten seeds, interleaved by seed.
# Never writes best_model.pt or results/heldout_predictions.csv.
cd "$(dirname "$0")/.." || exit 1
PY=./.venv/Scripts/python.exe
LOG=logs/imbalance; RES=results/imbalance; MOD=models/imbalance
mkdir -p "$LOG" "$RES" "$MOD"
Q="timestamp,temperature.gpu,clocks.current.sm,power.draw,utilization.gpu,clocks_event_reasons.sw_power_cap,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown"
snap() { nvidia-smi --query-gpu=$Q --format=csv,noheader | tr -d '\r'; }
sha256sum best_model.pt > "$LOG/best_model_sha256_before.txt"
ORDER=0

run() {  # arm seed train_list epochs
  local arm=$1 seed=$2 list=$3 epochs=$4
  ORDER=$((ORDER + 1))
  local dir="$MOD/$arm/seed$seed" log="$LOG/${arm}_seed$seed.log"
  echo "RUN arm=$arm seed=$seed order=$ORDER list=$list epochs=$epochs" > "$log"
  echo "START $(date -Is) | $(snap)" >> "$log"
  $PY -u scripts/train.py --train-list "$list" --epochs "$epochs" --seed "$seed" --out "$dir/best_val.pt" --epoch-dir "$dir" >> "$log" 2>&1 || { echo "TRAIN_FAILED" >> "$log"; echo "TRAIN_FAILED $arm $seed"; return; }
  echo "END $(date -Is) | $(snap)" >> "$log"
  $PY scripts/evaluate.py --checkpoint "$dir/best_val.pt" --holdout-list storms_holdout_baseline.txt --out "$RES/${arm}_seed${seed}_holdout3.csv" > "$LOG/${arm}_seed${seed}_eval_holdout3.log" 2>&1 || echo "EVAL_FAILED $arm $seed holdout3"
  $PY scripts/evaluate.py --checkpoint "$dir/best_val.pt" --holdout-list storms_holdout.txt --out "$RES/${arm}_seed${seed}_holdout4.csv" > "$LOG/${arm}_seed${seed}_eval_holdout4.log" 2>&1 || echo "EVAL_FAILED $arm $seed holdout4"
  echo "$arm seed $seed (order $ORDER) done $(date +%T): $(grep 'overall MAE' "$LOG/${arm}_seed${seed}_eval_holdout3.log" | tr -s ' ') | 4-storm $(grep 'overall MAE' "$LOG/${arm}_seed${seed}_eval_holdout4.log" | tr -s ' ')"
}

for s in 0 1 2 3 4 5 6 7 8 9; do
  run A_baseline_40ep   $s storms_train_baseline.txt 40
  run B_expanded_29ep   $s storms_train.txt 29
  run C_expanded_40ep   $s storms_train.txt 40
done

sha256sum best_model.pt
cat "$LOG/best_model_sha256_before.txt"
echo "ALL DONE $(date -Is)"
