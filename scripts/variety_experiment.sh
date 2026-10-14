#!/usr/bin/env bash
# Variety-against-volume experiment: dense (the original nine storms at hourly) vs sparse (all 19 training storms at
# 3-hourly), each run training on its own frame list (results/variety/frame_lists/<arm>_seed<s>.txt, 1013 frames,
# matched on 5 kt bins, drawn fresh per seed by scripts/variety_subsets.py). Equal steps: 51 epochs x 32 batches.
# L1, unmasked, unrotated; storms_val_expanded.txt for both arms; evaluated on H9. Arm order alternates each round.
# Design: logs/decisions.md, 2026-10-14 "Variety-against-volume experiment". Keeps only each run's selected
# checkpoint. Never writes best_model.pt, models/baseline_9kt.pt, results/heldout_predictions.csv or site/.
cd "$(dirname "$0")/.." || exit 1
PY=./.venv/Scripts/python.exe
TRAIN=storms_train_expanded.txt   # the frame list restricts each arm to its own frames
VAL=storms_val_expanded.txt
HOLD=storms_holdout_h9.txt
EPOCHS=51
LISTS=results/variety/frame_lists
LOG=logs/variety; RES=results/variety; MOD=models/variety
mkdir -p "$LOG" "$RES" "$MOD/dense" "$MOD/sparse"
Q="timestamp,temperature.gpu,clocks.current.sm,power.draw,utilization.gpu,clocks_event_reasons.sw_power_cap,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown"
snap() { nvidia-smi --query-gpu=$Q --format=csv,noheader | tr -d '\r'; }
sha256sum best_model.pt models/baseline_9kt.pt > "$LOG/sha256_before.txt"
sha256sum scripts/variety_analysis.py > "$LOG/analysis_sha256_at_launch.txt"
sha256sum "$LISTS"/*_seed*.txt > "$LOG/frame_lists_sha256_at_launch.txt"
ORDER=0

run() {  # arm seed
  local arm=$1 seed=$2
  ORDER=$((ORDER + 1))
  local ckpt="$MOD/$arm/seed$seed.pt" log="$LOG/${arm}_seed$seed.log" frames="$LISTS/${arm}_seed$seed.txt"
  echo "RUN arm=$arm seed=$seed order=$ORDER list=$TRAIN val=$VAL frames=$frames epochs=$EPOCHS" > "$log"
  echo "START $(date -Is) | $(snap)" >> "$log"
  $PY -u scripts/train.py --train-list "$TRAIN" --val-list "$VAL" --frame-list "$frames" --epochs "$EPOCHS" --seed "$seed" --loss l1 --out "$ckpt" >> "$log" 2>&1 || { echo "TRAIN_FAILED" >> "$log"; echo "TRAIN_FAILED $arm $seed"; return; }
  echo "END $(date -Is) | $(snap)" >> "$log"
  $PY scripts/evaluate.py --checkpoint "$ckpt" --train-list "$TRAIN" --holdout-list "$HOLD" --out "$RES/${arm}_seed${seed}_h9.csv" > "$LOG/${arm}_seed${seed}_eval_h9.log" 2>&1 || echo "EVAL_FAILED $arm $seed"
  # secondary: same-storm fraction on the H9 frames, which neither arm trains on
  $PY scripts/storm_fraction.py --checkpoint "$ckpt" --storms-list "$HOLD" --out "$RES/${arm}_seed${seed}_storm_fraction_h9.json" > "$LOG/${arm}_seed${seed}_storm_fraction.log" 2>&1 || echo "FRACTION_FAILED $arm $seed"
  echo "$arm seed $seed (order $ORDER) done $(date +%T): H9 $(grep 'overall MAE' "$LOG/${arm}_seed${seed}_eval_h9.log" | tr -s ' ') | $(grep 'same-storm' "$LOG/${arm}_seed${seed}_storm_fraction.log" | cut -d' ' -f1-8)"
}

ROT=("dense sparse" "sparse dense")
for s in 0 1 2 3 4 5 6 7 8 9; do
  for arm in ${ROT[$((s % 2))]}; do
    run "$arm" "$s"
  done
done

sha256sum best_model.pt models/baseline_9kt.pt
cat "$LOG/sha256_before.txt"
sha256sum scripts/variety_analysis.py
cat "$LOG/analysis_sha256_at_launch.txt"
sha256sum -c "$LOG/frame_lists_sha256_at_launch.txt" --quiet && echo "frame lists unchanged since launch"
echo "ALL DONE $(date -Is)"
