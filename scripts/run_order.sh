#!/usr/bin/env bash
# Run-order / thermal test, then the 60-epoch traces. Baseline config throughout
# (original nine training storms). Never writes best_model.pt.
cd "$(dirname "$0")/.." || exit 1
PY=./.venv/Scripts/python.exe
LOG=logs/run_order; RES=results/run_order; MOD=models/run_order
mkdir -p "$LOG" "$RES" "$MOD" logs/epoch_traces_60 models/epoch_traces_60
Q="timestamp,temperature.gpu,clocks.current.graphics,clocks.current.sm,clocks.current.memory,power.draw,fan.speed,utilization.gpu,pstate,clocks_event_reasons.active,clocks_event_reasons.sw_power_cap,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.hw_slowdown"

sha256sum best_model.pt > "$LOG/best_model_sha256_before.txt"
nvidia-smi --query-gpu=$Q --format=csv -lms 2000 > "$LOG/gpu_telemetry.csv" &
TELE=$!
snap() { nvidia-smi --query-gpu=$Q --format=csv,noheader | tr -d '\r'; }
counters() { nvidia-smi -q | tr -d '\r' | grep -A6 "Counters" | grep -E "Capping|Slowdown|Sync" | sed 's/  */ /g' | paste -sd ';' -; }
temp() { nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | tr -d ' \r'; }
ORDER=0

run() {  # block seed [extra train.py args...]
  local block=$1 seed=$2; shift 2
  ORDER=$((ORDER + 1))
  local log="$LOG/${block}_seed${seed}.log"
  echo "RUN block=$block seed=$seed order=$ORDER" > "$log"
  echo "START $(date -Is) | $(snap)" >> "$log"
  echo "COUNTERS_START $(counters)" >> "$log"
  $PY -u scripts/train.py --train-list storms_train_baseline.txt --seed "$seed" "$@" >> "$log" 2>&1 || echo "TRAIN_FAILED" >> "$log"
  echo "END $(date -Is) | $(snap)" >> "$log"
  echo "COUNTERS_END $(counters)" >> "$log"
  echo "$block seed $seed (order $ORDER) done at $(date +%T), gpu $(temp) C"
}

evaluate() {  # block seed
  $PY scripts/evaluate.py --checkpoint "$MOD/$1_seed$2.pt" --holdout-list storms_holdout_baseline.txt \
      --out "$RES/$1_seed$2.csv" > "$LOG/$1_seed$2_eval.log" 2>&1 || echo "EVAL_FAILED $1 $2"
}

idle() {  # at least 3 minutes, then until the GPU is back near its idle temperature (cap: 5 more minutes)
  local t0=$(temp) n=0
  sleep 180
  while [ "$(temp)" -gt 40 ] && [ $n -lt 30 ]; do sleep 10; n=$((n + 1)); done
  echo "IDLE $(date -Is) from ${t0}C to $(temp)C after $((180 + n * 10))s" >> "$LOG/idle.log"
}

echo "=== spaced block: each run preceded by idle ==="
for s in 0 1 2 3 4; do
  idle
  run spaced $s --out "$MOD/spaced_seed$s.pt"
  evaluate spaced $s
done

echo "=== back-to-back block: no idle ==="
for s in 0 1 2 3 4; do
  run b2b $s --out "$MOD/b2b_seed$s.pt"
  evaluate b2b $s
done

echo "=== 60-epoch traces ==="
for s in 0 1 2; do
  run long60 $s --epochs 60 --out "models/epoch_traces_60/seed$s/best_val.pt" --epoch-dir "models/epoch_traces_60/seed$s"
  cp "$LOG/long60_seed$s.log" "logs/epoch_traces_60/train_seed$s.log"
done

kill $TELE 2>/dev/null
sha256sum best_model.pt
cat "$LOG/best_model_sha256_before.txt"
echo "ALL DONE $(date -Is)"
