#!/bin/bash
set -euo pipefail

RUNTIME_ROOT="/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/experiments/foreswe_weekly_reproduction_full512"
MAIN_LOG="$RUNTIME_ROOT/logs/weekly_att_interactive.log"
TRAIN_LOG="$RUNTIME_ROOT/outputs/forecasting_modified_weekly_att_tr_act_loss.txt"
HEARTBEAT="$RUNTIME_ROOT/logs/weekly_att_interactive_heartbeat.log"
RUN_SESSION="weekly_full512_debug"

previous_main_mtime=0
previous_train_mtime=0

while true; do
  timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  session_exists="no"
  status="waiting"

  if tmux list-sessions 2>/dev/null | rg -q "^${RUN_SESSION}:"; then
    session_exists="yes"
    status="running"
  fi

  main_mtime=0
  train_mtime=0
  if [ -f "$MAIN_LOG" ]; then
    main_mtime="$(stat -c %Y "$MAIN_LOG")"
  fi
  if [ -f "$TRAIN_LOG" ]; then
    train_mtime="$(stat -c %Y "$TRAIN_LOG")"
  fi

  if [ "$session_exists" = "yes" ] && [ "$main_mtime" = "$previous_main_mtime" ] && [ "$train_mtime" = "$previous_train_mtime" ]; then
    status="stalled"
  fi

  {
    echo "[$timestamp] status=$status session_exists=$session_exists"
    if [ -f "$MAIN_LOG" ]; then
      echo "[main tail]"
      tail -n 5 "$MAIN_LOG"
    fi
    if [ -f "$TRAIN_LOG" ]; then
      echo "[train tail]"
      tail -n 5 "$TRAIN_LOG"
    fi
    echo
  } >> "$HEARTBEAT"

  previous_main_mtime="$main_mtime"
  previous_train_mtime="$train_mtime"
  sleep 600
done
