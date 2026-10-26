#!/bin/bash
set -euo pipefail

SESSION_NAME="${1:-d2_att_debug}"
FSW_RUNTIME_ROOT="${FSW_RUN_ROOT:-/pscratch/sd/h/hyvchen/foreswe-gpu-optimization/runs}/D2_sierra_plus_500km"
MAIN_LOG="$FSW_RUNTIME_ROOT/logs/d2_att_interactive_debug.log"
TRAIN_LOG="$FSW_RUNTIME_ROOT/outputs/forecasting_modified_daily_att_tr_act_loss.txt"
HEARTBEAT="$FSW_RUNTIME_ROOT/logs/d2_att_interactive_debug_heartbeat.log"
STATE_FILE="$FSW_RUNTIME_ROOT/logs/d2_att_interactive_debug_state.txt"

mkdir -p "$FSW_RUNTIME_ROOT/logs"

prev_main_mtime="missing"
prev_train_mtime="missing"

while true; do
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  if tmux list-sessions 2>/dev/null | grep -q "^${SESSION_NAME}:"; then
    session_exists="yes"
  else
    session_exists="no"
  fi

  if [ -f "$MAIN_LOG" ]; then
    main_mtime="$(stat -c %Y "$MAIN_LOG" 2>/dev/null || echo missing)"
    main_tail="$(tail -n 5 "$MAIN_LOG" 2>/dev/null | sed 's/^/  /')"
  else
    main_mtime="missing"
    main_tail="  <main log missing>"
  fi

  if [ -f "$TRAIN_LOG" ]; then
    train_mtime="$(stat -c %Y "$TRAIN_LOG" 2>/dev/null || echo missing)"
    train_tail="$(tail -n 10 "$TRAIN_LOG" 2>/dev/null | sed 's/^/  /')"
  else
    train_mtime="missing"
    train_tail="  <training log missing>"
  fi

  if [ "$session_exists" = "no" ] && [ -f "$TRAIN_LOG" ]; then
    status="completed_or_stopped"
  elif [ "$main_mtime" = "$prev_main_mtime" ] && [ "$train_mtime" = "$prev_train_mtime" ]; then
    status="stalled_or_waiting"
  else
    status="running_or_progressing"
  fi

  {
    echo "[$ts] status=$status session_exists=$session_exists"
    echo "main_log=$MAIN_LOG"
    echo "train_log=$TRAIN_LOG"
    echo "main_mtime=$main_mtime train_mtime=$train_mtime"
    echo "recent main log:"
    echo "$main_tail"
    echo "recent training log:"
    echo "$train_tail"
    echo
  } >> "$HEARTBEAT"

  printf '%s\n' "$status" > "$STATE_FILE"

  prev_main_mtime="$main_mtime"
  prev_train_mtime="$train_mtime"

  sleep 600
done
