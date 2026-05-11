#!/bin/bash

# ===== Paths =====
BASE_DIR="/raid/p.bushipaka/emnlp"
SCRIPT_RUN="$BASE_DIR/eval/run_lm_eval_batch.py" 
SCRIPT_SUMMARY="$BASE_DIR/eval/summarize_lm_eval.py"
CONFIG="/raid/p.bushipaka/emnlp/eval/models.yaml"
RESULTS_DIR="$BASE_DIR/eval/results_mu/"
LOG_DIR="$BASE_DIR/logs"

# ===== Setup =====
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/lm_eval.log"
PID_FILE="$LOG_DIR/lm_eval.pid"

# Activate environment (edit if needed)
source $BASE_DIR/.venv/bin/activate

# ===== Run in background =====
nohup bash -c "
echo 'Starting lm-eval batch run at \$(date)'

python $SCRIPT_RUN $CONFIG

echo 'Batch run finished at \$(date)'

echo 'Starting summarization at \$(date)'

python $SCRIPT_SUMMARY $RESULTS_DIR

echo 'Summarization finished at \$(date)'
" &> "$LOG_FILE" &

PID=$!

# ===== Save PID =====
echo $PID > "$PID_FILE"

# ===== User info =====
echo "Launched lm-eval batch in background (PID: $PID)"
echo "Logs at: $LOG_FILE"
echo "PID file: $PID_FILE"
echo "To monitor: tail -f $LOG_FILE"
echo "To kill: kill $PID  or  kill \$(cat $PID_FILE)"