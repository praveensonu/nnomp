#!/bin/bash

# Store the PID for easier process management
nohup python /home/praveen/nnomp/eval/eval_auto.py &> /home/praveen/nnomp/logs/eval_auto.log &
PID=$!

# Save PID to file for later reference
echo $PID > /home/praveen/nnomp/logs/eval_auto.pid

# Inform user
echo "Launched eval_auto.py in background (PID: $PID)"
echo "Logs at: /home/praveen/nnomp/logs/eval_auto.log"
echo "To kill: kill $PID  or  kill \$(cat /home/praveen/nnomp/logs/eval_auto.pid)"