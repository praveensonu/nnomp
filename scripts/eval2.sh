#!/bin/bash

# Store the PID for easier process management
nohup python /home/praveen/nnomp/eval/eval.py &> /home/praveen/nnomp/logs/eval2.log &
PID=$!

# Save PID to file for later reference
echo $PID > /home/praveen/nnomp/logs/eval2.pid

# Inform user
echo "Launched eval.py in background (PID: $PID)"
echo "Logs at: /home/praveen/nnomp/logs/eval2.log"
echo "To kill: kill $PID  or  kill \$(cat /home/praveen/nnomp/logs/eval2.pid)"