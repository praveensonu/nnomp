#!/bin/bash

# Store the PID for easier process management
nohup python /home/praveen/nnomp/finetune.py &> /home/praveen/nnomp/logs/finetune.log &
PID=$!

# Save PID to file for later reference
echo $PID > /home/praveen/nnomp/logs/finetune.pid

# Inform user
echo "Launched finetune.py in background (PID: $PID)"
echo "Logs at: /home/praveen/nnomp/logs/finetune.log"
echo "To kill: kill $PID  or  kill \$(cat /home/praveen/nnomp/logs/finetune.pid)"