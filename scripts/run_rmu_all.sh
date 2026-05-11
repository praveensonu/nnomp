#!/bin/bash

# Store the PID for easier process management
nohup python /home/praveen/nnomp/unlearning/run_rmu.py &> /home/praveen/nnomp/logs/run_rmu_all.log &
PID=$!

# Save PID to file for later reference
echo $PID > /home/praveen/nnomp/logs/run_rmu_all.pid

# Inform user
echo "Launched gd.py in background (PID: $PID)"
echo "Logs at: /home/praveen/nnomp/logs/run_rmu_all.log"
echo "To kill: kill $PID  or  kill \$(cat /home/praveen/nnomp/logs/run_rmu_all.pid)"