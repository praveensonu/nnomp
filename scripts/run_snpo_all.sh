#!/bin/bash

# Store the PID for easier process management
nohup python /home/praveen/nnomp/unlearning/run_simnpo.py &> /home/praveen/nnomp/logs/run_snpo_all.log &
PID=$!

# Save PID to file for later reference
echo $PID > /home/praveen/nnomp/logs/simnpo.pid

# Inform user
echo "Launched simnpo.py in background (PID: $PID)"
echo "Logs at: /home/praveen/nnomp/logs/simnpo.log"
echo "To kill: kill $PID  or  kill \$(cat /home/praveen/nnomp/logs/simnpo.pid)"