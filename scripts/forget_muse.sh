#!/bin/bash

# Store the PID for easier process management
nohup python /home/praveen/nnomp/forget_selection_muse.py &> /home/praveen/nnomp/logs/forget_selection_muse.log &
PID=$!

# Save PID to file for later reference
echo $PID > /home/praveen/nnomp/logs/forget_selection_muse.pid

# Inform user
echo "Launched forget_selection_muse.py in background (PID: $PID)"
echo "Logs at: /home/praveen/nnomp/logs/forget_selection_muse.log"
echo "To kill: kill $PID  or  kill \$(cat /home/praveen/nnomp/logs/forget_selection_muse.pid)"