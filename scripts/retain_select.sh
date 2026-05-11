#!/bin/bash

# Store the PID for easier process management
nohup python /home/praveen/nnomp/retain_select.py &> /home/praveen/nnomp/logs/retain_select.log &
PID=$!

# Save PID to file for later reference
echo $PID > /home/praveen/nnomp/logs/retain_select.pid

# Inform user
echo "Launched retain_select.py in background (PID: $PID)"
echo "Logs at: /home/praveen/nnomp/logs/retain_select.log"
echo "To kill: kill $PID  or  kill \$(cat /home/praveen/nnomp/logs/retain_select.pid)"