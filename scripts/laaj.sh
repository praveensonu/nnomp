#!/bin/bash

# Store the PID for easier process management
nohup python /home/praveen/nnomp/eval/laaj.py &> /home/praveen/nnomp/logs/laaj.log &
PID=$!

# Save PID to file for later reference
echo $PID > /home/praveen/nnomp/logs/laaj.pid

# Inform user
echo "Launched laaj.py in background (PID: $PID)"
echo "Logs at: /home/praveen/nnomp/logs/laaj.log"
echo "To kill: kill $PID  or  kill \$(cat /home/praveen/nnomp/logs/laaj.pid)"