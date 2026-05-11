#!/bin/bash

nohup python -u /home/praveen/nnomp/forget_selection_bio.py &> /home/praveen/nnomp/logs/forget_selection_bio.log &
PID=$!

echo $PID > /home/praveen/nnomp/logs/forget_selection_bio.pid

echo "Launched forget_selection_bio.py in background (PID: $PID)"
echo "Logs at: /home/praveen/nnomp/logs/forget_selection_bio.log"
echo "To kill: kill $PID  or  kill \$(cat /home/praveen/nnomp/logs/forget_selection_bio.pid)"