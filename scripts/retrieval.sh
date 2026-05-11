#!/bin/bash

# Store the PID for easier process management
nohup python /raid/p.bushipaka/emnlp/trek.py &> /raid/p.bushipaka/emnlp/logs/trek.log &
PID=$!

# Save PID to file for later reference
echo $PID > /raid/p.bushipaka/emnlp/logs/trek.pid

# Inform user
echo "Launched trek.py in background (PID: $PID)"
echo "Logs at: /raid/p.bushipaka/emnlp/logs/trek.log"
echo "To kill: kill $PID  or  kill \$(cat /raid/p.bushipaka/emnlp/logs/trek.pid)"