#!/bin/bash

# Store the PID for easier process management
nohup python /raid/p.bushipaka/emnlp/trek_bio.py &> /raid/p.bushipaka/emnlp/logs/trek_bio.log &
PID=$!

# Save PID to file for later reference
echo $PID > /raid/p.bushipaka/emnlp/logs/trek_bio.pid

# Inform user
echo "Launched trek_bio.py in background (PID: $PID)"
echo "Logs at: /raid/p.bushipaka/emnlp/logs/trek_bio.log"
echo "To kill: kill $PID  or  kill \$(cat /raid/p.bushipaka/emnlp/logs/trek_bio.pid)"