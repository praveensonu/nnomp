#!/bin/bash

# Store the PID for easier process management
nohup python /raid/p.bushipaka/emnlp/unlearning/gd.py &> /raid/p.bushipaka/emnlp/logs/gd.log &
PID=$!

# Save PID to file for later reference
echo $PID > /raid/p.bushipaka/emnlp/logs/gd.pid

# Inform user
echo "Launched gd.py in background (PID: $PID)"
echo "Logs at: /raid/p.bushipaka/emnlp/logs/gd.log"
echo "To kill: kill $PID  or  kill \$(cat /raid/p.bushipaka/emnlp/logs/gd.pid)"