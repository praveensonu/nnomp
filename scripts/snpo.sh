#!/bin/bash

# Store the PID for easier process management
nohup python /raid/p.bushipaka/emnlp/unlearning/simnpo.py &> /raid/p.bushipaka/emnlp/logs/simnpo.log &
PID=$!

# Save PID to file for later reference
echo $PID > /raid/p.bushipaka/emnlp/logs/simnpo.pid

# Inform user
echo "Launched simnpo.py in background (PID: $PID)"
echo "Logs at: /raid/p.bushipaka/emnlp/logs/simnpo.log"
echo "To kill: kill $PID  or  kill \$(cat /raid/p.bushipaka/emnlp/logs/simnpo.pid)"