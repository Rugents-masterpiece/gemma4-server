#!/usr/bin/env bash
# Keep debug server alive: if it dies, restart, until flag file removed
for i in $(seq 1 100); do
  curl -s -m 3 http://127.0.0.1:18001/health | grep -q ok || {
    echo "restarting server at $(date +%T)"; ./debug-start.sh 2>&1 | tail -1
  }
  sleep 5
done
