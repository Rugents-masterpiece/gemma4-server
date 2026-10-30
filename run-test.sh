#!/usr/bin/env bash
# detached launcher: nohup bash run-test.sh <args> -- logs driver output, survives tool timeouts
cd /home/rugent/gemma4-server
echo "launch: $*"
nohup ./test-pool.sh "$@" > "logs/driver-$1-$2-$3-$4.log" 2>&1 &
echo "pid $!"
sleep 2
tail -1 logs/driver-run.log 2>/dev/null || true
