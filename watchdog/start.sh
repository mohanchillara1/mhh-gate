#!/bin/bash
# Canonical watchdog entry point. Run after any pod start: bash /workspace/watchdog/start.sh
# Launches the watchdog loop and the keeper (relaunches watchdog if its lock goes stale).
# cron does not exist in this container and would not survive a pod stop anyway —
# only /workspace persists. After a pod STOP/START a human (or Claude) runs this once.
W=/workspace
setsid nohup $W/watchdog/watchdog.sh >> $W/watchdog.log 2>&1 < /dev/null &
sleep 2
if ! pgrep -f "[k]eeper-loop-mhh" >/dev/null; then
  setsid nohup bash -c 'exec -a keeper-loop-mhh bash -c "while true; do
    kill -0 \$(cat /workspace/watchdog.pid 2>/dev/null) 2>/dev/null || \
      setsid /workspace/watchdog/watchdog.sh >> /workspace/watchdog.log 2>&1 < /dev/null &
    sleep 300; done"' >> $W/watchdog.log 2>&1 < /dev/null &
fi
echo "watchdog + keeper launched; STATUS at $W/STATUS.md"
