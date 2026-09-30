#!/usr/bin/env bash
# Background runner for fetch_matchups.py (2017-18 through 2025-26).
#   nohup scripts/run_matchups.sh >/dev/null 2>&1 &
# - Logs to logs/matchups.log, writes its PID to logs/matchups.pid.
# - caffeinate -i keeps the Mac from idle-sleeping while it runs (closing the lid still sleeps it).
# - fetch_matchups.py skips every game already cached, so re-running always resumes.
# - If the fetcher stops after a streak of failures (network dropped, laptop slept, rate limit),
#   wait 15 minutes and resume, up to 12 times.
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs
exec >>logs/matchups.log 2>&1
echo $$ > logs/matchups.pid
echo "=== runner started $(date) (pid $$)"
for attempt in $(seq 1 12); do
  caffeinate -i .venv/bin/python -u scripts/fetch_matchups.py --seasons 2017 2025
  code=$?
  echo "=== fetcher exited with code $code at $(date) (attempt $attempt)"
  [ "$code" -eq 0 ] && break
  [ "$code" -eq 130 ] && break   # stopped by hand
  echo "=== waiting 15 minutes before resuming"
  sleep 900
done
rm -f logs/matchups.pid
echo "=== runner finished $(date)"
