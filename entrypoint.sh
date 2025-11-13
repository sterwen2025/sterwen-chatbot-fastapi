#!/bin/sh
set -x

# Try to write to a log file in a known location
echo "ENTRYPOINT STARTING" > /tmp/entrypoint.log 2>&1
date >> /tmp/entrypoint.log 2>&1

# Also try stdout/stderr
echo "=== ENTRYPOINT SCRIPT STARTING ==="
echo "=== ENTRYPOINT SCRIPT STARTING ===" >&2

# Check if bash exists, otherwise use sh
if command -v bash >/dev/null 2>&1; then
    echo "Bash found" >> /tmp/entrypoint.log
else
    echo "Bash NOT found, using sh" >> /tmp/entrypoint.log
fi

# Basic system info
echo "Python: $(python --version 2>&1)" >> /tmp/entrypoint.log
echo "Supervisord: $(which supervisord 2>&1)" >> /tmp/entrypoint.log
echo "PWD: $(pwd)" >> /tmp/entrypoint.log

# Check for env vars
env | grep -E "CLAUDE|GEMINI|MONGO|PORT" >> /tmp/entrypoint.log 2>&1 || echo "No env vars" >> /tmp/entrypoint.log

echo "=== STARTING SUPERVISORD ==="
echo "Starting supervisord..." >> /tmp/entrypoint.log

# Start supervisord
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
