#!/bin/bash

# Unbuffer output
exec 1>&1
exec 2>&2

echo "=== ENTRYPOINT SCRIPT STARTING ===" >&2
echo "=== ENTRYPOINT SCRIPT STARTING ==="

echo "Testing basic output..." >&2
echo "Testing basic output..."

echo "Checking Python..." >&2
python --version 2>&1 || echo "Python check failed"

echo "Checking Supervisord..." >&2
which supervisord 2>&1 || echo "Supervisord not found"

echo "Working directory: $(pwd)" >&2
echo "Working directory: $(pwd)"

echo "Checking environment variables..." >&2
env | grep -E "CLAUDE|GEMINI|MONGO|PORT" || echo "No relevant env vars found"

echo "=== STARTING SUPERVISORD ===" >&2
echo "=== STARTING SUPERVISORD ==="

exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
