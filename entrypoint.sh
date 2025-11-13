#!/bin/bash
set -e

echo "=== Container Starting ==="
echo "Python version: $(python --version)"
echo "Supervisord location: $(which supervisord)"
echo "Working directory: $(pwd)"
echo "Environment variables:"
env | grep -E "CLAUDE|GEMINI|MONGO|PORT" || echo "No relevant env vars found"

echo "=== Starting Supervisord ==="
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
