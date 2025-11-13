#!/bin/sh
set -ex

echo "ENTRYPOINT STARTING"
echo "ENTRYPOINT STARTING" >&2

# Start supervisord directly - no extra diagnostics for now
/usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
