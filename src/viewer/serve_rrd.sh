#!/bin/sh
# Serve rerun .rrd recordings over HTTP so they can be viewed in a local browser.
#
# On the server (the machine holding the .rrd files):
#   ./src/viewer/serve_rrd.sh data/save/rrd/*.rrd
#
# On your local machine, forward both printed ports over SSH, e.g.
#   ssh -N -L 9090:localhost:9090 -L 9877:localhost:9877 <user>@<server>
# then open the URL printed below. The `localhost` in the middle of -L is
# resolved on the server; if this script runs on a different node than the one
# you SSH into, use that node's name there instead.
#
# Ports can be overridden:
#   WEB_PORT=9091 DATA_PORT=9878 ./src/viewer/serve_rrd.sh out.rrd
# Keep the local and remote port numbers identical: the web page hardcodes the
# data port into the connect URL.
#
# rerun < 0.23 streams to the web viewer over WebSocket, >= 0.23 over gRPC.
# This script picks the right flag and URL for the installed version.

set -e

WEB_PORT="${WEB_PORT:-9090}"
DATA_PORT="${DATA_PORT:-9877}"
MEMORY_LIMIT="${MEMORY_LIMIT:-4GB}"

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <file.rrd> [more.rrd ...]" >&2
    exit 1
fi

RERUN_VERSION=$(rerun --version | head -n 1 | awk '{print $2}')
RERUN_MAJOR=$(echo "$RERUN_VERSION" | cut -d. -f1)
RERUN_MINOR=$(echo "$RERUN_VERSION" | cut -d. -f2)

if [ "$RERUN_MAJOR" -eq 0 ] && [ "$RERUN_MINOR" -lt 23 ]; then
    DATA_FLAG="--ws-server-port"
    DATA_URL="ws%3A%2F%2Flocalhost%3A${DATA_PORT}"
else
    DATA_FLAG="--port"
    DATA_URL="rerun%2Bhttp%3A%2F%2Flocalhost%3A${DATA_PORT}%2Fproxy"
fi

echo "rerun      : ${RERUN_VERSION}"
echo "Web viewer : http://localhost:${WEB_PORT}?url=${DATA_URL}"
echo "(forward ports ${WEB_PORT} and ${DATA_PORT} to this machine first)"

exec rerun --serve-web \
    --web-viewer-port "${WEB_PORT}" \
    "${DATA_FLAG}" "${DATA_PORT}" \
    --server-memory-limit "${MEMORY_LIMIT}" \
    "$@"
