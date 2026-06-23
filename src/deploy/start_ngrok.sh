#!/usr/bin/env bash
set -euo pipefail

# Configuration
PORT="${API_PORT:-8000}"
HOST="${API_HOST:-0.0.0.0}"
NGROK_DOMAIN="${NGROK_DOMAIN:-}"
NGROK_AUTHTOKEN="${NGROK_AUTHTOKEN:-}"
API_LOG="${API_LOG:-api.log}"
NGROK_LOG="${NGROK_LOG:-ngrok.log}"
NGROK_CONFIG="${NGROK_CONFIG:-ngrok.yml}"
VENV_DIR="${VENV_DIR:-venv}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cleanup() {
    echo "Shutting down..."
    [ -f "$PROJECT_DIR/.api_pid" ] && kill "$(cat "$PROJECT_DIR/.api_pid")" 2>/dev/null || true
    [ -f "$PROJECT_DIR/.ngrok_pid" ] && kill "$(cat "$PROJECT_DIR/.ngrok_pid")" 2>/dev/null || true
    rm -f "$PROJECT_DIR/.api_pid" "$PROJECT_DIR/.ngrok_pid"
    exit 0
}
trap cleanup SIGINT SIGTERM

cd "$PROJECT_DIR"

if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
fi

echo "Starting FastAPI on $HOST:$PORT..."
nohup uvicorn src.api.main:app --host "$HOST" --port "$PORT" > "$API_LOG" 2>&1 &
API_PID=$!
echo "$API_PID" > .api_pid
echo "FastAPI running (PID: $API_PID)"

sleep 3
if ! kill -0 "$API_PID" 2>/dev/null; then
    echo "ERROR: FastAPI failed to start. Check $API_LOG"
    cat "$API_LOG"
    exit 1
fi

# Create ngrok config if auth token is provided
if [ -n "$NGROK_AUTHTOKEN" ]; then
    cat > "$NGROK_CONFIG" << EOF
version: "2"
authtoken: $NGROK_AUTHTOKEN
tunnels:
  tb-recovery-api:
    proto: http
    addr: $PORT
    ${NGROK_DOMAIN:+domain: $NGROK_DOMAIN}
EOF
fi

echo "Starting ngrok tunnel..."
NGROK_CMD="ngrok http $PORT --log=stdout"
if [ -f "$NGROK_CONFIG" ]; then
    NGROK_CMD="ngrok start --config $NGROK_CONFIG tb-recovery-api"
fi

nohup $NGROK_CMD > "$NGROK_LOG" 2>&1 &
NGROK_PID=$!
echo "$NGROK_PID" > .ngrok_pid
echo "ngrok tunnel starting (PID: $NGROK_PID)"

sleep 4
NGROK_URL=$(curl -s http://127.0.0.1:4040/api/tunnels 2>/dev/null | python3 -c "
import sys, json
data = json.load(sys.stdin)
for t in data.get('tunnels', []):
    print(t.get('public_url', ''))
" 2>/dev/null | head -1)

if [ -n "$NGROK_URL" ]; then
    echo ""
    echo "========================================"
    echo "  API is LIVE at: $NGROK_URL"
    echo "  Health check: $NGROK_URL/health"
    echo "  Docs:         $NGROK_URL/docs"
    echo "========================================"
    echo "$NGROK_URL" > .ngrok_url
else
    echo "ngrok URL not yet available. Check $NGROK_LOG"
fi

echo ""
echo "API log:  $API_LOG"
echo "ngrok log: $NGROK_LOG"
echo "Press Ctrl+C to stop."

wait
