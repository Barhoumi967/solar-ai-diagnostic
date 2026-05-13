#!/usr/bin/env bash
# Solar AI Diagnostic — startup script (Linux / macOS)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"
BACKEND="$SCRIPT_DIR/backend"
MODELS_DIR="$BACKEND/models/saved"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== Solar AI Diagnostic ===${NC}"

# ── Python check ──────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
  echo -e "${RED}ERROR: Python 3 not found. Install it from https://python.org${NC}"
  exit 1
fi

# ── Virtual environment ───────────────────────────────────────────────
if [ ! -d "$VENV" ]; then
  echo -e "${YELLOW}Creating virtual environment...${NC}"
  python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"

# ── Dependencies ──────────────────────────────────────────────────────
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -r "$BACKEND/requirements.txt" -q --disable-pip-version-check

# ── Train models if missing ───────────────────────────────────────────
if [ ! -f "$MODELS_DIR/best_model.pkl" ]; then
  echo -e "${YELLOW}Training AI models (first run — takes ~30 seconds)...${NC}"
  python3 "$BACKEND/models/train_all.py"
fi

# ── Find a free port ──────────────────────────────────────────────────
find_free_port() {
  python3 - <<EOF
import socket, sys
port = $1
while True:
    try:
        s = socket.socket()
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('', port))
        s.close()
        print(port)
        break
    except OSError:
        port += 1
EOF
}

PORT=$(find_free_port 5001)

if [ "$PORT" != "5001" ]; then
  echo -e "${YELLOW}Port 5001 in use — using port $PORT instead${NC}"
fi

# ── Start Flask ───────────────────────────────────────────────────────
echo -e "${GREEN}Starting server on http://localhost:$PORT${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop.${NC}"
echo ""

cd "$BACKEND"
PORT=$PORT python3 app.py
