#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "=== AI Labeler MCP — Setup ==="

if ! command -v python3 &>/dev/null; then
  echo "ERROR: python3 not found."
  exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python $PYTHON_VERSION detected."

if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi

source .venv/bin/activate
echo "Virtual environment activated."

echo "Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo ""
echo "=== Installation complete! ==="
echo ""
echo "Credentials needed (set once in claude_desktop_config.json → env):"
echo "  OPENAI_API_KEY   — for OpenAI models (gpt-4o, gpt-4.1, ...)"
echo "  GEMINI_API_KEY   — for Gemini models (gemini-2.0-flash, ...)"
echo ""
echo "Python binary: $(pwd)/.venv/bin/python"
echo "Server script: $(pwd)/server.py"
