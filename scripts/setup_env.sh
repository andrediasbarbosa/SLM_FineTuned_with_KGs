#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

if [ -d ".venv" ]; then
    echo "[setup_env] .venv already exists, skipping creation."
else
    echo "[setup_env] Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

echo "[setup_env] Upgrading pip..."
pip install --upgrade pip --quiet

echo "[setup_env] Installing CPU torch..."
pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet

echo "[setup_env] Installing HuggingFace stack..."
pip install transformers datasets peft trl accelerate sentencepiece --quiet

echo "[setup_env] Generating lockfile at requirements.lock..."
pip freeze > requirements.lock

echo "[setup_env] Done. Activate with: source .venv/bin/activate"
