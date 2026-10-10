#!/usr/bin/env bash
# ============================================================
#  LFM Memory Chat — launcher (Linux / Arch)
#  Creates .venv on first run (Arch's system pip is
#  externally-managed per PEP 668), installs deps, starts app.
# ============================================================
set -e
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
VENV=".venv"

if ! command -v "$PY" >/dev/null 2>&1; then
    echo "[ERROR] python3 not found. Install it with:"
    echo "        sudo pacman -S --needed python"
    exit 1
fi

if [ ! -x "$VENV/bin/python" ]; then
    echo "[SETUP] Creating virtual environment in $VENV ..."
    "$PY" -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

if ! python -c "import gradio, llama_cpp" >/dev/null 2>&1; then
    echo "[SETUP] Installing dependencies (first run only, may take a few minutes)..."
    python -m pip install --upgrade pip
    # Prebuilt CPU wheel for llama-cpp-python (no compiler needed).
    # Falls back to a source install if the wheel index is unreachable.
    python -m pip install llama-cpp-python --only-binary=llama-cpp-python \
        --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
        || python -m pip install llama-cpp-python
    python -m pip install -r requirements.txt
fi

if [ ! -f "models/LFM-2.5-1.2b-Instruct-roleplay-tuned-v2.Q4_K_M.gguf" ] \
   && [ -z "$LFM_MODEL_PATH" ]; then
    echo "[WARN] GGUF file not found in models/ — you can still preview"
    echo "       the UI later with:  python app.py --mock"
fi

echo "[START] Launching LFM Memory Chat → http://127.0.0.1:7860"
exec python app.py "$@"
