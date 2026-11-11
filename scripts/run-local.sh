#!/bin/zsh
set -eu
cd "$(dirname "$0")/.."
for app_port in 8000 8501; do
    if lsof -nP -iTCP:"$app_port" -sTCP:LISTEN >/dev/null 2>&1; then
        print -u2 "Port $app_port is already in use. Stop the existing application before starting a fresh copy."
        exit 1
    fi
done
mkdir -p .cache
export OLLAMA_MODELS="$PWD/artifacts/ollama"
export OLLAMA_HOST=http://127.0.0.1:11434
export OLLAMA_NO_CLOUD=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ollama_bin="${RETAILASSIST_OLLAMA_BIN:-$PWD/.cache/ollama/ollama}"
if ! curl -sf http://127.0.0.1:11434/api/tags >/dev/null; then
    "$ollama_bin" serve > .cache/ollama.log 2>&1 &
    ollama_pid=$!
fi
.venv/bin/python -m uvicorn retailassist.api:app --host 127.0.0.1 --port 8000 > .cache/api.log 2>&1 &
api_pid=$!
.venv/bin/python -m streamlit run app.py > .cache/ui.log 2>&1 &
ui_pid=$!
cleanup() {
    kill "$api_pid" "$ui_pid" 2>/dev/null || true
    if [ -n "${ollama_pid:-}" ]; then kill "$ollama_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM
print "RetailAssist: http://127.0.0.1:8501  |  API docs: http://127.0.0.1:8000/docs"
wait
