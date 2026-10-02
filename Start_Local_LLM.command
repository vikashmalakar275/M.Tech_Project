#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_MODELS="$PWD/.models"
if [ -x .tools/ollama/ollama ]; then
    exec .tools/ollama/ollama serve
elif command -v ollama >/dev/null 2>&1; then
    exec ollama serve
else
    printf 'Ollama is not installed. See the optional local LLM section in README.md.\n'
    exit 1
fi
