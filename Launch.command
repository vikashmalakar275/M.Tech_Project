#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
    printf 'Run the installation steps in README.md before opening OrbitWatch.\n'
    exit 1
fi
exec .venv/bin/python -m streamlit run app.py --server.address 127.0.0.1
