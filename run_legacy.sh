#!/bin/bash
# Run koopa-luigi with legacy environment (TF 2.13)

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ ! -d "$SCRIPT_DIR/venv_legacy" ]; then
    echo "Error: Legacy environment not found."
    echo "Run: ./setup_envs.sh /path/to/koopa"
    exit 1
fi

source "$SCRIPT_DIR/venv_legacy/bin/activate"
echo "Using LEGACY environment (TF 2.13)"
koopa-luigi "$@"