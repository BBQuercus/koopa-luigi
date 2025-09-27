#!/bin/bash
# Run koopa-luigi with modern environment (TF 2.17+)

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ ! -d "$SCRIPT_DIR/venv_modern" ]; then
    echo "Error: Modern environment not found."
    echo "Run: ./setup_envs.sh /path/to/koopa"
    exit 1
fi

source "$SCRIPT_DIR/venv_modern/bin/activate"
echo "Using MODERN environment (TF 2.17+)"
koopa-luigi "$@"