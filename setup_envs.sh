#!/bin/bash
# Simple setup script for modern and legacy environments

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
KOOPA_PATH="${1}"

echo "Setting up koopa-luigi environments..."
echo "Using koopa from: $KOOPA_PATH"
echo ""

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed."
    echo "Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Create modern environment
echo "Creating MODERN environment (TF 2.17+)..."
if [ -d "$SCRIPT_DIR/venv_modern" ]; then
    rm -rf "$SCRIPT_DIR/venv_modern"
fi
uv venv "$SCRIPT_DIR/venv_modern" --python 3.10
uv pip install -r "$SCRIPT_DIR/requirements_modern.txt" --python "$SCRIPT_DIR/venv_modern/bin/python"
if [ -d "$KOOPA_PATH" ]; then
    "$SCRIPT_DIR/venv_modern/bin/pip" install -e "$KOOPA_PATH" -q
fi
"$SCRIPT_DIR/venv_modern/bin/pip" install -e "$SCRIPT_DIR" -q
echo "✓ Modern environment ready"
echo ""

# Create legacy environment  
echo "Creating LEGACY environment (TF 2.13)..."
if [ -d "$SCRIPT_DIR/venv_legacy" ]; then
    rm -rf "$SCRIPT_DIR/venv_legacy"
fi
uv venv "$SCRIPT_DIR/venv_legacy" --python 3.10
uv pip install -r "$SCRIPT_DIR/requirements_legacy.txt" --python "$SCRIPT_DIR/venv_legacy/bin/python"
if [ -d "$KOOPA_PATH" ]; then
    "$SCRIPT_DIR/venv_legacy/bin/pip" install -e "$KOOPA_PATH" -q
fi
"$SCRIPT_DIR/venv_legacy/bin/pip" install -e "$SCRIPT_DIR" -q
echo "✓ Legacy environment ready"
echo ""

echo "Setup complete! Use:"
echo "  ./run_modern.sh --config koopa.cfg    # For recent models"
echo "  ./run_legacy.sh --config koopa.cfg    # For original models"