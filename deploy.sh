#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_ROOT="${INSTALL_ROOT:-$HOME/.local}"
BIN_PATH="$INSTALL_ROOT/bin/llm-mcp"

mkdir -p "$INSTALL_ROOT/bin"

if [ -L "$BIN_PATH" ]; then
    unlink "$BIN_PATH"
fi

cargo install --path "$ROOT_DIR" --root "$INSTALL_ROOT" --force

printf 'Installed llm-mcp to %s\n' "$BIN_PATH"
