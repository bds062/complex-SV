#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[metric_projection] The few-shot training entry point now uses the objectness + type classifier head." >&2
echo "[metric_projection] Forwarding to scripts/classification_head.sh." >&2
exec "$SCRIPT_DIR/classification_head.sh" "$@"
