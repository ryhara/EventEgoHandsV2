#!/usr/bin/env bash
# Initialize the hot3d submodule and apply the required local patch
# (num_pose_coeffs argument and device fix in mano_layer.py).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

git submodule update --init hot3d

PATCH="$REPO_ROOT/scripts/hot3d_mano_layer.patch"

if git -C hot3d apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "hot3d patch is already applied."
elif git -C hot3d apply --check "$PATCH" 2>/dev/null; then
    git -C hot3d apply "$PATCH"
    echo "hot3d patch applied."
else
    echo "ERROR: could not apply $PATCH to hot3d (unexpected submodule state)." >&2
    echo "Try: git -C hot3d checkout -- . && bash scripts/setup_hot3d.sh" >&2
    exit 1
fi
