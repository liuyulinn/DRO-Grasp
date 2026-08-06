#!/usr/bin/env bash
# One-shot environment bootstrap: download Isaac Gym Preview 4 and sync the uv env.
#
# Isaac Gym is not on PyPI, so pyproject.toml declares it as an optional extra
# pointing at ./third_party/isaacgym/python. This script fetches the tarball,
# unpacks it there, then runs `uv sync --extra isaacgym`.
#
# Re-runnable: skips the download if third_party/isaacgym/python already exists.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

ISAACGYM_DIR="$REPO_ROOT/third_party/isaacgym"
ISAACGYM_GDRIVE_ID="16WFp0n6hzqsVXS0XLU0Tb_54wrKEF8ky"
ISAACGYM_TARBALL="IsaacGym_Preview_4_Package.tar.gz"

if [[ ! -f "$ISAACGYM_DIR/python/setup.py" ]]; then
    echo "==> Downloading Isaac Gym Preview 4"
    command -v gdown >/dev/null 2>&1 || pip install --user gdown
    mkdir -p "$REPO_ROOT/third_party"
    gdown "https://drive.google.com/uc?id=${ISAACGYM_GDRIVE_ID}" -O "$ISAACGYM_TARBALL"
    tar -xf "$ISAACGYM_TARBALL" -C "$REPO_ROOT/third_party"
    rm "$ISAACGYM_TARBALL"
else
    echo "==> Isaac Gym already present at $ISAACGYM_DIR, skipping download"
fi

echo "==> Running uv sync --extra isaacgym"
uv sync --extra isaacgym
