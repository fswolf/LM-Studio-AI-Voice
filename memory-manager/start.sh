#!/usr/bin/env bash
# Start the memory manager from anywhere - the script knows where it lives.
cd "$(dirname "$(realpath "$0")")" || exit 1

exec python3 manager.py
