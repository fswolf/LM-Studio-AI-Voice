#!/usr/bin/env bash
#
# Commit and push everything in this repo.
#
#   ./push.sh                  -> "Update 2026-09-19 05:14"
#   ./push.sh fixed the VAD    -> that, as the commit message
#   ./push.sh -y tidy up       -> skip the confirmation
#
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

YES=0
if [ "${1:-}" = "-y" ] || [ "${1:-}" = "--yes" ]; then
    YES=1
    shift
fi

MESSAGE="${*:-Update $(date '+%Y-%m-%d %H:%M')}"

if ! git rev-parse --git-dir >/dev/null 2>&1; then
    echo "Not a git repo: $PWD" >&2
    exit 1
fi

# --- personal files that shouldn't be public -------------------------------
# .gitignore only covers untracked files. Anything committed before a rule
# was added keeps getting updated, which is how a chat log ends up on
# GitHub without anyone deciding to put it there.
LEAKY=$(git ls-files | grep -E \
    'history/conversation\.json|reminders/reminders\.json|Claude outputs/' \
    || true)

if [ -n "$LEAKY" ]; then
    echo "These are tracked but look personal:"
    echo "$LEAKY" | sed 's/^/  /'
    echo
    echo "Stop tracking them (keeps your local copy):"
    echo "$LEAKY" | sed 's|^|  git rm --cached "|; s|$|"|'
    echo
fi

# --- what's actually changing ----------------------------------------------
if [ -z "$(git status --porcelain)" ]; then
    echo "Nothing to commit."

    if [ -n "$(git log '@{upstream}..HEAD' --oneline 2>/dev/null || true)" ]; then
        echo "But there are unpushed commits:"
        git log '@{upstream}..HEAD' --oneline | sed 's/^/  /'
        printf 'Push them? [y/N] '

        if [ "$YES" = 1 ]; then
            echo y
        else
            read -r reply
            [ "$reply" = "y" ] || [ "$reply" = "Y" ] || exit 0
        fi

        git push
    fi

    exit 0
fi

echo "Changes to commit:"
git status --short | sed 's/^/  /'
echo
echo "Message: $MESSAGE"
echo "Branch:  $(git rev-parse --abbrev-ref HEAD)"
echo

if [ "$YES" != 1 ]; then
    printf 'Commit and push? [y/N] '
    read -r reply

    case "$reply" in
        y|Y) ;;
        *) echo "Cancelled."; exit 0 ;;
    esac
fi

git add -A
git commit -m "$MESSAGE"

# First push on a new branch needs an upstream; after that plain push.
if git rev-parse '@{upstream}' >/dev/null 2>&1; then
    git push
else
    git push -u origin "$(git rev-parse --abbrev-ref HEAD)"
fi

echo "Pushed."
