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
#
# Files marked skip-worktree are excluded: they stay in the repo at their
# committed contents while local edits are ignored, which is the right
# answer for a config file you want published but not synced.
SKIPPED=$(git ls-files -v | grep '^S ' | cut -c3- || true)

LEAKY=$(git ls-files | grep -E \
    'history/conversation\.json|reminders/reminders\.json|agent/memory\.json|Claude outputs/' \
    || true)

if [ -n "$SKIPPED" ]; then
    LEAKY=$(echo "$LEAKY" | grep -vxF "$SKIPPED" || true)
fi

if [ -n "$LEAKY" ]; then
    echo "These are tracked and will be committed as-is:"
    echo "$LEAKY" | sed 's/^/  /'
    echo
    echo "Keep it in the repo but stop syncing your local copy:"
    echo "$LEAKY" | sed 's|^|  git update-index --skip-worktree "|; s|$|"|'
    echo
    echo "Or drop it from the repo entirely (keeps your local file):"
    echo "$LEAKY" | sed 's|^|  git rm --cached "|; s|$|"|'
    echo
fi

# --- is the remote ahead of us? --------------------------------------------
# Editing a file in GitHub's web UI creates a commit you don't have, and
# the push then fails after you've already committed - which reads like
# the script broke when it didn't.
if git rev-parse '@{upstream}' >/dev/null 2>&1; then
    git fetch --quiet 2>/dev/null || true

    BEHIND=$(git rev-list --count 'HEAD..@{upstream}' 2>/dev/null || echo 0)

    if [ "${BEHIND:-0}" -gt 0 ]; then
        echo "The remote has $BEHIND commit(s) you don't have:"
        git log 'HEAD..@{upstream}' --oneline | sed 's/^/  /'
        echo
        echo "Pull them first:"
        echo "  git pull --rebase"
        echo
        echo "If that conflicts on a file you edited locally, your version"
        echo "is the one called --theirs during a rebase:"
        echo "  git checkout --theirs <file> && git add <file> && git rebase --continue"
        exit 1
    fi
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
    PUSH_OK=0
    git push || PUSH_OK=$?
else
    PUSH_OK=0
    git push -u origin "$(git rev-parse --abbrev-ref HEAD)" || PUSH_OK=$?
fi

if [ "$PUSH_OK" != 0 ]; then
    echo
    echo "Your commit is safe locally - only the push failed."
    echo "Usually the remote moved. Try:"
    echo "  git pull --rebase && git push"
    exit 1
fi

echo "Pushed."
