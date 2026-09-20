#!/usr/bin/env bash
#
# Commit and push everything in this repo.
#
#   ./push.sh                  -> "Update 2026-09-19 05:14"
#   ./push.sh fixed the VAD    -> that, as the commit message
#   ./push.sh -y tidy up       -> skip the confirmation
#
# Also offers to protect the files that ship with the repo but are tuned
# to this machine, so a later rebase can't overwrite them.
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

# --- personal files ---------------------------------------------------------
# .gitignore only covers untracked files. Anything committed before a rule
# was added keeps getting updated, which is how a chat log ends up on
# GitHub without anyone deciding to put it there.
#
# Two different problems, two different fixes.
#
# PRIVATE files should never be public at all - your conversations, your
# reminders, your transcript. Those want dropping from the repo.
#
# PERSONAL files should ship with the repo so a fresh clone works, but
# your local copy is tuned to your machine and shouldn't be pushed - or,
# worse, overwritten by a rebase pulling the committed version back over
# it. That is skip-worktree: the file stays in the repo at its committed
# contents while git stops watching your copy. It has eaten local edits
# to config.json and agent.json more than once.
PRIVATE_PATTERN='history/conversation\.json|history/transcript\.jsonl|reminders/reminders\.json|Claude outputs/'
PERSONAL_PATTERN='^config\.json$|^agent/agent\.json$|^agent/memory\.json$'

SKIPPED=$(git ls-files -v | grep '^S ' | cut -c3- || true)

not_skipped() {
    if [ -z "$SKIPPED" ]; then
        cat
    else
        grep -vxF "$SKIPPED" || true
    fi
}

PRIVATE=$(git ls-files | grep -E "$PRIVATE_PATTERN" | not_skipped || true)
PERSONAL=$(git ls-files | grep -E "$PERSONAL_PATTERN" | not_skipped || true)

if [ -n "$PRIVATE" ]; then
    echo "These are tracked and would be published:"
    echo "$PRIVATE" | sed 's/^/  /'
    echo
    echo "Drop them from the repo, keeping your local copies:"
    echo "$PRIVATE" | sed 's|^|  git rm --cached "|; s|$|"|'
    echo
fi

if [ -n "$PERSONAL" ]; then
    COUNT=$(echo "$PERSONAL" | wc -l)

    echo "$COUNT file(s) ship with the repo but are tuned to this machine:"
    echo "$PERSONAL" | sed 's/^/  /'
    echo
    echo "Marking them skip-worktree keeps the committed version public and"
    echo "stops git touching yours - no more rebases overwriting your config."
    echo "The trade-off: your local changes to them stop being pushed."
    printf 'Mark them now? [y/N] '

    if [ "$YES" = 1 ]; then
        echo y
        reply=y
    else
        read -r reply
    fi

    case "$reply" in
        y|Y)
            echo "$PERSONAL" | while IFS= read -r file; do
                [ -n "$file" ] || continue

                if git update-index --skip-worktree "$file" 2>/dev/null; then
                    echo "  protected $file"
                else
                    echo "  couldn't protect $file" >&2
                fi
            done

            # Anything just marked is no longer staged for this commit.
            SKIPPED=$(git ls-files -v | grep '^S ' | cut -c3- || true)
            echo
            ;;
        *)
            echo "  left alone - commit them as-is, or mark them later with:"
            echo "$PERSONAL" | sed 's|^|    git update-index --skip-worktree "|; s|$|"|'
            echo
            ;;
    esac
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

        # A protected file changing upstream is the one case where the
        # pull won't just work. skip-worktree stops git overwriting your
        # copy, which is the whole point - but git's answer is to refuse
        # the merge outright with "your local changes would be
        # overwritten", which looks like a broken repo rather than a
        # working safety catch. Say so here, with the way out.
        COLLIDES=""

        if [ -n "$SKIPPED" ]; then
            COLLIDES=$(git diff --name-only 'HEAD..@{upstream}' 2>/dev/null \
                | grep -xF "$SKIPPED" || true)
        fi

        if [ -n "$COLLIDES" ]; then
            echo "Heads up - the remote also changed file(s) you've protected:"
            echo "$COLLIDES" | sed 's/^/  /'
            echo
            echo "git will refuse the pull rather than overwrite them. To take"
            echo "the incoming version, keeping a copy of yours:"
            echo "$COLLIDES" | while IFS= read -r f; do
                [ -n "$f" ] || continue
                echo "  cp \"$f\" \"$f.mine\""
                echo "  git update-index --no-skip-worktree \"$f\""
                echo "  git checkout -- \"$f\""
            done
            echo "  git pull --rebase"
            echo "$COLLIDES" | while IFS= read -r f; do
                [ -n "$f" ] || continue
                echo "  git update-index --skip-worktree \"$f\"   # re-protect"
            done
            echo
            echo "Then merge anything you want back in from the .mine copies."
            echo
            exit 1
        fi

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
