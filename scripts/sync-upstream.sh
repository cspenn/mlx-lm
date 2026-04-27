#!/usr/bin/env bash
# Sync the timer-s1 feature branch on top of the latest upstream main.
# Run from the repo root. Working tree must be clean before running.
set -euo pipefail

UPSTREAM_REMOTE="${UPSTREAM_REMOTE:-origin}"
FORK_REMOTE="${FORK_REMOTE:-fork}"
MAIN_BRANCH="main"
FEATURE_BRANCH="timer-s1"

# Abort if working tree is dirty
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: working tree is dirty. Commit or stash all changes before syncing."
  exit 1
fi

current_branch=$(git branch --show-current)

# Safety tag on the feature branch tip before touching anything
git tag -f timer-s1-prev "${FEATURE_BRANCH}" 2>/dev/null || true
echo "Rollback anchor: timer-s1-prev"

# 1. Mirror main to upstream
echo "--- Fetching upstream ---"
git fetch "${UPSTREAM_REMOTE}" --tags

echo "--- Fast-forwarding ${MAIN_BRANCH} ---"
git checkout "${MAIN_BRANCH}"
git merge --ff-only "${UPSTREAM_REMOTE}/${MAIN_BRANCH}"
git push "${FORK_REMOTE}" "${MAIN_BRANCH}"

# 2. Rebase feature branch onto updated main
echo "--- Rebasing ${FEATURE_BRANCH} onto ${MAIN_BRANCH} ---"
git checkout "${FEATURE_BRANCH}"
git rebase "${MAIN_BRANCH}"
# rerere auto-resolves the MODEL_REMAPPING conflict after the first time

# 3. Verify
echo "--- Running Timer-S1 tests ---"
python -m pytest tests/test_timer_s1.py -q

echo "--- Checking MODEL_REMAPPING registration ---"
python -c "from mlx_lm.utils import MODEL_REMAPPING; assert MODEL_REMAPPING.get('Timer-S1') == 'timer_s1', 'Timer-S1 not in MODEL_REMAPPING'; print('  OK: Timer-S1 -> timer_s1')"

# 4. Publish
echo "--- Pushing ${FEATURE_BRANCH} to fork ---"
git push "${FORK_REMOTE}" "${FEATURE_BRANCH}" --force-with-lease

# Restore original branch
git checkout "${current_branch}" 2>/dev/null || true

echo ""
echo "Done. To roll back: git checkout ${FEATURE_BRANCH} && git reset --hard timer-s1-prev"
