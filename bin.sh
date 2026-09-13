#!/usr/bin/env bash
# Synchronize this TeX repository. Use --check for a local read-only preview.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
repo_root=$(pwd -P)
[[ "$(git rev-parse --show-toplevel)" == "$repo_root" ]] || {
    echo 'Error: script must live at the repository root.' >&2; exit 1;
}
mode=sync
message='Update TeX sources and compiled documents'
while [[ $# -gt 0 ]]; do
    case "$1" in
        --check) mode=check; shift ;;
        -m|--message)
            [[ $# -ge 2 && -n "$2" ]] || { echo 'A commit message is required.' >&2; exit 1; }
            message=$2; shift 2 ;;
        *) echo 'Usage: bash bin.sh [--check] [-m "commit message"]' >&2; exit 1 ;;
    esac
done
[[ "$(git symbolic-ref --quiet --short HEAD)" == main ]] || {
    echo 'Error: switch to main before synchronizing.' >&2; exit 1;
}
for state in MERGE_HEAD CHERRY_PICK_HEAD REVERT_HEAD rebase-merge rebase-apply; do
    [[ ! -e "$(git rev-parse --git-path "$state")" ]] || {
        echo "Error: unfinished Git operation ($state)." >&2; exit 1;
    }
done
[[ -z "$(git ls-files -u)" ]] || { echo 'Error: unresolved conflicts.' >&2; exit 1; }
git remote get-url origin
git status --short --branch
if [[ "$mode" == sync ]]; then
    git fetch origin
    read -r ahead behind < <(git rev-list --left-right --count HEAD...origin/main)
    if (( ahead > 0 && behind > 0 )); then
        echo 'Error: main and origin/main have diverged; review before merging.' >&2; exit 1
    fi
    if (( behind > 0 )); then
        [[ -z "$(git status --porcelain)" ]] || {
            echo 'Error: remote is ahead and local files have changes; reconcile them before syncing.' >&2; exit 1;
        }
        git merge --ff-only origin/main
    fi
fi
# Only ignored, untracked files with a matching .tex source may be deleted.
# Keep .bbl files: they can be required for publishing without BibTeX.
while IFS= read -r -d '' artifact; do
    case "$artifact" in
        *.synctex.gz) stem=${artifact%.synctex.gz} ;;
        *.run.xml) stem=${artifact%.run.xml} ;;
        *.fdb_latexmk) stem=${artifact%.fdb_latexmk} ;;
        *.aux|*.bcf|*.out|*.log|*.fls|*.blg|*.xdv|*.toc|*.lof|*.lot|*.nav|*.snm)
            stem=${artifact%.*} ;;
        *) continue ;;
    esac
    [[ -f "$stem.tex" && -f "$artifact" && ! -L "$artifact" ]] || continue
    printf 'Clean: %s\n' "$artifact"
    if [[ "$mode" == sync ]]; then rm -- "$artifact"; fi
done < <(git ls-files --others --ignored --exclude-standard -z)
if [[ "$mode" == check ]]; then
    git diff --stat HEAD
    echo 'Preview only; remote state was not fetched and no files were changed.'
    exit 0
fi
git diff --check
git diff --cached --check
git add -A -- .
if ! git diff --cached --quiet; then
    git diff --cached --stat
    git commit -m "$message"
fi
git push -u origin main
git status --short --branch
read -r ahead behind < <(git rev-list --left-right --count HEAD...origin/main)
[[ "$ahead" == 0 && "$behind" == 0 && -z "$(git status --porcelain)" ]] || {
    echo 'Error: synchronization finished with remaining differences.' >&2; exit 1;
}
echo 'Synchronized; working tree is clean.'
