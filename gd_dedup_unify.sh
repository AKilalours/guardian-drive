#!/usr/bin/env bash
set -euo pipefail

APPLY=0
if [[ "${1:-}" == "--apply" ]]; then
  APPLY=1
fi

ROOT="$(pwd)"
DATA_RAW="$ROOT/data/raw"
DATASETS="$ROOT/datasets"
STAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE="$DATASETS/_archive_dedup_$STAMP"

PTB_CANON="$DATA_RAW/ptbdb"
WESAD_CANON="$DATA_RAW/WESAD"

need_dir() {
  local d="$1"
  [[ -d "$d" ]] || { echo "Missing required directory: $d" >&2; exit 1; }
}

need_dir "$DATA_RAW"
need_dir "$DATASETS"
need_dir "$PTB_CANON/1.0.0"
need_dir "$WESAD_CANON/WESAD"

echo "Repo root: $ROOT"
echo "Canonical PTBDB : $PTB_CANON"
echo "Canonical WESAD : $WESAD_CANON"
echo

move_to_archive() {
  local p="$1"
  [[ -e "$p" || -L "$p" ]] || return 0

  mkdir -p "$ARCHIVE"
  local base
  base="$(basename "$p")"

  if [[ $APPLY -eq 1 ]]; then
    echo "ARCHIVE $p -> $ARCHIVE/$base"
    mv "$p" "$ARCHIVE/$base"
  else
    echo "WOULD ARCHIVE $p -> $ARCHIVE/$base"
  fi
}

make_symlink() {
  local target="$1"
  local link="$2"

  if [[ -L "$link" ]]; then
    echo "Already symlink: $link -> $(readlink "$link")"
    return 0
  fi

  if [[ -e "$link" ]]; then
    echo "Refusing to overwrite existing non-symlink: $link" >&2
    exit 1
  fi

  if [[ $APPLY -eq 1 ]]; then
    ln -s "$target" "$link"
    echo "LINK $link -> $target"
  else
    echo "WOULD LINK $link -> $target"
  fi
}

echo "=== Candidate duplicates ==="
[[ -e "$DATASETS/ptbdb" ]] && echo "$DATASETS/ptbdb"
[[ -e "$DATASETS/WESAD" ]] && echo "$DATASETS/WESAD"
[[ -e "$DATASETS/physionet.org" ]] && echo "$DATASETS/physionet.org"
[[ -e "$DATASETS/WESAD.zip" ]] && echo "$DATASETS/WESAD.zip"
[[ -e "$DATASETS/sleep-edfx.zip" ]] && echo "$DATASETS/sleep-edfx.zip"
[[ -e "$DATASETS/isles2022" ]] && echo "$DATASETS/isles2022"
echo

# archive duplicates / irrelevant heavy extras
move_to_archive "$DATASETS/ptbdb"
move_to_archive "$DATASETS/WESAD"
move_to_archive "$DATASETS/physionet.org"
move_to_archive "$DATASETS/WESAD.zip"
move_to_archive "$DATASETS/sleep-edfx.zip"
move_to_archive "$DATASETS/isles2022"

# recreate lightweight compatibility links
make_symlink "../data/raw/ptbdb" "$DATASETS/ptbdb"
make_symlink "../data/raw/WESAD" "$DATASETS/WESAD"

echo
echo "=== Final check ==="
if [[ $APPLY -eq 1 ]]; then
  ls -ld "$DATASETS/ptbdb" "$DATASETS/WESAD"
  echo
  echo "PTB records sample:"
  find "$PTB_CANON/1.0.0" -maxdepth 2 -name '*.hea' | head -5
  echo
  echo "WESAD sample:"
  find "$WESAD_CANON/WESAD" -maxdepth 2 -name 'S2.pkl' | head -5
else
  echo "Dry run complete. Re-run with --apply to make changes."
fi
