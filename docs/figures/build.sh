#!/bin/sh
set -e
cd "$(dirname "$0")"
out=$PWD/../src/assets/theory
tut=$PWD/../src/assets/tutorials
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$out" "$tut"
for f in $(ls *.tex | grep -v '^\(fig\|preamble\)\.tex$'); do
  n=${f%.tex}
  TEXINPUTS=.: lualatex -interaction=batchmode -halt-on-error -output-directory="$tmp" -jobname="$n" "\\def\\fig{$n}\\input{fig}" >/dev/null
  pdftocairo -svg "$tmp/$n.pdf" "$out/$n.svg"
done
for f in tutorials/*.tex; do
  n=$(basename "$f" .tex)
  TEXINPUTS=.: lualatex -interaction=batchmode -halt-on-error -output-directory="$tmp" -jobname="$n" "\\def\\fig{tutorials/$n}\\input{fig}" >/dev/null
  pdftocairo -svg "$tmp/$n.pdf" "$tut/$n.svg"
done
