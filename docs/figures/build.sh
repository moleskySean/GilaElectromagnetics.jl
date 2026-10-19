#!/bin/sh
set -e
cd "$(dirname "$0")"
out=$PWD/../src/assets/theory
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$out"
for f in $(ls *.tex | grep -v '^\(fig\|preamble\)\.tex$'); do
  n=${f%.tex}
  TEXINPUTS=.: lualatex -interaction=batchmode -halt-on-error -output-directory="$tmp" -jobname="$n" "\\def\\fig{$n}\\input{fig}" >/dev/null
  pdftocairo -svg "$tmp/$n.pdf" "$out/$n.svg"
done
