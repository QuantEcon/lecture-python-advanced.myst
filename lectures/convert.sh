#!/bin/bash
FILES=./*.md
for f in $FILES
do
  echo "Processing $f file..."
  jupytext $f --to ipynb
done