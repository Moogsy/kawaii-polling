#!/usr/bin/env bash

# Enable nullglob so the loop is skipped if no .csv files exist in a directory
shopt -s nullglob

# Recursively iterate over all CSVs
while IFS= read -r -d '' file; do
  # Get the line count
  count=$(wc -l < "$file")
  # If it isn't 253, echo the filename
  if [[ "$count" -ne 253 ]]; then
    echo "$file"
  fi
done < <(find . -type f -name '*.csv' -print0)

