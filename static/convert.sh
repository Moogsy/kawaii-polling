#!/bin/zsh
#
# Recursively convert every *.png under the current directory to WebP
# using ffmpeg.  Existing .webp files are left untouched.
# ---------------------------------------------------------------

# Find all *.png files (case–insensitive, safe for spaces/new-lines)
find . -type f -iname '*.png' -print0 |
while IFS= read -r -d '' png; do
    webp="${png%.*}.webp"          # same path/name, new extension

    # Skip if target already exists and is newer or same age
    if [ -e "$webp" ] && [ "$webp" -nt "$png" ]; then
        printf '✔  %s -> %s (up-to-date)\n' "$png" "$webp"
        continue
    fi

    printf '▶  %s -> %s\n' "$png" "$webp"
    # -y overwrites automatically if the file exists but is older
    if ffmpeg -loglevel error -y -i "$png" "$webp"; then
        printf '✓  Done\n'
    else
        printf '✗  Failed converting %s\n' "$png" >&2
    fi
done

