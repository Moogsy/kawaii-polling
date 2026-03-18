# LocalRun

Runs the image-rating session locally as a desktop GUI.
Raters see each pose image one at a time and score it on three Likert scales
(Kawaii, Warmth, Expressiveness) using on-screen buttons or keyboard shortcuts.
Results are saved to a CSV that feeds directly into the `Analysis/` pipeline.

## Usage

```bash
python main.py
```

You will be prompted for a rater name, then the rating window opens.
When all images have been scored the window closes and the results are written to:

```
Ratings/<rater_name>_ratings.csv
```

If the file cannot be written, the CSV is printed to stdout as a fallback.

## Controls

| Input | Action |
|-------|--------|
| Click a Likert button | Record score for that scale |
| Keys `1`–`5` | Record score for the next unanswered scale in order |
| `Enter` / "Next" button | Advance to the next image (only works when all three scales are answered) |

## Files

| File | Purpose |
|------|---------|
| `main.py` | Entry point — discovers images, interleaves them across categories with `sample_approx_2d`, prompts for rater name, launches the rater, and saves the CSV |
| `image_rater.py` | `ImageRater` class — builds the matplotlib GUI (image display + Likert buttons), handles input events, and collects records into a DataFrame |
| `compress_images.py` | One-off utility — compresses all PNGs in `Pictures/` to under 1 MB using adaptive palette quantisation and writes output to `Pictures_Compressed/` |

## Image layout

Images must be placed under `LocalRun/Pictures/` following this structure:

```
Pictures/
  <Category>/
    blurred_<Model>.png
    ...
```

The presentation order is interleaved across categories (not purely random) so that
a rater never rates all images from one category in a row.
