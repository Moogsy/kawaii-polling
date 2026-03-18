# Analysis

Statistical analysis pipeline for the kawaii perception study described in the paper
*Computational Analysis of Kawaii Perception from Body Postures* (Nguyen & Laohakangvalvit).

## Entry point

**`analyze_df.py`** — loads all rating CSVs (self-ratings + external raters), merges them
into a single DataFrame, and runs the six analysis sections in order:

| Part | Function | Question answered |
|------|----------|-------------------|
| I | `run_part1_context` | Dataset size and structure |
| II | `run_part2_global_stats` | Score distributions across pose categories |
| III | `run_part3_correlations` | Correlations between Kawaii, Warmth, Expressiveness |
| IV | `run_part4_inter_category` | Which pose types score highest, and who are their best/worst/typical representatives? |
| V | `run_part5_intra_category` | Most/least divisive images; does the performer's physique affect perception? |
| VI | `run_part6_perception` | Rater bias, floor/ceiling effects, inter-rater consistency |

Run from the `Analysis/` directory:

```bash
python analyze_df.py
```

The script resolves paths relative to the repository root automatically, expecting:
- `../static/<Category>/blurred_<Model>.png` — anonymized pose images
- `../Ratings/*.csv` — external rater data
- `../SelfRatings/all_ratings_new_format.csv` — performer self-ratings

## `metrics/` package

Each module handles one analytical concern:

| Module | Responsibility |
|--------|---------------|
| `summarize_context.py` | Dataset overview counts and averages |
| `global_distribution.py` | Per-criterion Likert distributions and descriptive stats |
| `corr_per_rating.py` | Pairwise Spearman correlations with LOWESS scatter plots |
| `best_contenders.py` | Best / typical / worst model grid per criterion × category |
| `most_divisive_poses.py` | Most and least divisive pose-model pairs by score std |
| `model_significance.py` | Kruskal-Wallis test for performer effect on perception |
| `perceptions_analysis.py` | Rater bias stats and inter-rater Spearman consistency |

## Setup

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```
