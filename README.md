# Kawaii Polling

Repository for the project **Computational Analysis of Kawaii Perception from Body Postures**.

This codebase supports the full workflow used in the study:
- data collection,
- perceptual annotation,
- statistical analysis,
- and computational modeling from pose geometry.

---

## 1) Project objective

The core research question is:

> Can measurable geometric features of **body posture alone** explain perceived **kawaii**?

Unlike most prior kawaii studies centered on face or styling cues, this project focuses on posture-only signals (with blurred faces) and tests interpretable geometric predictors.

---

## 2) Paper summary (mapped to this repo)

The included documents (`narratif.pdf`, `details.pdf`) describe a two-phase experimental protocol and computational pipeline.

### Conceptual framing
Kawaii is treated as a two-layer construct:
1. **Perceptual layer** (gentleness, vulnerability, approachability),
2. **Socio-affective layer** (warmth, empathy, prosocial orientation).

### Experimental design
#### Phase 1 — Self-evaluation
- 12 participants,
- 7 expressive pose categories,
- initial evaluation over 6 dimensions,
- category representatives selected for phase 2.

#### Phase 2 — External evaluation
- 15 independent raters,
- 84 images (12 models × 7 categories),
- 3 perceptual criteria: **Kawaii**, **Warmth**, **Expressiveness**,
- 3,780 valid external judgments reported in the paper summary.

### Computational pipeline
- Landmark extraction with **MediaPipe Pose** (33 keypoints),
- normalized geometric descriptors (translation/scale/orientation invariance),
- interpretable feature families (angles, balance, compactness, head/limb signals),
- mixed/statistical modeling and feature-importance analysis.

### Main findings (paper narrative)
- Kawaii is strongly coupled with **Warmth** (and moderately with Expressiveness).
- Postural configuration is a major driver of perception.
- **Vertical center of gravity (COG_Y)** emerges as the strongest geometric predictor of kawaii.

---

## 3) What is in this repository

## Data assets
### Images
`static/` contains 7 pose categories with blurred `.webp` files:
- `Cool-Clever`
- `Dependant-Needy`
- `Escapist-Dreamy`
- `Joyful-Smiling`
- `Normal-Warmup`
- `Playful-Clumsy`
- `Shy-Gentle`

In the current snapshot, each category has 12 images, for **84 total images**.

### Ratings
- `Ratings/` — collected participant ratings (web/local runs),
- `SelfRatings/` — self-ratings and converted datasets.

Common schema used by analysis code:

| Column | Meaning |
|---|---|
| `Category` | Pose category |
| `Model` | Performer/model ID |
| `Rating` | `Kawaii`, `Warmth`, or `Expressiveness` |
| `RaterID` | Evaluator identifier |
| `Score` | Likert score (`1..5`) |

Repo snapshot (all committed CSVs combined):
- 4,032 rows,
- 27 unique raters,
- 84 unique `(Category, Model)` pairs,
- score range 1–5.

> Note: this repository total includes self-rating and additional files; paper-reported totals for Phase 2 external ratings are lower (3,780).

---

## 4) Annotation interfaces

## Web app (`app.py`)
Routes:
- `/` start/resume session,
- `/rate` image + three Likert questions,
- `/submit` persists 3 records per image,
- `/thank_you` completion page.

Important details:
- image order is generated with category-adjacency constraints via `utils/samplers.py`,
- category cardinality is validated before sampling,
- output filenames are sanitized and deduplicated (`web_<name>_ratings*.csv`).

Run locally:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open: `http://127.0.0.1:5000`

## Local UI (`LocalRun/`)
Provides an offline Matplotlib-based rater interface for local collection/testing.

---

## 5) Analysis pipeline (`Analysis/`)

Entrypoint:

```bash
python -m Analysis
```

`Analysis/analyze_df.py` runs six sections:
1. context summary,
2. global score distributions,
3. pairwise correlations,
4. inter-category representatives,
5. intra-category divisiveness + model effects,
6. rater bias and consistency diagnostics.

Most plotting/stat modules live in `Analysis/metrics/`.

---

## 6) Computational modeling (`AI/`)

Main scripts:
- `AI/export_agg_scores.py` — aggregate targets from ratings,
- `AI/geometric_normalized.py` and `AI/__main__.py` — pose-landmark geometric features + regressors,
- `AI/clip_ridge.py` — CLIP embedding + PCA + Ridge baseline.

These scripts evaluate whether interpretable pose geometry can predict perceived kawaii and related perceptual scores.

---

## 7) Deployment

With Docker:

```bash
docker compose up --build
```

- Flask served by Gunicorn (internal 8000),
- Nginx reverse proxy for external ports.

---

## 8) Caveats

- The codebase mixes French and English names/comments.
- `docker-compose.yml` includes a `static/Pictures` bind path; adjust if your deployment uses category folders directly under `static/`.
- For strict paper replication, freeze dependencies and version output figures/tables.

---

## 9) File map

- `app.py` — Flask rating app
- `templates/` — web pages
- `static/` — image assets
- `Ratings/`, `SelfRatings/` — rating CSV data
- `utils/samplers.py` — constrained sequence sampler
- `LocalRun/` — local rater UI
- `Analysis/` — statistical analysis
- `AI/` — modeling experiments
- `narratif.pdf`, `details.pdf` — study documents
