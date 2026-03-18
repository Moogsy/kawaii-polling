# AI

Computational pipeline for the paper's §3.3 and §4.4 — connecting body geometry to
perceived kawaii scores through pose feature extraction, predictive modelling, and
mixed-effects statistical analysis.

## Pipeline order

```
export_agg_scores.py          # 1. prepare agg_phase2.csv
    ↓
__main__.py                   # 2. quick RF baseline (raw keypoints)
clip_ridge.py                 # 2b. pixel-based CLIP baseline
    ↓
geometric_normalized.py       # 3. normalized geometry + RF → feature importance
    ↓
pray.py  /  stats_mmodel.py   # 4. mixed-effects model → Table 11 / 12
```

## Files

### `export_agg_scores.py` — data preparation
Reads all external rating CSVs from `../Ratings/`, strips self-ratings
(`RaterID == Model`), and aggregates mean kawaii scores per (Category, Model) pair.
Saves the result to `agg_phase2.csv`, which is the input for the modelling scripts.
This corresponds to the Phase 2 data cleaning described in §3.1.

### `__main__.py` — raw-keypoint Random Forest baseline
Extracts MediaPipe Pose landmarks from each image without normalization, computes a
basic feature vector (elbow/knee angles, arm-to-torso ratios, center of gravity), and
trains a Random Forest to predict `kawaii_mean`. This was an early exploratory run
before the proper normalization step was introduced.

### `clip_ridge.py` — pixel-based CLIP + PCA + Ridge baseline
Embeds each image with CLIP ViT-B/32, reduces to 20 PCA components, and fits a Ridge
regression with GroupKFold cross-validation (grouped by performer).
This is the pixel-level baseline reported in §4.4.2, which yielded R² ≈ −0.1 to −0.2,
demonstrating that raw image embeddings carry insufficient signal for this task and
motivating the geometric approach.

### `geometric_normalized.py` — normalized geometry + Random Forest (§4.4 / Table 13)
The refined geometric pipeline.  Landmarks are normalized to the torso reference frame
(translation, scale, and orientation invariance — §3.3) before feature extraction.
Person-level and category-level biases are removed via residualization before training
a shallow, regularized Random Forest.  The resulting feature importances correspond
directly to **Table 13** of the paper, with `COG_Y` (vertical center of gravity)
ranking first.

### `pray.py` — mixed-effects model, full pipeline (§4.4 / Equation 1 / Tables 11–12)
Implements the paper's primary statistical model:

```
Score ~ C(Category) + COG_Y + COG_X + Elbow_Min_Angle + Knee_Min_Angle
        + R_Arm_Ratio + L_Arm_Ratio + Head_Tilt + Head_Rot
        + (1 | RaterID) + (1 | Model) + (1 | ImageID)
```

Pose landmarks are extracted and normalized per image, geometric features are
z-scored, and a `statsmodels` MixedLM is fitted for each rating dimension
(Kawaii, Warmth, Expressiveness).  Nakagawa & Schielzeth marginal / conditional R²
are computed.  Fixed-effect tables and model summaries are saved to `mixedlm_outputs/`.
Results correspond to **Tables 11 and 12** and the finding that `COG_Y` is the only
significant geometric predictor (β = 0.274, p = 0.007).

### `stats_mmodel.py` — mixed-effects model, CLI version
A command-line variant of `pray.py` that accepts pre-computed ratings and feature CSVs
as arguments.  Useful for re-running the LMM with different feature sets or data
subsets without modifying source code.

```bash
python stats_mmodel.py \
  --ratings concat_ratings.csv \
  --features features_pose.csv \
  --out-prefix results/mixedlm \
  --standardize
```

## Setup

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
# CLIP also requires:
pip install git+https://github.com/openai/CLIP.git  # for clip_ridge.py only
```
