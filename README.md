# Intuitive Physics Eye-Tracking Study

Heuristic and Simulation Strategies in Intuitive Physics Judgments: Evidence from Eye Tracking and Computational Modeling

## Overview

This repository contains the experiment program, analysis code, and behavioral data for a study investigating how people combine heuristic and simulation strategies when predicting physical outcomes. Participants viewed a "Planko" board (ball falling through planks into catchers) and predicted where the ball would land, while their eye movements were recorded with an EyeLink 1000 eye tracker.

### Key Research Questions

- **H1**: Do eye-movement patterns (trajectory overlap, tracking time) vary with heuristic validity (HV) and simulation uncertainty (SU)?
- **H2**: Do gaze allocation patterns across regions of interest (predicted trajectory, critical uncertainty zone, catchers) differ by HV condition?
- **H3**: Under high uncertainty, do participants shift gaze toward critical uncertainty regions, reflecting adaptive strategy switching?
- **Computational Modeling**: Does a coexist model (continuous weighting of heuristic and simulation) better account for behavior than pure-heuristic, pure-simulation, or Sosa-switch alternatives?

## Repository Structure

```
intuitive-physics-eyetracking/
├── README.md
├── requirements.txt              # Python analysis dependencies + R package notes
├── .gitignore
│
├── experiment/
│   ├── main_task/                # Main experiment (PsychoPy + EyeLink)
│   │   ├── config.py             # Experiment parameters
│   │   ├── stimulus_generator.py # Stimulus generation (offline)
│   │   ├── experiment_runtime.py # Experiment runtime (online)
│   │   ├── rating_experiment.py  # [NEW] Independent rating experiment (binary judgment task)
│   │   ├── physics_utils.py      # Pymunk physics engine utilities
│   │   ├── EyeLinkCoreGraphicsPsychoPy.py  # EyeLink integration
│   │   └── requirements.txt      # Experiment / stimulus-generation dependencies
│   └── practice/                 # Practice/training task
│       ├── config.py
│       ├── physics_utils.py
│       ├── generate_practice_stimuli.py
│       └── practice_demo.py
│
├── analysis/                     # All analysis scripts (numbered by execution order)
│   ├── 01_preprocess_behavior.py           # Behavioral data preprocessing
│   ├── 02_behavior_inferential_stats.R     # Behavioral ANOVA & post-hoc tests
│   ├── 03_gaze_pipeline.R                  # Eye-tracking gaze pipeline (R) — requires raw ASC files
│   ├── 04_H2_eye_analysis.R               # H2: ROI & trajectory overlap analysis
│   ├── 05_H3_eye_analysis.R               # H3: CU region & strategy switching
│   ├── 06_eye_metrics_visualization.py     # Publication-quality figures
│   ├── 07_model_fit_compare.py             # Bayesian model fitting & comparison
│   ├── 08_coexist_model_analysis.py        # Detailed coexist model analysis
│   ├── 09_coexist_plot_curves.py           # Coexist model visualization
│   ├── 10_sosa_blended_model.py            # Sosa blended algorithm for RT
│   └── independent_rating_analysis.R       # [NEW] Independent rating experiment analysis (R script)
│
├── data/
│   ├── raw/
│   │   ├── behavior/             # Per-subject behavioral CSVs (N=44)
│   │   └── eye_tracking/         # Raw ASC files; gitignored and may be absent in a fresh clone
│   ├── processed/                # Analysis outputs
│   │   ├── behavior_summary.csv
│   │   ├── behavior_raw_data.csv
│   │   ├── eye_metrics_all_subjects.csv
│   │   ├── eye_metrics_subject_means.csv
│   │   ├── gaze_analysis_all_subjects.RData
│   │   ├── judgment_*.csv        # [NEW] Independent rating analysis outputs
│   │   └── model_comparison_waic_loo.csv
│   ├── rating_data/              # [NEW] Independent rating experiment data (N=20)
│   │   └── sub-{ID}/
│   │       └── ratings.csv       # Binary judgment data
│   ├── stimuli/
│   │   └── stimulus_config.csv   # Full stimulus configuration
│   └── README.md                 # Data documentation
│
└── figures/                      # Generated figures (from analysis scripts)
```

## Experimental Design

- **Task**: Predict which catcher (left or right) a ball will land in after bouncing off planks
- **Design**: 2 (HV: congruent vs. ambiguous) × 5 (SU levels: 1–5) within-subjects
- **Trials**: 350 trials across 7 blocks (50 trials/block)
- **Eye tracking**: EyeLink 1000, 1000 Hz monocular sampling
- **Response**: Mouse click (left/right prediction) + confidence slider (0–100%)

## Supplementary Rating Experiment [NEW]

To examine whether the main SU effect could be attributed to simple changes in heuristic cue availability across SU levels, we conducted an independent binary judgment task with separate participants (N=20).

- **Task**: Judge whether a straight-line trajectory exists in the static board configuration (congruent stimuli only)
- **Response**: Binary judgment (j = "exists", f = "does not exist")
- **Trials**: 50 trials per subject (10 per SU level)
- **Data**: Saved in `data/rating_data/sub-{ID}/ratings.csv`
- **Analysis**: `analysis/independent_rating_analysis.R` — runs repeated-measures ANOVA on "yes" judgments by SU level

**Key finding**: The relationship between SU and straight-trajectory judgments is **non-monotonic**, suggesting the main experiment's SU effect cannot be reduced to a simple "higher SU → less available heuristic" explanation.

### Conditions

| Factor | Levels | Description |
|--------|--------|-------------|
| **Heuristic Validity (HV)** | congruent, ambiguous, conflict | Whether the straight-line heuristic agrees with the true physics outcome |
| **Simulation Uncertainty (SU)** | 1–5 | How sensitive the outcome is to small perturbations (higher = more uncertain) |

## Quick Start

### 1. Install dependencies

**Python** (for analysis scripts 01, 06–10):
```bash
pip install -r requirements.txt
```

**R** (for analysis scripts 02–05 and `independent_rating_analysis.R`):
```r
install.packages(c("dplyr", "tidyr", "purrr", "ggplot2",
                    "ez", "emmeans", "jsonlite", "zoo"))
```

**Experiment / stimulus generation** (optional; needed to run `stimulus_generator.py`, `experiment_runtime.py`, or `rating_experiment.py`):
```bash
pip install -r experiment/main_task/requirements.txt
```

`pylink` must be installed from the SR Research EyeLink SDK rather than from the public Python package index.

### 2. Run the analysis pipeline

Scripts are numbered in recommended execution order:

```bash
cd analysis/

# Step 1: Preprocess behavioral data
python 01_preprocess_behavior.py

# Step 2: Behavioral inferential statistics (R)
Rscript 02_behavior_inferential_stats.R

# Step 3: Eye-tracking gaze pipeline (R) — requires raw ASC files
# Outputs: eye_metrics_all_subjects.csv, eye_metrics_subject_means.csv,
#          gaze_analysis_all_subjects.RData
Rscript 03_gaze_pipeline.R

# Step 4-5: Eye-tracking statistical analyses (R)
Rscript 04_H2_eye_analysis.R
Rscript 05_H3_eye_analysis.R

# Step 6: Generate figures (Python)
python 06_eye_metrics_visualization.py

# Step 7-10: Computational modeling (Python, requires pymc)
python 07_model_fit_compare.py
python 08_coexist_model_analysis.py
python 09_coexist_plot_curves.py
python 10_sosa_blended_model.py

# [NEW] Step 11: Supplementary rating experiment analysis (R)
Rscript independent_rating_analysis.R
```

### 3. Eye-tracking raw data

Raw eye-tracking `.asc` files are **gitignored** because of their size, so `data/raw/eye_tracking/` may be empty in a fresh clone even if a local working copy contains the files.

> **OSF project**: [https://osf.io/cy79h/](https://osf.io/cy79h/)

To reproduce the full eye-tracking analysis pipeline (Steps 3–5):
1. Download the `.asc` files from the OSF project if they are absent locally
2. Place them in `data/raw/eye_tracking/` with the naming convention `X{nn}.asc` (e.g., `X01.asc`)
3. Run `Rscript analysis/03_gaze_pipeline.R` to regenerate trial-level eye metrics, subject-level summaries, and the combined `.RData` workspace used for inspection

The current project layout contains behavioral data for 44 participants and eye-tracking files for 43 participants (`X38.asc` is absent while `data/raw/behavior/38.csv` is present), so eye-tracking analyses use one fewer subject than behavioral analyses by design.

**Steps 1–2 and 6–10 can run without reprocessing the raw eye-tracking files** by using the processed data already included in this repository.

## Computational Models

Four hierarchical Bayesian models are compared:

| Model | Description | Key Parameters |
|-------|-------------|----------------|
| **Coexist** | Continuous heuristic–simulation mixture | wH (heuristic weight), p_sim, p_heur |
| **Pure Heuristic** | HV-driven logistic | alpha, beta_HV |
| **Pure Simulation** | SU-driven logistic | alpha, beta_SU |
| **Sosa Switch** | Trajectory-based binary gate | gate (0/1), p_sim, p_heur |

Models are fit with PyMC (NUTS sampler) and compared via WAIC, LOO-CV, AIC, and BIC.

## Key Eye Metrics

| Metric | Description |
|--------|-------------|
| **I_over_U (Trajectory Overlap)** | Intersection-over-union of gaze path and true trajectory |
| **Track_ms** | Longest continuous gaze run along the predicted trajectory |
| **Prop_PT / Duration_PT** | Gaze proportion and duration in the predicted trajectory ROI |
| **Prop_CU / Duration_CU** | Gaze proportion and duration in the critical uncertainty ROI |
| **Fixations_PT / Fixations_CU** | Fixation counts in PT and CU ROIs |
| **PT_entries / CU_visits** | Compatibility aliases retained for downstream scripts; currently computed with the same run-count logic as `Fixations_PT` / `Fixations_CU` |

## Software Requirements

- **Python** >= 3.10
- **R** >= 4.2
- **Main Experiment / Stimulus Generation**: PsychoPy 2023.2.3, pymunk 6.6.0, tqdm, pylink (SR Research SDK)
- **Supplementary Rating Experiment**: PsychoPy 2023.2.3, pandas, numpy
- **Analysis**: Install Python packages from `requirements.txt`; install the R packages listed in the Quick Start section

### [NEW] Independent Rating Task

To run the supplementary rating experiment:
```bash
cd experiment/main_task/
python rating_experiment.py
```

To analyze the rating data:
```r
cd analysis/
Rscript independent_rating_analysis.R
```
Outputs are saved to `data/processed/` as `judgment_*.csv` summary files.

## License

This project is shared for academic purposes. Please cite the associated publication if you use this code or data.

## Contact

For questions or to request raw eye-tracking data, please contact the authors.
