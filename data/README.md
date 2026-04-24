# Data Directory

## Structure

```
data/
├── raw/
│   ├── behavior/          # Per-subject behavioral CSV files (N=44)
│   └── eye_tracking/      # Raw eye-tracking data (EDF/ASC, not in repo due to size)
├── processed/             # Outputs from analysis pipeline
│   ├── behavior_summary.csv              # Subject x HV x SU behavioral summary
│   ├── behavior_raw_data.csv             # Combined trial-level behavioral data
│   ├── eye_metrics_subject_means.csv     # Subject-level eye metric means
│   ├── demo_subjects.csv                 # Demographic information
│   ├── desc_by_HV.csv                    # Descriptive stats by HV condition
│   ├── desc_by_HV_SU.csv                # Descriptive stats by HV x SU
│   ├── model_comparison_waic_loo.csv     # Computational model comparison
│   └── ...                               # Additional analysis outputs
└── stimuli/
    └── stimulus_config.csv               # Full stimulus configuration (350 boards)
```

## Raw Behavioral Data

Each file in `raw/behavior/` is named `{SubjectID}.csv` and contains trial-level data with columns:
- **SubjectID, Gender, Age, BallExperience**: Demographics
- **TrialID, BlockID, BoardID**: Trial identifiers
- **SU_level**: Simulation Uncertainty level (1-5)
- **HV_primary**: Heuristic type (congruent / ambiguous / conflict)
- **Choice, Correct, RT**: Behavioral responses
- **Confidence, Confidence_RT**: Confidence ratings

## Raw Eye-Tracking Data

Eye-tracking files (`.asc`, 28 subjects, ~4.5 GB total) are hosted on **OSF** due to GitHub file size limits:

> **OSF project**: [https://osf.io/XXXXX/](https://osf.io/XXXXX/) *(replace with actual link)*

Download all `.asc` files and place them in `raw/eye_tracking/` with naming convention `X{nn}.asc` (e.g., `X01.asc`).

## Stimulus Configuration

`stimuli/stimulus_config.csv` contains the complete board configuration for all 350 trials:
- **BoardID, SU_raw, SU_level**: Board identification and uncertainty
- **HV_primary**: Heuristic validity type
- **PhysOutcome**: True physics outcome (left/right)
- **Ball_start_x, Ball_start_y**: Ball starting position
- **True_trajectory**: Ball trajectory (JSON array of {x, y, t})
- **Planks**: Plank positions (JSON array of {x, y, angle, ...})
