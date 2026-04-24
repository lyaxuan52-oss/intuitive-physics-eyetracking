# ============================================================
# 05_H3_eye_analysis.R
# H3: CU region gaze metrics & strategy switching for high-difficulty trials
#
# Focuses on SU levels 3-5 (high difficulty) to test:
#   1. Trajectory overlap: 2(HV) x 3(SU=3,4,5) ANOVA
#   2. Tracking time: 2(HV) x 3(SU=3,4,5) ANOVA
#   3. CU gaze proportion: 2(HV) x 3(SU=3,4,5) ANOVA
#   4. CU visits: 2(HV) x 3(SU=3,4,5) ANOVA
#
# Dependencies: dplyr, ggplot2, ez, emmeans, tidyr
#
# Usage:
#   cd analysis/
#   Rscript 05_H3_eye_analysis.R
# ============================================================

library(dplyr)
library(ggplot2)
library(ez)
library(emmeans)
library(tidyr)

# ---------- Path configuration ----------
script_dir <- tryCatch(
  dirname(normalizePath(sys.frame(1)$ofile)),
  error = function(e) getwd()
)
project_root <- dirname(script_dir)
data_dir    <- file.path(project_root, "data", "processed")
figures_dir <- file.path(project_root, "figures")
dir.create(figures_dir, showWarnings = FALSE, recursive = TRUE)

# ---------- 1. Load data ----------

eye_file <- file.path(data_dir, "eye_metrics_subject_means.csv")
if (!file.exists(eye_file)) {
  stop("Cannot find eye metrics file: ", eye_file)
}

eye_data <- read.csv(eye_file, stringsAsFactors = FALSE)

# Filter: congruent/ambiguous, SU levels 3-5 (high difficulty)
eye <- eye_data |>
  mutate(
    subject_id = factor(subject_id),
    HV_primary = factor(trimws(HV_primary)),
    SU_level = as.integer(SU_level)
  ) |>
  filter(HV_primary %in% c("congruent", "ambiguous"),
         SU_level %in% c(3, 4, 5)) |>
  mutate(SU_level = factor(SU_level))

cat("High-difficulty subset: ", nrow(eye), " rows\n")
cat("HV levels:", levels(eye$HV_primary), "\n")
cat("SU levels:", levels(eye$SU_level), "\n")
cat("N subjects:", length(unique(eye$subject_id)), "\n")

# ---------- 2. Trajectory overlap: 2(HV) x 3(SU) ----------

cat("\n========== Trajectory Overlap: 2(HV) x 3(SU=3,4,5) ==========\n")

anova_traj <- ezANOVA(
  data = eye,
  dv = I_over_U,
  wid = subject_id,
  within = .(HV_primary, SU_level),
  type = 3,
  detailed = TRUE
)
print(anova_traj)

traj_summary <- eye |>
  group_by(HV_primary, SU_level) |>
  summarise(M = mean(I_over_U, na.rm = TRUE),
            SD = sd(I_over_U, na.rm = TRUE),
            SE = SD / sqrt(n()), .groups = "drop")

p1 <- ggplot(traj_summary, aes(x = SU_level, y = M,
                                color = HV_primary, group = HV_primary)) +
  geom_line(size = 1.2) + geom_point(size = 4) +
  geom_errorbar(aes(ymin = M - SE, ymax = M + SE), width = 0.1) +
  labs(title = "H3: Trajectory Overlap (High Difficulty)",
       x = "SU Level", y = "I/U", color = "HV") +
  theme_bw(base_size = 13)

ggsave(file.path(figures_dir, "H3_trajectory_overlap.png"),
       p1, width = 8, height = 6, dpi = 300)

# ---------- 3. Tracking time: 2(HV) x 3(SU) ----------

cat("\n========== Tracking Time: 2(HV) x 3(SU=3,4,5) ==========\n")

anova_track <- ezANOVA(
  data = eye,
  dv = Track_ms,
  wid = subject_id,
  within = .(HV_primary, SU_level),
  type = 3,
  detailed = TRUE
)
print(anova_track)

track_summary <- eye |>
  group_by(HV_primary, SU_level) |>
  summarise(M = mean(Track_ms, na.rm = TRUE),
            SD = sd(Track_ms, na.rm = TRUE),
            SE = SD / sqrt(n()), .groups = "drop")

p2 <- ggplot(track_summary, aes(x = SU_level, y = M,
                                 color = HV_primary, group = HV_primary)) +
  geom_line(size = 1.2) + geom_point(size = 4) +
  geom_errorbar(aes(ymin = M - SE, ymax = M + SE), width = 0.1) +
  labs(title = "H3: Tracking Time (High Difficulty)",
       x = "SU Level", y = "Tracking Time (ms)", color = "HV") +
  theme_bw(base_size = 13)

ggsave(file.path(figures_dir, "H3_tracking_time.png"),
       p2, width = 8, height = 6, dpi = 300)

# ---------- 4. CU gaze proportion: 2(HV) x 3(SU) ----------

cat("\n========== CU Gaze Proportion: 2(HV) x 3(SU=3,4,5) ==========\n")

anova_cu_prop <- ezANOVA(
  data = eye,
  dv = Prop_CU,
  wid = subject_id,
  within = .(HV_primary, SU_level),
  type = 3,
  detailed = TRUE
)
print(anova_cu_prop)

cu_prop_summary <- eye |>
  group_by(HV_primary, SU_level) |>
  summarise(M = mean(Prop_CU, na.rm = TRUE),
            SD = sd(Prop_CU, na.rm = TRUE),
            SE = SD / sqrt(n()), .groups = "drop")

p3 <- ggplot(cu_prop_summary, aes(x = SU_level, y = M,
                                   color = HV_primary, group = HV_primary)) +
  geom_line(size = 1.2) + geom_point(size = 4) +
  geom_errorbar(aes(ymin = M - SE, ymax = M + SE), width = 0.1) +
  labs(title = "H3: CU Gaze Proportion (High Difficulty)",
       x = "SU Level", y = "CU Gaze Proportion", color = "HV") +
  theme_bw(base_size = 13)

ggsave(file.path(figures_dir, "H3_CU_gaze_proportion.png"),
       p3, width = 8, height = 6, dpi = 300)

# ---------- 5. CU visits: 2(HV) x 3(SU) ----------

cat("\n========== CU Visits: 2(HV) x 3(SU=3,4,5) ==========\n")

if ("Fixations_CU" %in% colnames(eye)) {
  anova_cu_visits <- ezANOVA(
    data = eye,
    dv = Fixations_CU,
    wid = subject_id,
    within = .(HV_primary, SU_level),
    type = 3,
    detailed = TRUE
  )
  print(anova_cu_visits)

  cu_visits_summary <- eye |>
    group_by(HV_primary, SU_level) |>
    summarise(M = mean(Fixations_CU, na.rm = TRUE),
              SD = sd(Fixations_CU, na.rm = TRUE),
              SE = SD / sqrt(n()), .groups = "drop")

  p4 <- ggplot(cu_visits_summary, aes(x = SU_level, y = M,
                                       color = HV_primary, group = HV_primary)) +
    geom_line(size = 1.2) + geom_point(size = 4) +
    geom_errorbar(aes(ymin = M - SE, ymax = M + SE), width = 0.1) +
    labs(title = "H3: CU Visits (High Difficulty)",
         x = "SU Level", y = "CU Fixation Count", color = "HV") +
    theme_bw(base_size = 13)

  ggsave(file.path(figures_dir, "H3_CU_visits.png"),
         p4, width = 8, height = 6, dpi = 300)
} else {
  cat("[INFO] Fixations_CU column not found; skipping CU visits analysis.\n")
}

cat("\n========== H3 Analysis Complete ==========\n")
cat("Figures saved to:", figures_dir, "\n")
