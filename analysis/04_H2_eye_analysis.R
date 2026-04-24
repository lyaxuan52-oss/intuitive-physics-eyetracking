# ============================================================
# 04_H2_eye_analysis.R
# Eye-tracking H2 analysis based on aggregated subject-level data
#   1. Trajectory overlap: 2(HV) x 5(SU) repeated-measures ANOVA
#   2. ROI analysis: 2(HV) x 3(ROI) repeated-measures ANOVA
#      - Gaze proportion, duration, fixation count
#   3. Simple effects analysis for significant interactions
#   4. Descriptive statistics summary table
#
# Dependencies: dplyr, ggplot2, ez, emmeans, tidyr, grid, gridExtra, gtable
#   install.packages(c("dplyr", "ggplot2", "ez", "emmeans", "tidyr",
#                      "gridExtra", "gtable"))
#
# Usage:
#   cd analysis/
#   Rscript 04_H2_eye_analysis.R
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

# ---------- 1. Load aggregated data ----------

eye_file <- file.path(data_dir, "eye_metrics_subject_means.csv")

cat("Eye metrics file:", eye_file, "\n")
cat("File exists:", file.exists(eye_file), "\n")

if (!file.exists(eye_file)) {
  stop("Cannot find eye metrics file: ", eye_file)
}

eye_data <- read.csv(eye_file, stringsAsFactors = FALSE)

cat("Data dimensions:", dim(eye_data), "\n")
cat("Columns:", colnames(eye_data), "\n")

# Keep only congruent and ambiguous conditions
eye <- eye_data |>
  mutate(
    subject_id = factor(subject_id),
    HV_primary = factor(trimws(HV_primary)),
    SU_level = factor(SU_level)
  ) |>
  filter(!is.na(subject_id), !is.na(HV_primary), !is.na(SU_level)) |>
  filter(HV_primary %in% c("congruent", "ambiguous"))

cat("Filtered data dimensions:", dim(eye), "\n")
cat("HV levels:", levels(eye$HV_primary), "\n")
cat("SU levels:", levels(eye$SU_level), "\n")
cat("N subjects:", length(unique(eye$subject_id)), "\n")

# ---------- 2. Trajectory overlap: 2(HV) x 5(SU) ANOVA ----------

cat("\n========== Trajectory Overlap: 2(HV) x 5(SU) ANOVA ==========\n")

anova_trajectory <- ezANOVA(
  data = eye,
  dv = I_over_U,
  wid = subject_id,
  within = .(HV_primary, SU_level),
  type = 3,
  detailed = TRUE
)
print(anova_trajectory)

traj_summary <- eye |>
  group_by(HV_primary, SU_level) |>
  summarise(
    n = n(),
    mean_iou = mean(I_over_U, na.rm = TRUE),
    sd_iou = sd(I_over_U, na.rm = TRUE),
    se_iou = sd(I_over_U, na.rm = TRUE) / sqrt(n),
    .groups = "drop"
  )

cat("\nTrajectory overlap descriptive statistics:\n")
print(traj_summary)

p_trajectory <- ggplot(traj_summary, aes(x = SU_level, y = mean_iou, 
                                         color = HV_primary, group = HV_primary)) +
  geom_line(size = 1.2, position = position_dodge(width = 0.1)) +
  geom_point(size = 4, position = position_dodge(width = 0.1)) +
  geom_errorbar(aes(ymin = mean_iou - se_iou, ymax = mean_iou + se_iou),
                width = 0.1, size = 1, position = position_dodge(width = 0.1)) +
  labs(title = "Trajectory Overlap: 2(HV) x 5(SU)",
       x = "SU Level", 
       y = "Trajectory Overlap (I/U)",
       color = "HV Condition") +
  theme_bw() +
  theme(
    plot.title = element_text(size = 14, hjust = 0.5),
    axis.text = element_text(size = 12),
    axis.title = element_text(size = 12),
    legend.text = element_text(size = 11)
  )

print(p_trajectory)
ggsave(file.path(figures_dir, "Trajectory_overlap_HVxSU.png"), 
       p_trajectory, width = 10, height = 7, dpi = 300)

# ---------- 3. ROI Analysis: 2(HV) x 3(ROI) ANOVA ----------

cat("\n========== ROI Analysis: 2(HV) x 3(ROI) ANOVA ==========\n")

# 3.1 Gaze proportion
cat("\n--- 3.1 Gaze Proportion ---\n")

roi_prop_data <- eye |>
  group_by(subject_id, HV_primary) |>
  summarise(
    Prop_PT = mean(Prop_PT, na.rm = TRUE),
    Prop_CU = mean(Prop_CU, na.rm = TRUE), 
    Prop_catcher = mean(Prop_catcher, na.rm = TRUE),
    .groups = "drop"
  ) |>
  pivot_longer(
    cols = c(Prop_PT, Prop_CU, Prop_catcher),
    names_to = "ROI",
    values_to = "Proportion"
  ) |>
  mutate(
    ROI = factor(ROI, levels = c("Prop_PT", "Prop_CU", "Prop_catcher"),
                 labels = c("PT", "CU", "Catcher"))
  ) |>
  filter(!is.na(Proportion))

anova_roi_prop <- ezANOVA(
  data = roi_prop_data,
  dv = Proportion,
  wid = subject_id,
  within = .(HV_primary, ROI),
  type = 3,
  detailed = TRUE
)
print(anova_roi_prop)

roi_prop_summary <- roi_prop_data |>
  group_by(HV_primary, ROI) |>
  summarise(
    n = n(),
    mean_prop = mean(Proportion, na.rm = TRUE),
    sd_prop = sd(Proportion, na.rm = TRUE),
    se_prop = sd(Proportion, na.rm = TRUE) / sqrt(n),
    .groups = "drop"
  )

cat("\nGaze proportion descriptive statistics:\n")
print(roi_prop_summary)

p_roi_prop <- ggplot(roi_prop_summary, aes(x = ROI, y = mean_prop, 
                                           color = HV_primary, group = HV_primary)) +
  geom_line(size = 1.2, position = position_dodge(width = 0.1)) +
  geom_point(size = 4, position = position_dodge(width = 0.1)) +
  geom_errorbar(aes(ymin = mean_prop - se_prop, ymax = mean_prop + se_prop),
                width = 0.1, size = 1, position = position_dodge(width = 0.1)) +
  labs(title = "ROI Gaze Proportion: 2(HV) x 3(ROI)",
       x = "ROI Region", 
       y = "Gaze Proportion",
       color = "HV Condition") +
  theme_bw() +
  theme(
    plot.title = element_text(size = 14, hjust = 0.5),
    axis.text = element_text(size = 12),
    axis.title = element_text(size = 12),
    legend.text = element_text(size = 11)
  )

print(p_roi_prop)
ggsave(file.path(figures_dir, "ROI_proportions_HVxROI.png"), 
       p_roi_prop, width = 10, height = 7, dpi = 300)

# 3.2 Fixation duration
cat("\n--- 3.2 Fixation Duration ---\n")

roi_duration_data <- eye |>
  group_by(subject_id, HV_primary) |>
  summarise(
    Duration_PT = mean(Duration_PT, na.rm = TRUE),
    Duration_CU = mean(Duration_CU, na.rm = TRUE), 
    Duration_catcher = mean(Duration_catcher, na.rm = TRUE),
    .groups = "drop"
  ) |>
  pivot_longer(
    cols = c(Duration_PT, Duration_CU, Duration_catcher),
    names_to = "ROI",
    values_to = "Duration"
  ) |>
  mutate(
    ROI = factor(ROI, levels = c("Duration_PT", "Duration_CU", "Duration_catcher"),
                 labels = c("PT", "CU", "Catcher"))
  ) |>
  filter(!is.na(Duration))

anova_roi_duration <- ezANOVA(
  data = roi_duration_data,
  dv = Duration,
  wid = subject_id,
  within = .(HV_primary, ROI),
  type = 3,
  detailed = TRUE
)
print(anova_roi_duration)

roi_duration_summary <- roi_duration_data |>
  group_by(HV_primary, ROI) |>
  summarise(
    n = n(),
    mean_duration = mean(Duration, na.rm = TRUE),
    sd_duration = sd(Duration, na.rm = TRUE),
    se_duration = sd(Duration, na.rm = TRUE) / sqrt(n),
    .groups = "drop"
  )

cat("\nFixation duration descriptive statistics:\n")
print(roi_duration_summary)

p_roi_duration <- ggplot(roi_duration_summary, aes(x = ROI, y = mean_duration, 
                                                   color = HV_primary, group = HV_primary)) +
  geom_line(size = 1.2, position = position_dodge(width = 0.1)) +
  geom_point(size = 4, position = position_dodge(width = 0.1)) +
  geom_errorbar(aes(ymin = mean_duration - se_duration, ymax = mean_duration + se_duration),
                width = 0.1, size = 1, position = position_dodge(width = 0.1)) +
  labs(title = "ROI Fixation Duration: 2(HV) x 3(ROI)",
       x = "ROI Region", 
       y = "Duration (ms)",
       color = "HV Condition") +
  theme_bw() +
  theme(
    plot.title = element_text(size = 14, hjust = 0.5),
    axis.text = element_text(size = 12),
    axis.title = element_text(size = 12),
    legend.text = element_text(size = 11)
  )

print(p_roi_duration)
ggsave(file.path(figures_dir, "ROI_duration_HVxROI.png"), 
       p_roi_duration, width = 10, height = 7, dpi = 300)

# 3.3 Fixation count
cat("\n--- 3.3 Fixation Count ---\n")

roi_fixations_data <- eye |>
  group_by(subject_id, HV_primary) |>
  summarise(
    Fixations_PT = mean(Fixations_PT, na.rm = TRUE),
    Fixations_CU = mean(Fixations_CU, na.rm = TRUE), 
    Fixations_catcher = mean(Fixations_catcher, na.rm = TRUE),
    .groups = "drop"
  ) |>
  pivot_longer(
    cols = c(Fixations_PT, Fixations_CU, Fixations_catcher),
    names_to = "ROI",
    values_to = "Fixations"
  ) |>
  mutate(
    ROI = factor(ROI, levels = c("Fixations_PT", "Fixations_CU", "Fixations_catcher"),
                 labels = c("PT", "CU", "Catcher"))
  ) |>
  filter(!is.na(Fixations))

anova_roi_fixations <- ezANOVA(
  data = roi_fixations_data,
  dv = Fixations,
  wid = subject_id,
  within = .(HV_primary, ROI),
  type = 3,
  detailed = TRUE
)
print(anova_roi_fixations)

roi_fixations_summary <- roi_fixations_data |>
  group_by(HV_primary, ROI) |>
  summarise(
    n = n(),
    mean_fixations = mean(Fixations, na.rm = TRUE),
    sd_fixations = sd(Fixations, na.rm = TRUE),
    se_fixations = sd(Fixations, na.rm = TRUE) / sqrt(n),
    .groups = "drop"
  )

cat("\nFixation count descriptive statistics:\n")
print(roi_fixations_summary)

p_roi_fixations <- ggplot(roi_fixations_summary, aes(x = ROI, y = mean_fixations, 
                                                     color = HV_primary, group = HV_primary)) +
  geom_line(size = 1.2, position = position_dodge(width = 0.1)) +
  geom_point(size = 4, position = position_dodge(width = 0.1)) +
  geom_errorbar(aes(ymin = mean_fixations - se_fixations, ymax = mean_fixations + se_fixations),
                width = 0.1, size = 1, position = position_dodge(width = 0.1)) +
  labs(title = "ROI Fixation Count: 2(HV) x 3(ROI)",
       x = "ROI Region", 
       y = "Fixation Count",
       color = "HV Condition") +
  theme_bw() +
  theme(
    plot.title = element_text(size = 14, hjust = 0.5),
    axis.text = element_text(size = 12),
    axis.title = element_text(size = 12),
    legend.text = element_text(size = 11)
  )

print(p_roi_fixations)
ggsave(file.path(figures_dir, "ROI_fixations_HVxROI.png"), 
       p_roi_fixations, width = 10, height = 7, dpi = 300)

# ---------- 4. Simple effects analysis ----------

cat("\n========== Simple Effects Analysis ==========\n")

# 4.1 Trajectory overlap simple effects (HV x SU interaction)
cat("\n--- 4.1 Trajectory overlap simple effects ---\n")

cat("\n=== SU effect under Ambiguous ===\n")
eye_ambiguous <- eye |> filter(HV_primary == "ambiguous")
simple_traj_ambiguous <- ezANOVA(
  data = eye_ambiguous,
  dv = I_over_U,
  wid = subject_id,
  within = SU_level,
  type = 3,
  detailed = FALSE
)
print(simple_traj_ambiguous)

cat("\n=== SU effect under Congruent ===\n")
eye_congruent <- eye |> filter(HV_primary == "congruent")
simple_traj_congruent <- ezANOVA(
  data = eye_congruent,
  dv = I_over_U,
  wid = subject_id,
  within = SU_level,
  type = 3,
  detailed = FALSE
)
print(simple_traj_congruent)

cat("\n=== Pairwise comparisons under Ambiguous ===\n")
model_traj_amb <- aov(I_over_U ~ SU_level + Error(subject_id/SU_level), 
                      data = eye_ambiguous)
emm_traj_amb <- emmeans(model_traj_amb, ~ SU_level)
pairs_traj_amb <- pairs(emm_traj_amb, adjust = "none")
print(pairs_traj_amb)

cat("\n=== Pairwise comparisons under Congruent ===\n")
model_traj_con <- aov(I_over_U ~ SU_level + Error(subject_id/SU_level), 
                      data = eye_congruent)
emm_traj_con <- emmeans(model_traj_con, ~ SU_level)
pairs_traj_con <- pairs(emm_traj_con, adjust = "none")
print(pairs_traj_con)

simple_trajectory <- rbind(
  as.data.frame(pairs_traj_amb) |> mutate(HV_primary = "ambiguous"),
  as.data.frame(pairs_traj_con) |> mutate(HV_primary = "congruent")
)

# 4.2 ROI duration simple effects (HV x ROI interaction)
cat("\n--- 4.2 ROI duration simple effects ---\n")

roi_duration_PT <- roi_duration_data |> filter(ROI == "PT")
roi_duration_CU <- roi_duration_data |> filter(ROI == "CU")
roi_duration_Catcher <- roi_duration_data |> filter(ROI == "Catcher")

cat("\n=== HV paired t-test at PT ===\n")
t_test_PT <- t.test(Duration ~ HV_primary, data = roi_duration_PT, paired = TRUE)
print(t_test_PT)

cat("\n=== HV paired t-test at CU ===\n")
t_test_CU <- t.test(Duration ~ HV_primary, data = roi_duration_CU, paired = TRUE)
print(t_test_CU)

cat("\n=== HV paired t-test at Catcher ===\n")
t_test_Catcher <- t.test(Duration ~ HV_primary, data = roi_duration_Catcher, paired = TRUE)
print(t_test_Catcher)

model_dur_PT <- aov(Duration ~ HV_primary + Error(subject_id/HV_primary), 
                    data = roi_duration_PT)
emm_dur_PT <- emmeans(model_dur_PT, ~ HV_primary)
pairs_dur_PT <- pairs(emm_dur_PT, adjust = "none")

model_dur_CU <- aov(Duration ~ HV_primary + Error(subject_id/HV_primary), 
                    data = roi_duration_CU)
emm_dur_CU <- emmeans(model_dur_CU, ~ HV_primary)
pairs_dur_CU <- pairs(emm_dur_CU, adjust = "none")

model_dur_Catcher <- aov(Duration ~ HV_primary + Error(subject_id/HV_primary), 
                         data = roi_duration_Catcher)
emm_dur_Catcher <- emmeans(model_dur_Catcher, ~ HV_primary)
pairs_dur_Catcher <- pairs(emm_dur_Catcher, adjust = "none")

simple_duration <- rbind(
  as.data.frame(pairs_dur_PT) |> mutate(ROI = "PT"),
  as.data.frame(pairs_dur_CU) |> mutate(ROI = "CU"),
  as.data.frame(pairs_dur_Catcher) |> mutate(ROI = "Catcher")
)

# 4.3 ROI fixation count simple effects
cat("\n--- 4.3 ROI fixation count simple effects ---\n")

roi_fixations_PT <- roi_fixations_data |> filter(ROI == "PT")
roi_fixations_CU <- roi_fixations_data |> filter(ROI == "CU")
roi_fixations_Catcher <- roi_fixations_data |> filter(ROI == "Catcher")

cat("\n=== HV paired t-test at PT ===\n")
t_test_fix_PT <- t.test(Fixations ~ HV_primary, data = roi_fixations_PT, paired = TRUE)
print(t_test_fix_PT)

cat("\n=== HV paired t-test at CU ===\n")
t_test_fix_CU <- t.test(Fixations ~ HV_primary, data = roi_fixations_CU, paired = TRUE)
print(t_test_fix_CU)

cat("\n=== HV paired t-test at Catcher ===\n")
t_test_fix_Catcher <- t.test(Fixations ~ HV_primary, data = roi_fixations_Catcher, paired = TRUE)
print(t_test_fix_Catcher)

model_fix_PT <- aov(Fixations ~ HV_primary + Error(subject_id/HV_primary), 
                    data = roi_fixations_PT)
emm_fix_PT <- emmeans(model_fix_PT, ~ HV_primary)
pairs_fix_PT <- pairs(emm_fix_PT, adjust = "none")

model_fix_CU <- aov(Fixations ~ HV_primary + Error(subject_id/HV_primary), 
                    data = roi_fixations_CU)
emm_fix_CU <- emmeans(model_fix_CU, ~ HV_primary)
pairs_fix_CU <- pairs(emm_fix_CU, adjust = "none")

model_fix_Catcher <- aov(Fixations ~ HV_primary + Error(subject_id/HV_primary), 
                         data = roi_fixations_Catcher)
emm_fix_Catcher <- emmeans(model_fix_Catcher, ~ HV_primary)
pairs_fix_Catcher <- pairs(emm_fix_Catcher, adjust = "none")

simple_fixations <- rbind(
  as.data.frame(pairs_fix_PT) |> mutate(ROI = "PT"),
  as.data.frame(pairs_fix_CU) |> mutate(ROI = "CU"),
  as.data.frame(pairs_fix_Catcher) |> mutate(ROI = "Catcher")
)

# 4.4 Summary table
cat("\n--- 4.4 Simple effects summary table ---\n")

traj_results <- simple_trajectory |>
  mutate(Metric = "Trajectory Overlap (I/U)", HV = HV_primary,
         Comparison = as.character(contrast)) |>
  select(Metric, HV, Comparison, estimate, SE, df, t.ratio, p.value)

duration_results <- simple_duration |>
  mutate(Metric = "Duration (ms)", HV = as.character(contrast),
         Comparison = ROI) |>
  select(Metric, HV, Comparison, estimate, SE, df, t.ratio, p.value)

fixations_results <- simple_fixations |>
  mutate(Metric = "Fixation Count", HV = as.character(contrast),
         Comparison = ROI) |>
  select(Metric, HV, Comparison, estimate, SE, df, t.ratio, p.value)

simple_effects_summary <- rbind(traj_results, duration_results, fixations_results)

simple_effects_summary <- simple_effects_summary |>
  mutate(
    estimate = round(estimate, 4),
    SE = round(SE, 4),
    df = round(df, 0),
    t.ratio = round(t.ratio, 3),
    p_formatted = ifelse(p.value < 0.001, "<.001", 
                         ifelse(p.value < 0.01, sprintf("%.3f", p.value),
                                sprintf("%.4f", p.value))),
    sig = ifelse(p.value < 0.05, "*", "")
  ) |>
  select(Metric, HV, Comparison, estimate, SE, df, t.ratio, p_formatted, sig)

cat("\n========== Simple Effects Summary ==========\n")
print(simple_effects_summary, row.names = FALSE)

write.csv(simple_effects_summary, file.path(data_dir, "simple_effects_summary.csv"), 
          row.names = FALSE, fileEncoding = "UTF-8")
cat("\nSimple effects summary saved to:", file.path(data_dir, "simple_effects_summary.csv"), "\n")

# ---------- 5. Summary ----------

cat("\n========== H2 Eye Analysis Summary ==========\n")
cat("Trajectory overlap: 2(HV) x 5(SU) repeated-measures ANOVA\n")
cat("ROI gaze proportion: 2(HV) x 3(ROI) repeated-measures ANOVA\n")
cat("ROI fixation duration: 2(HV) x 3(ROI) repeated-measures ANOVA\n")
cat("ROI fixation count: 2(HV) x 3(ROI) repeated-measures ANOVA\n")
cat("Simple effects for significant interactions\n")
cat("\nFigures saved to:", figures_dir, "\n")
cat("Analysis complete!\n")
