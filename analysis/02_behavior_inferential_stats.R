# ------------------------------------------------------------
# 02_behavior_inferential_stats.R
#
# Inferential statistics for behavioral data.
# Reads behavior_summary.csv (output of 01_preprocess_behavior.py) and runs:
#   1. HV 3-level manipulation check (Accuracy / RT / Confidence)
#      via repeated-measures ANOVA + pairwise comparisons
#   2. 2x5 HV (congruent vs ambiguous) x SU (1-5) repeated-measures ANOVA
#      (Accuracy / RT), with simple effects analysis
#   3. Visualization of main effects and interactions
#
# Dependencies: ez, emmeans, ggplot2, dplyr
#   install.packages(c("ez", "emmeans", "ggplot2", "dplyr"))
#
# Usage:
#   cd analysis/
#   Rscript 02_behavior_inferential_stats.R
# ------------------------------------------------------------

library(ez)
library(emmeans)
library(ggplot2)
library(dplyr)

# Path configuration (relative to this script in analysis/)
script_dir <- tryCatch(
  dirname(normalizePath(sys.frame(1)$ofile)),
  error = function(e) getwd()
)
project_root <- dirname(script_dir)
output_dir   <- file.path(project_root, "data", "processed")
figures_dir  <- file.path(project_root, "figures")
dir.create(figures_dir, showWarnings = FALSE, recursive = TRUE)

summary_file <- file.path(output_dir, "behavior_summary.csv")

cat("================ Behavioral Inferential Statistics ================\n")
cat("Data directory:", output_dir, "\n\n")

if (!file.exists(summary_file)) {
  stop("behavior_summary.csv not found. Please run 01_preprocess_behavior.py first.")
}

dat_sum <- read.csv(summary_file, stringsAsFactors = TRUE)

# Factor setup
dat_sum$SubjectID      <- factor(dat_sum$SubjectID)
dat_sum$Heuristic_Type <- factor(dat_sum$Heuristic_Type,
                                 levels = c("congruent", "ambiguous", "conflict"))

# Build HV 3-level table (collapse across SU)
rm_hv <- aggregate(cbind(Mean_Accuracy, Mean_RT, Mean_Confidence) ~
                     SubjectID + Heuristic_Type,
                   data = dat_sum,
                   FUN = mean)

# Build 2x5 design table (congruent vs ambiguous, SU = 1-5)
rm_2x5 <- subset(dat_sum,
                 Heuristic_Type %in% c("congruent", "ambiguous") & SU_Level > 0)

# Ensure factor types
rm_hv$SubjectID       <- factor(rm_hv$SubjectID)
rm_hv$Heuristic_Type  <- factor(rm_hv$Heuristic_Type,
                                levels = c("congruent", "ambiguous", "conflict"))

rm_2x5$SubjectID      <- factor(rm_2x5$SubjectID)
rm_2x5$Heuristic_Type <- factor(rm_2x5$Heuristic_Type,
                                levels = c("congruent", "ambiguous"))
rm_2x5$SU_Level       <- factor(rm_2x5$SU_Level,
                                levels = sort(unique(rm_2x5$SU_Level)))


# ------------------------------------------------------------
# 1. HV 3-level manipulation check
# ------------------------------------------------------------

cat("[1] HV 3-level manipulation check (ambiguous / congruent / conflict)\n")

## 1.1 Accuracy
cat("\n[1.1] Accuracy repeated-measures ANOVA\n")
anova_acc <- ezANOVA(
  data    = rm_hv,
  dv      = Mean_Accuracy,
  wid     = SubjectID,
  within  = .(Heuristic_Type),
  type    = 3,
  detailed = TRUE
)
print(anova_acc)

cat("\n[1.1] Pairwise comparisons (Accuracy)\n")
emm_acc <- emmeans(
  lm(Mean_Accuracy ~ Heuristic_Type + SubjectID, data = rm_hv),
  ~ Heuristic_Type
)
post_acc <- contrast(emm_acc, method = "pairwise", adjust = "none")
print(post_acc)

## 1.2 RT
cat("\n[1.2] RT repeated-measures ANOVA\n")
anova_rt <- ezANOVA(
  data    = rm_hv,
  dv      = Mean_RT,
  wid     = SubjectID,
  within  = .(Heuristic_Type),
  type    = 3,
  detailed = TRUE
)
print(anova_rt)

cat("\n[1.2] Pairwise comparisons (RT)\n")
emm_rt <- emmeans(
  lm(Mean_RT ~ Heuristic_Type + SubjectID, data = rm_hv),
  ~ Heuristic_Type
)
post_rt <- contrast(emm_rt, method = "pairwise", adjust = "none")
print(post_rt)

## 1.3 HV main effect plots
cat("\n[1.3] Generating HV main effect plots...\n")

fig_hv_acc <- ggplot(rm_hv, aes(x = Heuristic_Type, y = Mean_Accuracy)) +
  stat_summary(fun = mean, geom = "bar", fill = "#4C72B0") +
  stat_summary(fun.data = mean_se, geom = "errorbar", width = .2) +
  ylab("Accuracy") + xlab("Heuristic Type") +
  theme_minimal(base_size = 14)
ggsave(file.path(figures_dir, "fig_HV_Accuracy.png"),
       fig_hv_acc, width = 5, height = 4, dpi = 300)

fig_hv_rt <- ggplot(rm_hv, aes(x = Heuristic_Type, y = Mean_RT)) +
  stat_summary(fun = mean, geom = "bar", fill = "#55A868") +
  stat_summary(fun.data = mean_se, geom = "errorbar", width = .2) +
  ylab("RT (ms)") + xlab("Heuristic Type") +
  theme_minimal(base_size = 14)
ggsave(file.path(figures_dir, "fig_HV_RT.png"),
       fig_hv_rt, width = 5, height = 4, dpi = 300)


# ------------------------------------------------------------
# 2. 2x5 HV x SU repeated-measures ANOVA
# ------------------------------------------------------------

cat("\n[2] 2x5 HV (congruent vs ambiguous) x SU (1-5) ANOVA\n")

## 2.0 Descriptive statistics
cat("\n[2.0] Descriptive statistics for 2x5 design\n")
desc_stats <- rm_2x5 %>%
  group_by(Heuristic_Type, SU_Level) %>%
  summarise(
    Acc_M = mean(Mean_Accuracy, na.rm = TRUE),
    Acc_SD = sd(Mean_Accuracy, na.rm = TRUE),
    RT_M = mean(Mean_RT, na.rm = TRUE),
    RT_SD = sd(Mean_RT, na.rm = TRUE),
    N = n(),
    .groups = 'drop'
  )
print(desc_stats)
write.csv(desc_stats, file.path(output_dir, "desc_stats_2x5.csv"), row.names = FALSE)

## 2.1 Accuracy
cat("\n[2.1] Accuracy 2x5 ANOVA\n")
anova_acc_2x5 <- ezANOVA(
  data    = rm_2x5,
  dv      = Mean_Accuracy,
  wid     = SubjectID,
  within  = .(Heuristic_Type, SU_Level),
  type    = 3,
  detailed = TRUE
)
print(anova_acc_2x5)

## 2.2 RT
cat("\n[2.2] RT 2x5 ANOVA\n")
anova_rt_2x5 <- ezANOVA(
  data    = rm_2x5,
  dv      = Mean_RT,
  wid     = SubjectID,
  within  = .(Heuristic_Type, SU_Level),
  type    = 3,
  detailed = TRUE
)
print(anova_rt_2x5)

## 2.3 Simple effects
cat("\n[2.3] Simple effects analysis (HV x SU interaction)\n")

cat("\n[2.3.1] SU effect at each HV level (Accuracy)\n")
emm_acc_2x5 <- emmeans(
  lm(Mean_Accuracy ~ Heuristic_Type * SU_Level + SubjectID, data = rm_2x5),
  ~ SU_Level | Heuristic_Type
)
simple_acc <- contrast(emm_acc_2x5, method = "pairwise", adjust = "none")
print(simple_acc)

cat("\n[2.3.2] SU effect at each HV level (RT)\n")
emm_rt_2x5 <- emmeans(
  lm(Mean_RT ~ Heuristic_Type * SU_Level + SubjectID, data = rm_2x5),
  ~ SU_Level | Heuristic_Type
)
simple_rt <- contrast(emm_rt_2x5, method = "pairwise", adjust = "none")
print(simple_rt)

## 2.4 Interaction plots
cat("\n[2.4] Generating 2x5 interaction plots...\n")

fig_2x5_acc <- ggplot(rm_2x5, aes(x = SU_Level, y = Mean_Accuracy,
                                   color = Heuristic_Type,
                                   group = Heuristic_Type)) +
  stat_summary(fun = mean, geom = "line") +
  stat_summary(fun = mean, geom = "point") +
  stat_summary(fun.data = mean_se, geom = "errorbar", width = .15) +
  ylab("Accuracy") + xlab("SU Level") +
  theme_minimal(base_size = 14)
ggsave(file.path(figures_dir, "fig_2x5_Accuracy.png"),
       fig_2x5_acc, width = 6, height = 4, dpi = 300)

fig_2x5_rt <- ggplot(rm_2x5, aes(x = SU_Level, y = Mean_RT,
                                  color = Heuristic_Type,
                                  group = Heuristic_Type)) +
  stat_summary(fun = mean, geom = "line") +
  stat_summary(fun = mean, geom = "point") +
  stat_summary(fun.data = mean_se, geom = "errorbar", width = .15) +
  ylab("RT (ms)") + xlab("SU Level") +
  theme_minimal(base_size = 14)
ggsave(file.path(figures_dir, "fig_2x5_RT.png"),
       fig_2x5_rt, width = 6, height = 4, dpi = 300)

cat("\n================ Behavioral inferential statistics complete ================\n")
