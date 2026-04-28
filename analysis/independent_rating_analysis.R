library(dplyr)
library(ggplot2)
library(ez)
library(tidyr)

analysis_dir <- getwd()
project_root <- normalizePath(file.path(analysis_dir, ".."), winslash = "/", mustWork = TRUE)
rating_data_candidates <- c(
  file.path(analysis_dir, "rating_data"),
  file.path(project_root, "experiment", "正式实验", "rating_data")
)
rating_data_root <- rating_data_candidates[file.exists(rating_data_candidates)][1]
if (is.na(rating_data_root)) {
  stop("未找到评分数据目录。")
}
output_dir <- file.path(analysis_dir, "rating_results")
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

cat("评分数据目录:\n", rating_data_root, "\n")

rating_files <- list.files(
  path = rating_data_root,
  pattern = "ratings\\.csv$",
  recursive = TRUE,
  full.names = TRUE
)

if (length(rating_files) == 0) {
  stop("未找到评分数据文件，请先运行 rating_experiment.py 收集 ratings.csv。")
}

cat("找到评分数据文件数:", length(rating_files), "\n")
print(rating_files)

rating_raw <- bind_rows(lapply(rating_files, read.csv, stringsAsFactors = FALSE))

required_cols <- c("SubjectID", "BoardID", "SU_level", "StraightTrajectoryExists", "RT_ms")
missing_cols <- setdiff(required_cols, names(rating_raw))
if (length(missing_cols) > 0) {
  stop("评分数据缺少必要列: ", paste(missing_cols, collapse = ", "))
}

judgment <- rating_raw %>%
  mutate(
    SubjectID = factor(trimws(as.character(SubjectID))),
    BoardID = trimws(as.character(BoardID)),
    SU_level = factor(as.integer(SU_level), levels = 1:5),
    StraightTrajectoryExists = as.numeric(StraightTrajectoryExists),
    RT_ms = as.numeric(RT_ms)
  ) %>%
  filter(!is.na(SubjectID), !is.na(SU_level), !is.na(StraightTrajectoryExists))

cat("\n总判断条目:", nrow(judgment), "\n")
cat("被试数:", n_distinct(judgment$SubjectID), "\n")
cat("项目数:", n_distinct(judgment$BoardID), "\n")
cat("SU水平分布:\n")
print(table(judgment$SU_level, useNA = "ifany"))

overall_subject_summary <- judgment %>%
  group_by(SubjectID) %>%
  summarise(
    overall_yes_rate = mean(StraightTrajectoryExists, na.rm = TRUE),
    overall_mean_rt = mean(RT_ms, na.rm = TRUE),
    n_trials = n(),
    .groups = "drop"
  )

overall_su_counts <- judgment %>%
  group_by(SU_level) %>%
  summarise(
    n_yes = sum(StraightTrajectoryExists, na.rm = TRUE),
    n_total = n(),
    overall_yes_rate = mean(StraightTrajectoryExists, na.rm = TRUE),
    .groups = "drop"
  )

subject_su_summary <- judgment %>%
  group_by(SubjectID, SU_level) %>%
  summarise(
    yes_rate = mean(StraightTrajectoryExists, na.rm = TRUE),
    mean_rt = mean(RT_ms, na.rm = TRUE),
    n_trials = n(),
    .groups = "drop"
  )

coverage_table <- subject_su_summary %>%
  count(SubjectID) %>%
  rename(n_su_levels = n)

cat("\n每位被试包含的 SU 水平数量分布:\n")
print(table(coverage_table$n_su_levels))

subject_su_complete <- subject_su_summary %>%
  group_by(SubjectID) %>%
  filter(n() == 5) %>%
  ungroup()

if (n_distinct(subject_su_complete$SubjectID) < 2) {
  stop("完整覆盖 5 个 SU 水平的被试少于 2 人，无法进行重复测量 ANOVA。")
}

cat("\n纳入 ANOVA 的完整被试数:", n_distinct(subject_su_complete$SubjectID), "\n")

anova_judgment <- ezANOVA(
  data = subject_su_complete,
  dv = yes_rate,
  wid = SubjectID,
  within = .(SU_level),
  type = 3,
  detailed = TRUE
)

cat("\n========== 判断 ANOVA：SU 对直线轨迹存在性判断的影响 ==========\n")
print(anova_judgment)

anova_rt <- ezANOVA(
  data = subject_su_complete,
  dv = mean_rt,
  wid = SubjectID,
  within = .(SU_level),
  type = 3,
  detailed = TRUE
)

cat("\n========== RT ANOVA：SU 对二分判断反应时的影响 ==========\n")
print(anova_rt)

judgment_summary <- subject_su_complete %>%
  group_by(SU_level) %>%
  summarise(
    n_subjects = n_distinct(SubjectID),
    mean_yes_rate = mean(yes_rate, na.rm = TRUE),
    sd_yes_rate = sd(yes_rate, na.rm = TRUE),
    se_yes_rate = sd_yes_rate / sqrt(n_subjects),
    sd_rt = sd(mean_rt, na.rm = TRUE),
    mean_rt = mean(mean_rt, na.rm = TRUE),
    se_rt = sd_rt / sqrt(n_subjects),
    .groups = "drop"
  )

item_summary <- judgment %>%
  group_by(BoardID, SU_level) %>%
  summarise(
    item_yes_rate = mean(StraightTrajectoryExists, na.rm = TRUE),
    item_mean_rt = mean(RT_ms, na.rm = TRUE),
    n_raters = n_distinct(SubjectID),
    .groups = "drop"
  )

item_su_summary <- item_summary %>%
  group_by(SU_level) %>%
  summarise(
    n_items = n(),
    mean_item_yes_rate = mean(item_yes_rate, na.rm = TRUE),
    sd_item_yes_rate = sd(item_yes_rate, na.rm = TRUE),
    se_item_yes_rate = sd_item_yes_rate / sqrt(n_items),
    .groups = "drop"
  )

chance_test_results <- bind_rows(lapply(split(subject_su_complete, subject_su_complete$SU_level), function(df) {
  if (nrow(df) < 2) {
    return(data.frame(
      SU_level = as.character(df$SU_level[1]),
      n_subjects = nrow(df),
      mean_yes_rate = mean(df$yes_rate, na.rm = TRUE),
      t = NA_real_,
      df = NA_real_,
      p = NA_real_
    ))
  }

  if (sd(df$yes_rate, na.rm = TRUE) == 0) {
    mean_yes_rate <- mean(df$yes_rate, na.rm = TRUE)
    return(data.frame(
      SU_level = as.character(df$SU_level[1]),
      n_subjects = nrow(df),
      mean_yes_rate = mean_yes_rate,
      t = ifelse(mean_yes_rate > 0.5, Inf, ifelse(mean_yes_rate < 0.5, -Inf, 0)),
      df = nrow(df) - 1,
      p = ifelse(mean_yes_rate == 0.5, 1, 0)
    ))
  }

  test_res <- t.test(df$yes_rate, mu = 0.5)
  data.frame(
    SU_level = as.character(df$SU_level[1]),
    n_subjects = nrow(df),
    mean_yes_rate = mean(df$yes_rate, na.rm = TRUE),
    t = unname(test_res$statistic),
    df = unname(test_res$parameter),
    p = test_res$p.value
  )
}))

pairwise_judgment <- pairwise.t.test(
  x = subject_su_complete$yes_rate,
  g = subject_su_complete$SU_level,
  paired = TRUE,
  p.adjust.method = "holm"
)

pairwise_judgment_df <- as.data.frame(as.table(pairwise_judgment$p.value)) %>%
  rename(SU_level_1 = Var1, SU_level_2 = Var2, p_holm = Freq) %>%
  filter(!is.na(p_holm))

write.csv(overall_subject_summary, file.path(output_dir, "judgment_subject_overall_summary.csv"), row.names = FALSE)
write.csv(overall_su_counts, file.path(output_dir, "judgment_overall_counts_by_su.csv"), row.names = FALSE)
write.csv(subject_su_complete, file.path(output_dir, "judgment_subject_su_summary.csv"), row.names = FALSE)
write.csv(judgment_summary, file.path(output_dir, "judgment_summary_by_su.csv"), row.names = FALSE)
write.csv(item_summary, file.path(output_dir, "judgment_item_summary.csv"), row.names = FALSE)
write.csv(item_su_summary, file.path(output_dir, "judgment_item_summary_by_su.csv"), row.names = FALSE)
write.csv(as.data.frame(anova_judgment$ANOVA), file.path(output_dir, "judgment_anova_su.csv"), row.names = FALSE)
write.csv(as.data.frame(anova_rt$ANOVA), file.path(output_dir, "judgment_rt_anova_su.csv"), row.names = FALSE)
write.csv(chance_test_results, file.path(output_dir, "judgment_yes_rate_vs_0.5.csv"), row.names = FALSE)
write.csv(pairwise_judgment_df, file.path(output_dir, "judgment_pairwise_holm.csv"), row.names = FALSE)

p_judgment <- ggplot(judgment_summary, aes(x = SU_level, y = mean_yes_rate, group = 1)) +
  geom_line(linewidth = 1.1, color = "#2C7FB8") +
  geom_point(size = 3.5, color = "#2C7FB8") +
  geom_errorbar(aes(ymin = mean_yes_rate - se_yes_rate, ymax = mean_yes_rate + se_yes_rate), width = 0.12, linewidth = 0.8, color = "#2C7FB8") +
  labs(
    title = "congruent 条件下判断为存在直线轨迹的比例随 SU 的变化",
    x = "SU 水平",
    y = "判断为存在的比例"
  ) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.1)) +
  theme_bw() +
  theme(
    plot.title = element_text(size = 14, hjust = 0.5),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 11)
  )

ggsave(
  filename = file.path(output_dir, "judgment_yes_rate_by_su.png"),
  plot = p_judgment,
  width = 8,
  height = 5.5,
  dpi = 300
)

p_judgment_subjects <- ggplot(subject_su_complete, aes(x = SU_level, y = yes_rate, group = SubjectID)) +
  geom_line(alpha = 0.25, color = "#9ECAE1") +
  geom_point(alpha = 0.35, color = "#3182BD", size = 1.8) +
  stat_summary(aes(group = 1), fun = mean, geom = "line", linewidth = 1.2, color = "#08519C") +
  stat_summary(aes(group = 1), fun = mean, geom = "point", size = 3.2, color = "#08519C") +
  labs(
    title = "各被试在不同 SU 水平下判断为存在直线轨迹的比例",
    x = "SU 水平",
    y = "判断为存在的比例"
  ) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.1)) +
  theme_bw() +
  theme(
    plot.title = element_text(size = 14, hjust = 0.5),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 11)
  )

ggsave(
  filename = file.path(output_dir, "judgment_yes_rate_by_su_subjects.png"),
  plot = p_judgment_subjects,
  width = 8,
  height = 5.5,
  dpi = 300
)

p_rt <- ggplot(judgment_summary, aes(x = SU_level, y = mean_rt, group = 1)) +
  geom_line(linewidth = 1.1, color = "#D95F0E") +
  geom_point(size = 3.5, color = "#D95F0E") +
  geom_errorbar(aes(ymin = mean_rt - se_rt, ymax = mean_rt + se_rt), width = 0.12, linewidth = 0.8, color = "#D95F0E") +
  labs(
    title = "congruent 条件下二分判断反应时随 SU 的变化",
    x = "SU 水平",
    y = "平均反应时 (ms)"
  ) +
  theme_bw() +
  theme(
    plot.title = element_text(size = 14, hjust = 0.5),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 11)
  )

ggsave(
  filename = file.path(output_dir, "judgment_rt_by_su.png"),
  plot = p_rt,
  width = 8,
  height = 5.5,
  dpi = 300
)

p_item <- ggplot(item_su_summary, aes(x = SU_level, y = mean_item_yes_rate, group = 1)) +
  geom_line(linewidth = 1.1, color = "#31A354") +
  geom_point(size = 3.5, color = "#31A354") +
  geom_errorbar(aes(ymin = mean_item_yes_rate - se_item_yes_rate, ymax = mean_item_yes_rate + se_item_yes_rate), width = 0.12, linewidth = 0.8, color = "#31A354") +
  labs(
    title = "项目层面判断为存在直线轨迹的比例随 SU 的变化",
    x = "SU 水平",
    y = "项目平均存在比例"
  ) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.1)) +
  theme_bw() +
  theme(
    plot.title = element_text(size = 14, hjust = 0.5),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 11)
  )

ggsave(
  filename = file.path(output_dir, "judgment_item_yes_rate_by_su.png"),
  plot = p_item,
  width = 8,
  height = 5.5,
  dpi = 300
)

cat("\n========== 描述统计：被试层判断为存在的比例 ==========" , "\n")
print(judgment_summary)

cat("\n========== 原始条目计数：各 SU 水平判断为存在的次数 ==========" , "\n")
print(overall_su_counts)

cat("\n========== 描述统计：项目层判断为存在的比例 ==========" , "\n")
print(item_su_summary)

cat("\n========== 各 SU 水平是否高于 0.5 ==========" , "\n")
print(chance_test_results)

cat("\n========== SU 水平两两比较（Holm 校正） ==========" , "\n")
print(pairwise_judgment_df)

cat("\n分析完成，输出目录:\n", output_dir, "\n")
