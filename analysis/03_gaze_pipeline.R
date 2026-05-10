# ============================================================
# 03_gaze_pipeline.R
# Eye-tracking preprocessing + ROI assignment + trial-level eye metrics
#
# Reads raw ASC files and per-subject behavioral CSVs, then:
#   1. Parses ASC eye samples, blink events, and trial messages
#   2. Preprocesses gaze: blink removal, interpolation, smoothing
#   3. Assigns ROIs (static, predicted trajectory PT, critical uncertainty CU)
#   4. Computes trial-level eye metrics (fixation, tracking, trajectory overlap)
#   5. Outputs combined CSV and RData for all subjects
#
# Dependencies: dplyr, tidyr, purrr, jsonlite, zoo
#   install.packages(c("dplyr", "tidyr", "purrr", "jsonlite", "zoo"))
#
# Usage:
#   cd analysis/
#   Rscript 03_gaze_pipeline.R
# ============================================================

library(dplyr)
library(tidyr)
library(purrr)
library(jsonlite)
library(zoo)

# ---------- Global parameters ----------

SAMPLING_RATE_HZ <- 1000
SAMPLE_DT_MS     <- 1000 / SAMPLING_RATE_HZ

# Screen & stimulus (consistent with experiment config.py)
SCREEN_W <- 1920
SCREEN_H <- 1080

CATCHER_WIDTH  <- 360
CATCHER_HEIGHT <- 30
CATCHER_Y      <- SCREEN_H - 100
CATCHER_LEFT_X <- SCREEN_W * 0.40
CATCHER_RIGHT_X<- SCREEN_W * 0.60

BALL_RADIUS    <- 10
MIDLINE_X      <- (CATCHER_LEFT_X + CATCHER_RIGHT_X) / 2

# Maximum saccade amplitude (pixels): slightly larger than screen diagonal
MAX_SACCADE_AMPLITUDE_PX <- sqrt(SCREEN_W^2 + SCREEN_H^2) * 1.5

# Jitter parameters (consistent with config.py)
N_JITTER_SIMULATIONS <- 100
JITTER_POSITION_STD  <- 2

# PT region: True trajectory +/- fixed width (~2 degrees, ~70px)
PT_WIDTH_PX <- 70

# CU threshold (>20% jitter trajectories land on opposite side)
CU_BIFURC_THRESHOLD <- 0.20

# Time windows (TRIAL_START = TRIALID at 0ms)
TASK_ONSET_MS <- 800
INIT_START_MS <- 800
INIT_END_MS   <- 1600

# Gaze smoothing window (5ms)
GAZE_SMOOTH_WIN <- 5

BLINK_EXTEND_MS        <- 100
GAZE_MAX_INTERP_GAP_MS <- 75
GAZE_MAX_INTERP_GAP_N  <- as.integer(GAZE_MAX_INTERP_GAP_MS / SAMPLE_DT_MS)
INIT_MAX_TRACKLOSS_PROP <- 0.5
TASK_MAX_TRACKLOSS_PROP <- 0.5

# ---------- Common functions: read ASC, trial segmentation ----------

read_asc_raw <- function(asc_path) {
  lines <- readLines(asc_path, encoding = "UTF-8", warn = FALSE)
  
  sample_lines <- grep("^[0-9]", lines, value = TRUE, useBytes = TRUE)
  
  samples <- read.table(
    text = sample_lines,
    header = FALSE, fill = TRUE,
    na.strings = c(".", "NaN", ""),
    stringsAsFactors = FALSE
  )
  samples <- samples[, 1:4]
  names(samples) <- c("time", "x", "y", "pupil")
  
  eblink_lines <- grep("^EBLINK", lines, value = TRUE, useBytes = TRUE)
  
  if (length(eblink_lines) > 0) {
    eblink <- read.table(text = eblink_lines, header = FALSE, fill = TRUE,
                         stringsAsFactors = FALSE)
    eblink <- eblink[, c(3,4,5)]
    names(eblink) <- c("start", "end", "dur")
  } else {
    eblink <- data.frame(start=numeric(0), end=numeric(0), dur=numeric(0))
  }
  
  msg_lines <- grep("^MSG", lines, value = TRUE, useBytes = TRUE)
  
  msg <- read.table(text = msg_lines, header = FALSE, fill = TRUE,
                    stringsAsFactors = FALSE)
  names(msg)[1:3] <- c("tag", "time", "code")
  
  list(samples = samples,
       eblink  = eblink,
       msg     = msg)
}

get_trial_bounds <- function(msg) {
  trial_rows <- msg[msg$code == "TRIALID", ]
  if (nrow(trial_rows) == 0) stop("No TRIALID messages found in ASC")
  trial_ids   <- trial_rows$V4
  start_times <- as.numeric(trial_rows$time)
  end_times   <- c(start_times[-1] - 1, Inf)
  data.frame(
    trial_id   = as.integer(trial_ids),
    start_time = start_times,
    end_time   = end_times
  )
}

assign_trial_to_times <- function(df, time_col, trial_bounds) {
  df$trial_id      <- NA_integer_
  df$time_in_trial <- NA_real_
  for (i in seq_len(nrow(trial_bounds))) {
    s <- trial_bounds$start_time[i]
    e <- trial_bounds$end_time[i]
    tid <- trial_bounds$trial_id[i]
    mask <- df[[time_col]] >= s & df[[time_col]] <= e
    df$trial_id[mask]      <- tid
    df$time_in_trial[mask] <- df[[time_col]][mask] - s
  }
  df
}

# ---------- stimulus_config & ROI helpers ----------

load_stim_config <- function(path) {
  read.csv(path, stringsAsFactors = FALSE)
}

parse_true_traj <- function(traj_json) {
  if (is.na(traj_json) || traj_json == "") {
    return(data.frame(t = numeric(0), x = numeric(0), y = numeric(0)))
  }
  df <- fromJSON(traj_json)
  df[, c("t", "x", "y")]
}

parse_planks <- function(planks_json) {
  if (is.na(planks_json) || planks_json == "") {
    return(data.frame(x = numeric(0), y = numeric(0)))
  }
  df <- fromJSON(planks_json)
  df[, c("x", "y")]
}

in_rect <- function(x, y, xmin, xmax, ymin, ymax) {
  !is.na(x) & !is.na(y) &
    x >= xmin & x <= xmax &
    y >= ymin & y <= ymax
}

point_to_polyline_dist <- function(x, y, traj_df) {
  dx <- traj_df$x - x
  dy <- traj_df$y - y
  sqrt(min(dx * dx + dy * dy, na.rm = TRUE))
}

points_to_polyline_dist <- function(xs, ys, traj_df) {
  vapply(seq_along(xs),
         function(i) point_to_polyline_dist(xs[i], ys[i], traj_df),
         numeric(1))
}

build_static_rois_for_board <- function(board_row) {
  left_catcher <- list(
    name = "catcher_left",
    xmin = CATCHER_LEFT_X - CATCHER_WIDTH / 2,
    xmax = CATCHER_LEFT_X + CATCHER_WIDTH / 2,
    ymin = CATCHER_Y      - CATCHER_HEIGHT / 2,
    ymax = CATCHER_Y      + CATCHER_HEIGHT / 2
  )
  right_catcher <- list(
    name = "catcher_right",
    xmin = CATCHER_RIGHT_X - CATCHER_WIDTH / 2,
    xmax = CATCHER_RIGHT_X + CATCHER_WIDTH / 2,
    ymin = CATCHER_Y       - CATCHER_HEIGHT / 2,
    ymax = CATCHER_Y       + CATCHER_HEIGHT / 2
  )
  x0 <- board_row$Ball_start_x
  y0 <- board_row$Ball_start_y
  ball_start <- list(
    name = "ball_start",
    xmin = x0 - BALL_RADIUS,
    xmax = x0 + BALL_RADIUS,
    ymin = y0 - BALL_RADIUS,
    ymax = y0 + BALL_RADIUS
  )
  planks_df <- parse_planks(board_row$Planks)
  if (nrow(planks_df) > 0) {
    planks_list <- purrr::pmap(planks_df, function(x, y, ...) {
      list(
        xmin = x - 100 / 2,
        xmax = x + 100 / 2,
        ymin = y - 10  / 2,
        ymax = y + 10  / 2
      )
    })
    xmin_all <- min(purrr::map_dbl(planks_list, "xmin"))
    xmax_all <- max(purrr::map_dbl(planks_list, "xmax"))
    ymin_all <- min(purrr::map_dbl(planks_list, "ymin"))
    ymax_all <- max(purrr::map_dbl(planks_list, "ymax"))
    planks_union <- list(
      name = "planks_all",
      xmin = xmin_all,
      xmax = xmax_all,
      ymin = ymin_all,
      ymax = ymax_all
    )
  } else {
    planks_union <- NULL
  }
  rois <- list(left_catcher, right_catcher, ball_start)
  if (!is.null(planks_union)) rois <- c(rois, list(planks_union))
  rois
}

assign_static_roi <- function(xs, ys, rois) {
  out <- rep("other", length(xs))
  for (r in rois) {
    mask <- in_rect(xs, ys, r$xmin, r$xmax, r$ymin, r$ymax)
    out[mask] <- r$name
  }
  out
}

assign_PT_roi <- function(xs, ys, traj_df, width_px = PT_WIDTH_PX) {
  if (nrow(traj_df) == 0) return(rep(FALSE, length(xs)))
  dists <- points_to_polyline_dist(xs, ys, traj_df)
  dists <= width_px
}

compute_CU_timepoints_sim <- function(board_row, true_traj_df,
                                      n_sim = N_JITTER_SIMULATIONS,
                                      sigma_x = JITTER_POSITION_STD) {
  if (nrow(true_traj_df) == 0) return(rep(FALSE, 0))
  if (!is.null(board_row$HV_primary) &&
      board_row$HV_primary == "conflict") {
    return(rep(FALSE, nrow(true_traj_df)))
  }
  if (!is.null(board_row$BoardID)) {
    bid_num <- as.integer(gsub("[^0-9]", "", board_row$BoardID))
    if (!is.na(bid_num)) set.seed(bid_num)
  }
  x_true    <- true_traj_df$x
  true_side <- ifelse(x_true >= MIDLINE_X, 1, -1)
  deltas <- rnorm(n_sim, mean = 0, sd = sigma_x)
  X_mat  <- outer(x_true, deltas, "+")
  
  cu_flags <- logical(length(x_true))
  for (i in seq_along(x_true)) {
    side_true_i <- true_side[i]
    side_jits_i <- ifelse(X_mat[i, ] >= MIDLINE_X, 1, -1)
    bif <- side_jits_i != side_true_i
    prop_bf <- mean(bif, na.rm = TRUE)
    cu_flags[i] <- prop_bf >= CU_BIFURC_THRESHOLD
  }
  cu_flags
}

assign_CU_roi <- function(xs, ys, traj_df, cu_time_flags,
                          width_px = PT_WIDTH_PX) {
  if (length(cu_time_flags) == 0 || all(!cu_time_flags)) {
    return(rep(FALSE, length(xs)))
  }
  if (length(cu_time_flags) != nrow(traj_df)) {
    stop("CU time flags length does not match True trajectory rows")
  }
  cu_traj <- traj_df[cu_time_flags, c("x", "y"), drop = FALSE]
  dists <- points_to_polyline_dist(xs, ys, cu_traj)
  dists <= width_px
}

# ---------- Gaze preprocessing: interpolation + smoothing ----------

preprocess_gaze <- function(samples, eblink, trial_bounds) {
  samples <- assign_trial_to_times(samples, "time", trial_bounds)
  samples <- samples[!is.na(samples$trial_id), ]
  
  samples$is_blink <- FALSE
  for (i in seq_len(nrow(eblink))) {
    s <- eblink$start[i] - BLINK_EXTEND_MS
    e <- eblink$end[i]   + BLINK_EXTEND_MS
    mask <- samples$time >= s & samples$time <= e
    samples$is_blink[mask] <- TRUE
  }
  
  samples <- samples %>%
    group_by(trial_id) %>%
    arrange(time) %>%
    mutate(
      x_raw = ifelse(is_blink | x < 0 | x > SCREEN_W |
                       is.na(pupil) | pupil <= 0, NA, x),
      y_raw = ifelse(is_blink | y < 0 | y > SCREEN_H |
                       is.na(pupil) | pupil <= 0, NA, y),
      x_interp = na.spline(x_raw, x = time, na.rm = FALSE,
                           maxgap = GAZE_MAX_INTERP_GAP_N),
      y_interp = na.spline(y_raw, x = time, na.rm = FALSE,
                           maxgap = GAZE_MAX_INTERP_GAP_N),
      x_interp = ifelse(x_interp < 0 | x_interp > SCREEN_W, NA, x_interp),
      y_interp = ifelse(y_interp < 0 | y_interp > SCREEN_H, NA, y_interp),
      x_smooth = rollapply(x_interp, width = GAZE_SMOOTH_WIN,
                           FUN = mean, fill = NA, align = "center"),
      y_smooth = rollapply(y_interp, width = GAZE_SMOOTH_WIN,
                           FUN = mean, fill = NA, align = "center")
    ) %>%
    ungroup()
  
  samples <- assign_trial_to_times(samples, "time", trial_bounds)
  samples
}

# ---------- Single-trial ROI assignment ----------

assign_rois_for_trial <- function(samples_trial, board_row) {
  true_traj   <- parse_true_traj(board_row$True_trajectory)
  static_rois <- build_static_rois_for_board(board_row)
  
  roi_static <- assign_static_roi(samples_trial$x_smooth,
                                  samples_trial$y_smooth,
                                  static_rois)
  roi_pt <- assign_PT_roi(samples_trial$x_smooth,
                          samples_trial$y_smooth,
                          true_traj)
  cu_flags_time <- compute_CU_timepoints_sim(board_row, true_traj)
  roi_cu <- assign_CU_roi(samples_trial$x_smooth,
                          samples_trial$y_smooth,
                          true_traj,
                          cu_flags_time)
  
  cbind(samples_trial,
        ROI_static = roi_static,
        ROI_PT     = roi_pt,
        ROI_CU     = roi_cu)
}

# ---------- ROI-level trial metrics ----------

count_runs <- function(mask) {
  if (length(mask) == 0 || all(is.na(mask))) return(NA_real_)
  mask[is.na(mask)] <- FALSE
  r <- rle(mask)
  sum(r$values)
}

compute_roi_metrics_for_trial <- function(samples_trial) {
  tid <- unique(samples_trial$trial_id)

  rt_ms <- NA_real_
  if ("RT" %in% names(samples_trial)) {
    rt_vals <- unique(samples_trial$RT)
    rt_vals <- rt_vals[!is.na(rt_vals)]
    if (length(rt_vals) == 1) rt_ms <- as.numeric(rt_vals)
  }
  task_end_ms <- if (!is.na(rt_ms) && rt_ms > 0) TASK_ONSET_MS + rt_ms else Inf

  task_mask <- samples_trial$time_in_trial >= TASK_ONSET_MS &
    samples_trial$time_in_trial <= task_end_ms &
    !is.na(samples_trial$x_smooth)
  task_all_mask <- samples_trial$time_in_trial >= TASK_ONSET_MS &
    samples_trial$time_in_trial <= task_end_ms

  pt_mask  <- task_mask & (samples_trial$ROI_PT %in% TRUE)
  cu_mask  <- task_mask & (samples_trial$ROI_CU %in% TRUE)
  cat_mask <- task_mask & (samples_trial$ROI_static %in% c("catcher_left", "catcher_right"))

  valid_samples      <- sum(task_mask)
  total_task_samples <- sum(task_all_mask)
  tl <- if (total_task_samples > 0) 1 - valid_samples / total_task_samples else NA_real_

  if (!is.na(tl) && tl > TASK_MAX_TRACKLOSS_PROP) {
    return(data.frame(
      trial_id = tid,
      Prop_PT = NA_real_, Prop_CU = NA_real_, Prop_catcher = NA_real_,
      Duration_PT = NA_real_, Duration_CU = NA_real_, Duration_catcher = NA_real_,
      Fixations_PT = NA_real_, Fixations_CU = NA_real_, Fixations_catcher = NA_real_,
      CU_visits = NA_real_, PT_entries = NA_real_
    ))
  }

  data.frame(
    trial_id         = tid,
    Prop_PT          = if (valid_samples > 0) sum(pt_mask)  / valid_samples else NA_real_,
    Prop_CU          = if (valid_samples > 0) sum(cu_mask)  / valid_samples else NA_real_,
    Prop_catcher     = if (valid_samples > 0) sum(cat_mask) / valid_samples else NA_real_,
    Duration_PT      = sum(pt_mask)  * SAMPLE_DT_MS,
    Duration_CU      = sum(cu_mask)  * SAMPLE_DT_MS,
    Duration_catcher = sum(cat_mask) * SAMPLE_DT_MS,
    Fixations_PT     = count_runs(pt_mask),
    Fixations_CU     = count_runs(cu_mask),
    Fixations_catcher = count_runs(cat_mask),
    CU_visits        = count_runs(cu_mask),
    PT_entries       = count_runs(pt_mask)
  )
}

compute_eye_metrics_for_trial <- function(samples_trial, true_traj_df) {
  tid <- unique(samples_trial$trial_id)

  rt_ms <- NA_real_
  if ("RT" %in% names(samples_trial)) {
    rt_vals <- unique(samples_trial$RT)
    rt_vals <- rt_vals[!is.na(rt_vals)]
    if (length(rt_vals) == 1) {
      rt_ms <- as.numeric(rt_vals)
    }
  }
  if (!is.na(rt_ms) && rt_ms > 0) {
    task_end_ms <- TASK_ONSET_MS + rt_ms
  } else {
    task_end_ms <- Inf
  }

  init_mask <- samples_trial$time_in_trial >= INIT_START_MS &
    samples_trial$time_in_trial <  INIT_END_MS &
    !is.na(samples_trial$x_smooth)

  task_mask <- samples_trial$time_in_trial >= TASK_ONSET_MS &
    samples_trial$time_in_trial <= task_end_ms &
    !is.na(samples_trial$x_smooth)

  init_all_mask <- samples_trial$time_in_trial >= INIT_START_MS &
    samples_trial$time_in_trial <  INIT_END_MS
  task_all_mask <- samples_trial$time_in_trial >= TASK_ONSET_MS &
    samples_trial$time_in_trial <= task_end_ms

  # Init window is fixed at 800ms
  Init_total_time_ms <- INIT_END_MS - INIT_START_MS
  Init_total_samples <- sum(init_all_mask)
  Init_valid_samples <- sum(init_mask)
  Init_trackloss_prop <- if (Init_total_samples > 0) {
    1 - Init_valid_samples / Init_total_samples
  } else NA_real_

  r_init <- rle(init_mask)
  Init_n_fix <- sum(r_init$values)

  dx <- diff(samples_trial$x_smooth)
  dy <- diff(samples_trial$y_smooth)
  pair_mask <- head(init_mask, -1) & tail(init_mask, -1)
  sacc_amp  <- sqrt(dx^2 + dy^2)
  # Clean physically impossible large amplitude jumps
  sacc_amp[sacc_amp > MAX_SACCADE_AMPLITUDE_PX] <- NA_real_
  Init_mean_sacc_amp <- if (any(pair_mask)) {
    mean(sacc_amp[pair_mask], na.rm = TRUE)
  } else NA_real_
  if (!is.na(Init_mean_sacc_amp) && Init_mean_sacc_amp > MAX_SACCADE_AMPLITUDE_PX) {
    Init_mean_sacc_amp <- NA_real_
  }

  non_catcher_mask <- task_mask &
    !(samples_trial$ROI_static %in% c("catcher_left", "catcher_right"))

  GS_time_ms <- sum(non_catcher_mask) * SAMPLE_DT_MS

  eye_time_ms   <- sum(task_mask) * SAMPLE_DT_MS

  inter_time_ms <- sum(task_mask & samples_trial$ROI_PT) * SAMPLE_DT_MS
  if (nrow(true_traj_df) > 1) {
    traj_duration_ms <- (max(true_traj_df$t) - min(true_traj_df$t)) * 1000
  } else {
    traj_duration_ms <- NA_real_
  }
  union_ms <- if (!is.na(traj_duration_ms)) {
    eye_time_ms + traj_duration_ms - inter_time_ms
  } else NA_real_
  I_over_U <- if (!is.na(union_ms) && union_ms > 0) inter_time_ms / union_ms else NA_real_

  track_mask <- task_mask & samples_trial$ROI_PT
  r_track <- rle(track_mask)
  max_run <- if (any(r_track$values)) max(r_track$lengths[r_track$values]) else 0
  Track_ms <- max_run * SAMPLE_DT_MS

  Task_total_samples <- sum(task_all_mask)
  Task_valid_samples <- sum(task_mask)
  Task_trackloss_prop <- if (Task_total_samples > 0) {
    1 - Task_valid_samples / Task_total_samples
  } else NA_real_

  if (!is.na(Init_trackloss_prop) && Init_trackloss_prop > INIT_MAX_TRACKLOSS_PROP) {
    Init_total_time_ms <- NA_real_
    Init_n_fix <- NA_integer_
    Init_mean_sacc_amp <- NA_real_
  }

  if (!is.na(Task_trackloss_prop) && Task_trackloss_prop > TASK_MAX_TRACKLOSS_PROP) {
    GS_time_ms <- NA_real_
    I_over_U <- NA_real_
    Track_ms <- NA_real_
  }

  data.frame(
    trial_id            = tid,
    Init_total_time_ms  = Init_total_time_ms,
    Init_n_fix          = Init_n_fix,
    Init_mean_sacc_amp  = Init_mean_sacc_amp,
    GS_time_ms          = GS_time_ms,
    I_over_U            = I_over_U,
    Track_ms            = Track_ms,
    Init_trackloss_prop = Init_trackloss_prop,
    Task_trackloss_prop = Task_trackloss_prop
  )
}

prepare_behavior_trials <- function(behav) {
  if (!("HV_primary" %in% names(behav)) && ("Heuristic_Type" %in% names(behav))) {
    behav$HV_primary <- behav$Heuristic_Type
  }
  if (!("SU_level" %in% names(behav)) && ("SU_Level" %in% names(behav))) {
    behav$SU_level <- behav$SU_Level
  }
  if (!("RT" %in% names(behav)) && all(c("Response_time", "Stim_onset") %in% names(behav))) {
    behav$RT <- (as.numeric(behav$Response_time) - as.numeric(behav$Stim_onset)) * 1000
  }

  behav$RT <- suppressWarnings(as.numeric(behav$RT))
  behav$TrialID <- suppressWarnings(as.integer(behav$TrialID))
  behav$BlockID <- suppressWarnings(as.integer(behav$BlockID))
  behav$SU_level <- suppressWarnings(as.integer(behav$SU_level))
  behav$Correct <- suppressWarnings(as.numeric(behav$Correct))
  behav$HV_primary <- trimws(as.character(behav$HV_primary))

  response_complete <- rep(TRUE, nrow(behav))
  if ("Choice" %in% names(behav)) {
    response_complete <- response_complete & !is.na(behav$Choice) & trimws(as.character(behav$Choice)) != ""
  }
  if ("Correct" %in% names(behav)) {
    response_complete <- response_complete & !is.na(behav$Correct)
  }
  if ("RT" %in% names(behav)) {
    response_complete <- response_complete & !is.na(behav$RT)
  }

  keep <- response_complete & !is.na(behav$RT) & behav$RT >= 200
  behav <- behav[keep, , drop = FALSE]
  if (nrow(behav) == 0) {
    return(behav)
  }

  rt_mean <- mean(behav$RT, na.rm = TRUE)
  rt_sd <- stats::sd(behav$RT, na.rm = TRUE)
  if (!is.na(rt_sd) && rt_sd > 0) {
    behav <- behav[behav$RT <= (rt_mean + 3 * rt_sd), , drop = FALSE]
  }
  if (nrow(behav) == 0) {
    return(behav)
  }

  split_key <- interaction(behav$HV_primary, behav$SU_level, drop = TRUE, lex.order = TRUE)
  groups <- split(behav, split_key)
  trimmed <- lapply(groups, function(grp) {
    hv <- tolower(trimws(as.character(grp$HV_primary[1])))
    su <- suppressWarnings(as.integer(grp$SU_level[1]))
    target_n <- NA_integer_
    if (hv == "conflict") {
      target_n <- 100L
    } else if (hv %in% c("congruent", "ambiguous") && !is.na(su) && su %in% 1:5) {
      target_n <- 25L
    }
    grp <- grp[order(grp$TrialID, grp$BlockID, na.last = TRUE), , drop = FALSE]
    if (is.na(target_n) || nrow(grp) <= target_n) {
      return(grp)
    }
    idx <- unique(round(seq(1, nrow(grp), length.out = target_n)))
    while (length(idx) < target_n) {
      idx <- unique(c(idx, max(idx) + 1L))
      idx <- idx[idx <= nrow(grp)]
    }
    grp[idx[seq_len(target_n)], , drop = FALSE]
  })

  dplyr::bind_rows(trimmed)
}

# ---------- Main function: single-subject gaze analysis ----------

run_gaze_analysis_for_subject <- function(
    asc_path,
    behav_csv_path = NULL,
    stim_config_path
) {
  asc <- read_asc_raw(asc_path)
  trial_bounds <- get_trial_bounds(asc$msg)

  samples_gaze <- preprocess_gaze(asc$samples, asc$eblink, trial_bounds)

  # Merge behavioral data if provided; drop conflict trials; keep RT for task window
  if (!is.null(behav_csv_path) && file.exists(behav_csv_path)) {
    behav <- read.csv(behav_csv_path, stringsAsFactors = FALSE)
    behav <- prepare_behavior_trials(behav)
    behav_small <- behav %>%
      dplyr::select(TrialID, BlockID, BoardID,
                    SU_level, HV_primary, HV_secondary_count,
                    HV_secondary_collision, PhysOutcome, RT)

    samples_joined <- samples_gaze %>%
      left_join(behav_small, by = c("trial_id" = "TrialID")) %>%
      mutate(HV_primary = trimws(HV_primary)) %>%
      dplyr::filter(!is.na(BoardID), HV_primary != "conflict")
  } else {
    samples_joined <- samples_gaze
    samples_joined$BoardID <- NA_character_
  }

  stim_cfg <- load_stim_config(stim_config_path)

  # Assign ROIs for all trials
  split_trials <- split(samples_joined, samples_joined$trial_id)
  roi_list <- lapply(split_trials, function(df_trial) {
    bid <- unique(df_trial$BoardID)
    row_board <- stim_cfg %>% dplyr::filter(BoardID == bid)
    if (nrow(row_board) != 1) {
      stop(paste("BoardID not found or not unique:", bid))
    }
    assign_rois_for_trial(df_trial, row_board)
  })
  samples_with_roi <- do.call(rbind, roi_list)
  rownames(samples_with_roi) <- NULL

  # Compute trial-level eye metrics
  split_roi_trials <- split(samples_with_roi, samples_with_roi$trial_id)
  eye_list <- lapply(split_roi_trials, function(df_trial) {
    bid <- unique(df_trial$BoardID)
    row_board <- stim_cfg %>% dplyr::filter(BoardID == bid)
    true_traj <- parse_true_traj(row_board$True_trajectory)
    compute_eye_metrics_for_trial(df_trial, true_traj)
  })
  eye_metrics <- do.call(rbind, eye_list)
  rownames(eye_metrics) <- NULL

  # Compute ROI metrics and join
  roi_list <- lapply(split_roi_trials, compute_roi_metrics_for_trial)
  roi_metrics <- do.call(rbind, roi_list)
  rownames(roi_metrics) <- NULL
  eye_metrics <- eye_metrics %>%
    left_join(roi_metrics, by = "trial_id")

  # Merge behavioral data to eye metrics
  if (!is.null(behav_csv_path) && file.exists(behav_csv_path)) {
    result <- eye_metrics %>%
      left_join(behav_small, by = c("trial_id" = "TrialID"))
  } else {
    result <- eye_metrics
  }

  list(
    trial_eye_metrics = result,
    samples_with_roi  = samples_with_roi
  )
}

# ============================================================
# Batch processing: run on all subjects
# ============================================================

if (interactive() || !exists("SKIP_AUTO_RUN")) {

  cat("\n", strrep("=", 60), "\n", sep = "")
  cat("Starting batch eye-tracking analysis for all subjects...\n")
  cat(strrep("=", 60), "\n\n")

  # ---------- 1. Path configuration ----------

  script_dir <- tryCatch(
    dirname(normalizePath(sys.frame(1)$ofile)),
    error = function(e) getwd()
  )
  project_root <- dirname(script_dir)

  eye_dir    <- file.path(project_root, "data", "raw", "eye_tracking")
  behav_dir  <- file.path(project_root, "data", "raw", "behavior")
  stim_file  <- file.path(project_root, "data", "stimuli", "stimulus_config.csv")
  output_dir <- file.path(project_root, "data", "processed")

  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

  cat("Project root:", project_root, "\n")
  cat("Eye data dir:", eye_dir, "\n")
  cat("Behavior dir:", behav_dir, "\n")
  cat("Stimulus config:", stim_file, "\n")
  cat("Config exists:", file.exists(stim_file), "\n")

  if (!file.exists(stim_file)) {
    stop("Cannot find stimulus_config.csv: ", stim_file)
  }

  # Get all X-prefixed .asc files in eye_tracking folder
  asc_files <- list.files(eye_dir, pattern = "^X[0-9]+\\.asc$", full.names = FALSE)
  if (length(asc_files) == 0) {
    stop("No X-prefixed .asc files found in eye_tracking folder")
  }

  # Extract subject numbers (X01.asc -> 01)
  subject_nums <- gsub("^X([0-9]+)\\.asc$", "\\1", asc_files)
  subjects <- paste0("X", subject_nums)

  # Corresponding behavioral files (strip leading zeros: 01 -> 1.csv)
  behav_files <- paste0(as.integer(subject_nums), ".csv")

  all_eye_metrics   <- list()
  all_samples_gaze  <- list()

  for (i in seq_along(subjects)) {
    subj <- subjects[i]
    asc_file   <- file.path(eye_dir, paste0(subj, ".asc"))
    behav_file <- file.path(behav_dir, behav_files[i])

    cat("\n----------\n")
    cat("Subject:", subj, "\n")

    if (!file.exists(asc_file)) {
      cat("  Skipping: ASC file not found:", asc_file, "\n")
      next
    }
    if (!file.exists(behav_file)) {
      cat("  Skipping: Behavioral file not found:", behav_file, "\n")
      next
    }

    cat("  ASC file:", asc_file, "\n")
    cat("  Behavioral file:", behav_file, "\n")
    cat("  Processing...\n")

    tryCatch({
      res <- run_gaze_analysis_for_subject(
        asc_path = asc_file,
        behav_csv_path = behav_file,
        stim_config_path = stim_file
      )

      res$trial_eye_metrics$subject_id <- subj
      res$samples_with_roi$subject_id  <- subj

      all_eye_metrics[[subj]]  <- res$trial_eye_metrics
      all_samples_gaze[[subj]] <- res$samples_with_roi

      cat("  Done, subject", subj, ":", nrow(res$trial_eye_metrics), "trials\n")
    }, error = function(e) {
      cat("  Error: subject", subj, "failed -", e$message, "\n")
    })
  }

  if (length(all_eye_metrics) == 0) {
    stop("No subjects were successfully analyzed. Please check paths.")
  }

  # Combine and save trial-level metrics
  eye_metrics_all <- do.call(rbind, all_eye_metrics)
  rownames(eye_metrics_all) <- NULL
  metrics_file <- file.path(output_dir, "eye_metrics_all_subjects.csv")
  write.csv(eye_metrics_all, metrics_file, row.names = FALSE, fileEncoding = "UTF-8")
  cat("\nSaved:", metrics_file, "\n")

  # Aggregate to subject x HV x SU means
  eye_metrics_subject_means <- eye_metrics_all %>%
    mutate(
      HV_primary = trimws(HV_primary),
      SU_level   = as.integer(SU_level)
    ) %>%
    filter(HV_primary %in% c("congruent", "ambiguous"), !is.na(SU_level)) %>%
    group_by(subject_id, HV_primary, SU_level) %>%
    summarise(
      n_trials          = n(),
      I_over_U          = mean(I_over_U, na.rm = TRUE),
      Prop_PT           = mean(Prop_PT, na.rm = TRUE),
      Prop_CU           = mean(Prop_CU, na.rm = TRUE),
      Prop_catcher      = mean(Prop_catcher, na.rm = TRUE),
      Duration_PT       = mean(Duration_PT, na.rm = TRUE),
      Duration_CU       = mean(Duration_CU, na.rm = TRUE),
      Duration_catcher  = mean(Duration_catcher, na.rm = TRUE),
      Fixations_PT      = mean(Fixations_PT, na.rm = TRUE),
      Fixations_CU      = mean(Fixations_CU, na.rm = TRUE),
      Fixations_catcher = mean(Fixations_catcher, na.rm = TRUE),
      Track_ms          = mean(Track_ms, na.rm = TRUE),
      CU_visits         = mean(CU_visits, na.rm = TRUE),
      PT_entries        = mean(PT_entries, na.rm = TRUE),
      .groups = "drop"
    )
  subject_means_file <- file.path(output_dir, "eye_metrics_subject_means.csv")
  write.csv(eye_metrics_subject_means, subject_means_file, row.names = FALSE, fileEncoding = "UTF-8")
  cat("Saved:", subject_means_file, "\n")

  # Combine sample-level data and save as RData
  samples_gaze_all <- do.call(rbind, all_samples_gaze)
  rownames(samples_gaze_all) <- NULL
  rdata_file <- file.path(output_dir, "gaze_analysis_all_subjects.RData")
  save(eye_metrics_all, samples_gaze_all, file = rdata_file)
  cat("Saved:", rdata_file, "\n\n")

  cat("Eye metrics summary (all subjects combined):\n")
  print(summary(eye_metrics_all[, c("Init_total_time_ms", "GS_time_ms", "I_over_U", "Track_ms")]))

  cat("\n", strrep("=", 60), "\n", sep = "")
  cat("Batch eye-tracking analysis complete! Results saved to:", output_dir, "\n")
  cat(strrep("=", 60), "\n")
}
