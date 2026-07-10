library(ggplot2)
library(stringr)
library(patchwork)
library(tikzDevice)

# ── Reference energies ────────────────────────────────────────────────────────
ref_energies <- c(
  b100   = 2.8960365347241521e+00,
  b1000  = 8.5118466265830470e+00,
  b10000 = 2.6637148254725020e+01
)

# ── Colors matching Julia (cpl6, cpl4, cpl2, unia-purple) ────────────────────
# Requires in LaTeX preamble:
#   \definecolor{unia-purple}{RGB}
#   \definecolor{cpl1}{rgb}{0.8889,0.4356,0.2781}
#   \definecolor{cpl2}{rgb}{0.0,0.6056,0.9787}
#   \definecolor{cpl4}{rgb}{0.2422,0.6433,0.3044}
#   \definecolor{cpl6}{rgb}{0.2, 0.2, 0.2}
type_colors <- c(
  SL     = rgb(0.2,    0.2,    0.2   ),   # cpl6
  Depth2 = rgb(0.2422, 0.6433, 0.3044),   # cpl4
  Depth3 = rgb(0.0,    0.6056, 0.9787),   # cpl2
  Depth4 = rgb(173/255, 0,     124/255),  # unia-purple
  Depth5 = rgb(0.8706, 0.5608, 0.0196),   # cpl1
  Depth6 = rgb(0.7529, 0.1137, 0.1686)    # cpl3 or cpl5
  
)
type_shapes <- c(SL = 4, Depth2 = 18, Depth3 = 17, Depth4 = 15, Depth5 = 16, Depth6 = 25)
#                   x       diamond*      triangle*     square*

# ── Theme matching Julia axis style ──────────────────────────────────────────
julia_theme <- theme_bw(base_size = 8) +
  theme(
    legend.position    = "top",
    legend.direction   = "horizontal",
    legend.title       = element_blank(),
    legend.key.width   = unit(1.5, "cm"),
    panel.grid.major   = element_line(colour = rgb(0, 0, 0, 0.1), linewidth = 0.5),
    panel.grid.minor   = element_blank(),
    axis.ticks         = element_line(colour = "black"),
    axis.ticks.length  = unit(-0.15, "cm"),   # inside ticks
    axis.text          = element_text(size = 8, colour = "black"),
    axis.title         = element_text(size = 8, colour = "black"),
    panel.border       = element_rect(colour = "black", fill = NA)
  )

# ── Data loading ──────────────────────────────────────────────────────────────
read_org_table <- function(filename, coarse) {
  if (!file.exists(filename)) return(NULL)
  lines      <- readLines(filename)
  data_lines <- lines[grepl("\\|", lines) & !grepl("---", lines)]
  if (length(data_lines) == 0) return(NULL)
  df   <- read.csv(text = data_lines, sep = "|", strip.white = TRUE, check.names = FALSE)
  cols <- if (coarse) c("elapsed", "residual", "energy", "coarse") else c("elapsed", "residual", "energy")
  df[, cols]
}

load_depth <- function(filename, type_label, has_coarse, ref_energy) {
  d <- read_org_table(filename, has_coarse)
  if (is.null(d)) return(NULL)
  d$type        <- type_label
  d$iter        <- seq_len(nrow(d)) - 1L
  d$energy_diff <- abs(d$energy - ref_energy)
  if (!has_coarse) d$coarse <- NA
  d
}

# ── Panel builder ─────────────────────────────────────────────────────────────
make_panel <- function(df, marker_df, x_var, y_var, x_lab, y_lab) {
  df        <- df[!is.na(df[[y_var]]) & df[[y_var]] > 0, ]
  marker_df <- marker_df[!is.na(marker_df[[y_var]]) & marker_df[[y_var]] > 0, ]
  
  ggplot(df, aes(x = .data[[x_var]], y = .data[[y_var]],
                 color = type, linetype = type, shape = type)) +
    geom_line(linewidth = 0.8, show.legend = TRUE) +
    geom_point(data = marker_df,
               aes(x = .data[[x_var]], y = .data[[y_var]]),
               size = 2.5, show.legend = TRUE) +
    scale_y_log10() +
    scale_color_manual(values = type_colors) +
    scale_shape_manual(values = type_shapes) +
    scale_linetype_manual(values = c(SL = "dashed",
                                     Depth2 = "solid", Depth3 = "solid", Depth4 = "solid", 
                                     Depth5 = "solid", Depth6 = "solid")) +
    guides(color = guide_legend(override.aes = list(linewidth = 0.8))) +
    labs(x = x_lab, y = y_lab) +
    julia_theme
}

# ── Main loop ─────────────────────────────────────────────────────────────────
all_ml_files <- list.files(pattern = "ml_.*_l11.*.org")
#prefixes     <- unique(str_remove(all_ml_files, "_depth[23456].org"))
prefixes     <- unique(str_remove(all_ml_files, "_depth[234].org"))

for (pref in prefixes) {
  b_key      <- str_extract(pref, "b[0-9]+")
  ref_energy <- ref_energies[b_key]
  
  d2   <- load_depth(paste0(pref, "_depth2.org"), "Depth2", TRUE, ref_energy)
  d3   <- load_depth(paste0(pref, "_depth3.org"), "Depth3", TRUE, ref_energy)
  d4   <- load_depth(paste0(pref, "_depth4.org"), "Depth4", TRUE, ref_energy)
  #d5   <- load_depth(paste0(pref, "_depth5.org"), "Depth5", TRUE, ref_energy)
  #d6   <- load_depth(paste0(pref, "_depth6.org"), "Depth6", TRUE, ref_energy)
  d3_mix <- load_depth(paste)
  d_sl <- load_depth(paste0("sl_", b_key, "_l11.org"), "SL", FALSE, ref_energy)
  
  #plot_df <- rbind(d_sl, d2, d3, d4, d5, d6)
  plot_df <- rbind(d_sl, d2, d3, d4)
  if (is.null(plot_df) || nrow(plot_df) == 0) next
  
  time_df  <- plot_df[plot_df$elapsed <= 100, ]
  iter_mkr <- plot_df[!is.na(plot_df$coarse) & plot_df$coarse == "*", ]
  time_mkr <- time_df[!is.na(time_df$coarse) & time_df$coarse  == "*", ]
  
  e_lab <- "$|E(\\Phi_k) - E_{\\mathrm{ref}}|$"
  
  no_legend <- theme(legend.position = "none")
  
  # Plot 1: iteration-based
  p1 <- make_panel(plot_df, iter_mkr, "iter", "energy_diff", "iterations", e_lab) |
    make_panel(plot_df, iter_mkr, "iter", "residual",    "iterations", "$\\|R_k\\|$") + no_legend
  
  ggsave(filename = paste0("plot_iter_", pref, ".png"), plot = p1)
  #out1 <- paste0("plot_iter_", pref, ".tex")
  #tikz(out1, width = 5.8, height = 3.2)
  #print(p1)
  #dev.off()
  
  # Plot 2: time-based
  p2 <- make_panel(time_df, time_mkr, "elapsed", "energy_diff", "CPU time (sec)", e_lab) |
    make_panel(time_df, time_mkr, "elapsed", "residual",    "CPU time (sec)", "$\\|R_k\\|$") + no_legend
  
  ggsave(filename = paste0("plot_time_", pref, ".png"), plot = p2)
  #out2 <- paste0("plot_time_", pref, ".tex")
  #tikz(out2, width = 5.8, height = 3.2)
  #print(p2)
  #dev.off()
}