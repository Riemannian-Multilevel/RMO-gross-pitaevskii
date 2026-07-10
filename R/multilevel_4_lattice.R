library(stringr)

suffix     <- "_optical_lattice.org"
tex_suffix <- tools::file_path_sans_ext(suffix)  # "_optical_lattice"

# ── Reference energies ────────────────────────────────────────────────────────
ref_energies <- c(
  #b100   = 2.0442403664073602e+01,
  b1000  = 3.0354706131252627e+01
  #b10000 = 5.4388786473251827e+01
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

# ── Write data rows: "x y \\" per line ───────────────────────────────────────
write_data <- function(d, x_var, y_var, output_file,
                       time_cap = NULL, coarse_only = FALSE) {
  if (is.null(d)) return(invisible(NULL))
  if (!is.null(time_cap))  d <- d[d$elapsed <= time_cap, ]
  if (coarse_only)         d <- d[!is.na(d$coarse) & d$coarse == "*", ]
  xs    <- d[[x_var]]
  ys    <- d[[y_var]]
  valid <- !is.na(ys) & ys > 0
  if (!any(valid)) { message("Skipped (no data): ", output_file); return(invisible(NULL)) }
  writeLines(sprintf("%g %g \\\\", xs[valid], ys[valid]), output_file)
  message("Wrote: ", output_file)
}

# ── SL baselines (once per b_key) ────────────────────────────────────────────
for (b_key in names(ref_energies)) {
  ref_energy <- ref_energies[b_key]
  d_sl       <- load_depth(paste0("sl_", b_key, "_l11", suffix), "SL", FALSE, ref_energy)
  sl_stem    <- sprintf("data_sl_%s_l11", b_key)
  write_data(d_sl, "iter",    "residual",    sprintf("%s_iter_residual%s.tex",    sl_stem, tex_suffix))
  write_data(d_sl, "iter",    "energy_diff", sprintf("%s_iter_energy%s.tex",      sl_stem, tex_suffix))
  write_data(d_sl, "elapsed", "residual",    sprintf("%s_elapsed_residual%s.tex", sl_stem, tex_suffix))
  write_data(d_sl, "elapsed", "energy_diff", sprintf("%s_elapsed_energy%s.tex",   sl_stem, tex_suffix))
}

# ── ML methods: all methods × all depths × all x/y combinations ──────────────
suffix_pat      <- gsub(".", "\\.", suffix, fixed = TRUE)
all_depth_files <- list.files(pattern = paste0("^ml_.*_l11_depth[2-6]", suffix_pat, "$"))
method_groups   <- unique(str_extract(all_depth_files, "(?<=ml_).+(?=_l11_depth)"))

for (mg in method_groups) {
  b_key      <- str_extract(mg, "b[0-9]+")
  ref_energy <- ref_energies[b_key]
  base       <- paste0("ml_", mg, "_l11")
  
  depths <- list(
    depth2 = load_depth(paste0(base, "_depth2", suffix), "depth2", TRUE, ref_energy),
    depth3 = load_depth(paste0(base, "_depth3", suffix), "depth3", TRUE, ref_energy),
    depth4 = load_depth(paste0(base, "_depth4", suffix), "depth4", TRUE, ref_energy)
    #depth5 = load_depth(paste0(base, "_depth5", suffix), "depth5", TRUE, ref_energy),
    #depth6 = load_depth(paste0(base, "_depth6", suffix), "depth6", TRUE, ref_energy)
  )
  
  for (dep in names(depths)) {
    d <- depths[[dep]]
    if (is.null(d)) next
    stem <- sprintf("data_%s_%s", base, dep)
    
    write_data(d, "iter",    "residual",    sprintf("%s_iter_residual%s.tex",           stem, tex_suffix))
    write_data(d, "iter",    "residual",    sprintf("%s_iter_residual_coarse%s.tex",    stem, tex_suffix), coarse_only = TRUE)
    write_data(d, "iter",    "energy_diff", sprintf("%s_iter_energy%s.tex",             stem, tex_suffix))
    write_data(d, "iter",    "energy_diff", sprintf("%s_iter_energy_coarse%s.tex",      stem, tex_suffix), coarse_only = TRUE)
    write_data(d, "elapsed", "residual",    sprintf("%s_elapsed_residual%s.tex",        stem, tex_suffix))
    write_data(d, "elapsed", "residual",    sprintf("%s_elapsed_residual_coarse%s.tex", stem, tex_suffix), coarse_only = TRUE)
    write_data(d, "elapsed", "energy_diff", sprintf("%s_elapsed_energy%s.tex",          stem, tex_suffix))
    write_data(d, "elapsed", "energy_diff", sprintf("%s_elapsed_energy_coarse%s.tex",   stem, tex_suffix), coarse_only = TRUE)
  }
}