library(readr)
library(dplyr)
library(ggplot2)
library(purrr)

# ---- list of CSV files (any length n) ----
files <- c(
  "solve_ref_08.csv",
  "solve_ref_09.csv",
  "solve_ref_10.csv",
  "solve_ref_11.csv"
)

# ---- read all files and tag source ----
# df <- map_dfr(
#   files,
#   ~ read_csv(.x, show_col_types = FALSE),
#   .id = "file_id"
# ) %>%
#   mutate(
#     file = factor(files[as.integer(file_id)]),
#     iter = as.integer(iter)
#   )

read_dealii_convergence <- function(path) {
  lines <- readLines(path, warn = FALSE)
  
  keep <- grepl("^\\s*iter\\b", lines) | grepl("^\\s*\\d+\\b", lines)
  txt  <- paste(lines[keep], collapse = "\n")
  
  df <- readr::read_table(
    I(txt),
    col_names = TRUE,
    col_types = cols(.default = col_double()),
    na = c("-", "nan", "NaN")
  )
  
  # If ConvergenceTable prints the two rate columns without headers, read_table 
  # will create extra columns with placeholder names like `...6`, `...7`.
  # Rename to stable names.
  nms <- names(df)
  
  # Expected: iter lac_iter mass lambda residual [rate] [rate_log2] energy
  if (length(nms) == 8) {
    names(df) <- c("iter","lac_iter","mass","lambda","residual",
                   "residual_rate","residual_rate_log2","energy")
  }
  
  df
}

df <- map_dfr(files, read_dealii_convergence, .id = "file_id") %>%
  mutate(
    file = factor(files[as.integer(file_id)]),
    iter = as.integer(iter),
    lac_iter = as.integer(lac_iter)
  )

# ---- plot ----
ggplot(df, aes(x = iter, y = residual, color = file)) +
  geom_line(linewidth = 0.8) +
  geom_point(shape = 16, size = 2) +
  facet_wrap(~ file, ncol = length(files)) +
  labs(x = "iter", y = "residual", color = "File") +
  scale_y_log10() +
  theme_minimal() +
  theme(
    legend.position = c(0.98, 0.98),
    legend.justification = c("right", "top"),
    legend.background = element_rect(fill = scales::alpha("white", 0.8),
                                     color = NA),
    legend.key = element_blank()
  )

