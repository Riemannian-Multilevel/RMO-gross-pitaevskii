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
df <- map_dfr(
  files,
  ~ read_csv(.x, show_col_types = FALSE),
  .id = "file_id"
) %>%
  mutate(
    file = factor(files[as.integer(file_id)]),
    iter = as.integer(iter)
  )

# ---- plot ----
ggplot(df, aes(x = iter, y = residual, color = file)) +
  geom_line(linewidth = 0.8) +
  geom_point(shape = 16, size = 2) +
  facet_wrap(~ file, ncol = length(files)) +
  labs(x = "iter", y = "residual", color = "File") +
  theme_minimal() + scale_y_log10()

