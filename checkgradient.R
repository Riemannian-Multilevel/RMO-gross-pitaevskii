# Install required packages if you don't have them:
# install.packages(c("ggplot2", "dplyr", "readr", "purrr", "scales"))

library(ggplot2)
library(dplyr)
library(purrr)
library(scales)

# 1. Read all trial files matching the pattern
file_pattern <- "^checkgradient_coarse_energy_2d_[0-9]{3}\\.dat$"
file_list <- list.files(pattern = file_pattern)

if(length(file_list) == 0) {
  stop("No files found matching the pattern.")
}

# Read and combine all data into one dataframe
# Assuming the files have no headers, we name the columns 't' and 'Et'
df <- map_df(file_list, ~read.table(.x, col.names = c("t", "Et")))

# 2. Compute Mean and Standard Deviation for each 't'
df_summary <- df %>%
  group_by(t) %>%
  summarise(
    mean_Et = mean(Et),
    sd_Et = sd(Et),
    .groups = "drop"
  ) %>%
  # Handle log-scale lower bounds: if mean - sd is <= 0, the log plot will break.
  # We clamp the lower bound to a very small number to prevent errors.
  mutate(
    ymin = pmax(mean_Et - sd_Et, 1e-20),
    ymax = mean_Et + sd_Et
  )

# 3. Create the ggplot
p <- ggplot(df_summary, aes(x = t)) +
  # Shaded region for standard deviation
  geom_ribbon(aes(ymin = ymin, ymax = ymax), fill = "blue", alpha = 0.2) +
  
  # Mean line
  geom_line(aes(y = mean_Et), color = "blue", linewidth = 1) +
  
  # Reference line: y = 1e8 * t^2
  geom_function(fun = function(x) 1e8 * x^2, linetype = "dashed", color = "gray50", linewidth = 1) +
  
  # Annotation for the slope
  annotate("text", x = 1e-8, y = 2, label = "Slope = 2", color = "gray50", hjust = 0, size = 4) +
  
  # Log-Log scaling with 10^x mathematical formatting
  scale_x_log10(
    breaks = trans_breaks("log10", function(x) 10^x),
    labels = trans_format("log10", math_format(10^.x))
  ) +
  scale_y_log10(
    breaks = trans_breaks("log10", function(x) 10^x),
    labels = trans_format("log10", math_format(10^.x))
  ) +
  
  # Labels and Title
  labs(
    title = "Log-Log Plot of t vs E(t)",
    x = "t",
    y = "E(t)",
    caption = paste("Averaged over", length(file_list), "trials")
  ) +
  
  # Grid lines and overall theme (mimicking Gnuplot's clean grids)
  theme_bw() +
  theme(
    panel.grid.minor = element_line(color = "grey90"),
    panel.grid.major = element_line(color = "grey80"),
    plot.title = element_text(hjust = 0.5, face = "bold")
  )

# Display the plot
print(p)

# Save the plot to a high-res PDF or PNG
# ggsave("taylor_expansion_plot.pdf", plot = p, width = 8, height = 6)