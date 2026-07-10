library(ggplot2)

read_org_table <- function(filename) {
  # 1. Read all lines
  lines <- readLines(filename)
  
  # 2. Keep only lines that contain a pipe (data rows) and are not separators
  data_lines <- lines[grepl("\\|", lines) & !grepl("---", lines)]
  
  # 3. Read the cleaned lines directly as a CSV with pipe as the separator
  df <- read.csv(text = data_lines, sep = "|", strip.white = TRUE, check.names = FALSE)
  
  # 4. Clean column names to ensure they match what you expect
  # Org-mode tables often have leading/trailing empty columns due to outer pipes
  df <- df[, !apply(df, 2, function(x) all(is.na(x) | x == ""))]
  
  # 5. Force numeric conversion
  df$elapsed <- as.numeric(as.character(df$elapsed))
  df$residual <- as.numeric(as.character(df$residual))
  
  # 6. Remove any rows that failed conversion (NAs)
  df <- df[!is.na(df$elapsed) & !is.na(df$residual), ]
  
  return(df)
}

# Load the files
sl_data <- read_org_table("sl_b1000.org")
adj1_data <- read_org_table("ml_mass_adj1_b1000.org")
depth3_data <- read_org_table("ml_mass_adj1_b1000_depth3.org")

# Assign types
sl_data$type <- "sl_b1000"
adj1_data$type <- "ml_mass_adj1_b1000"
depth3_data$type <- "ml_mass_adj1_b1000_depth3"

# Combine
df <- rbind(sl_data[, c("elapsed", "residual", "type")],
            adj1_data[, c("elapsed", "residual", "type")],
            depth3_data[, c("elapsed", "residual", "type")])

# LIMIT: Filter the dataframe to only include rows where elapsed time <= 15
df <- df[df$elapsed <= 15, ]

# Plot
ggplot(df, aes(x = elapsed, y = residual, color = type, linetype = type)) +
  geom_line(linewidth = 1) +
  scale_y_log10() +
  scale_color_manual(values = c("sl_b1000" = "black", 
                                "ml_mass_adj1_b1000" = "blue", 
                                "ml_mass_adj1_b1000_depth3" = "red")) +
  scale_linetype_manual(values = c("sl_b1000" = "dotted", 
                                   "ml_mass_adj1_b1000" = "solid", 
                                   "ml_mass_adj1_b1000_depth3" = "solid")) +
  labs(title = "Residual vs Elapsed Time (Limit: 15s)", 
       x = "Elapsed Time (s)", 
       y = "Residual (log scale)") +
  theme_minimal()