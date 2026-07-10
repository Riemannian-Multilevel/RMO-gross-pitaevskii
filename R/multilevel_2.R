library(ggplot2)
library(stringr)
library(tikzDevice)

# 1. Updated reading function to handle missing columns gracefully
read_org_table <- function(filename, coarse) {
  if (!file.exists(filename)) return(NULL)
  lines <- readLines(filename)
  data_lines <- lines[grepl("\\|", lines) & !grepl("---", lines)]
  if(length(data_lines) == 0) return(NULL)
  
  df <- read.csv(text = data_lines, sep = "|", strip.white = TRUE, check.names = FALSE)

  if (coarse) {
    return(df[, c("elapsed", "residual", "energy", "coarse")])
  } else {
    return(df[, c("elapsed", "residual", "energy")])
  }
}

# 2. Identify unique prefixes
all_ml_files <- list.files(pattern = "ml_.*_l11.*.org")
#all_ml_files <- list.files(pattern = "ml_.*_l10.*.org")
prefixes <- unique(str_remove(all_ml_files, "_depth[234].org"))
#prefixes <- unique(str_remove(all_ml_files, "_depth[23].org"))

# 3. Loop
# TODO: energy varies with beta, level
#ref_energy_b100_l10   <- 2.8960505820276339e+00
#ref_energy_b1000_l10  <- 8.5118519924817804e+00
#ref_energy_b10000_l10 <- 2.6637150551653658e+01

ref_energy_b100_l11   <- 2.8960365347241521e+00
ref_energy_b1000_l11  <- 8.5118466265830470e+00
ref_energy_b10000_l11 <- 2.6637148254725020e+01

for (pref in prefixes) {

  # Load data
  d2 <- read_org_table(paste0(pref, "_depth2.org"), TRUE)
  if(!is.null(d2)) {
    d2$type   <- "Depth2"
    d2$energy <- d2$energy - ref_energy
  }
  d3 <- read_org_table(paste0(pref, "_depth3.org"), TRUE)
  if(!is.null(d3)) {
    d3$type   <- "Depth3"
    d3$energy <- d3$energy - ref_energy
  }
  d4 <- read_org_table(paste0(pref, "_depth4.org"), TRUE)
  if(!is.null(d4)) {
    d4$type   <- "Depth4"
    d4$energy <- d4$energy - ref_energy
  }
  
  # Load SL file
  f_sl <- paste0("sl_", str_extract(pref, "b[0-9]+"), "_l11.org")
  #f_sl <- paste0("sl_", str_extract(pref, "b[0-9]+"), "_l10.org")
  d_sl <- read_org_table(f_sl, FALSE)
  if(!is.null(d_sl)) {
    d_sl$type <- "SL"
    d_sl$coarse <- NA
    d_sl$energy <- d_sl$energy - ref_energy
  }
  
  # Merge all (now all have a 'coarse' column)
  #plot_df <- rbind(d2, d_sl)
  #plot_df <- rbind(d2, d3, d_sl)
  plot_df <- rbind(d2, d3, d4, d_sl)
  
  if (!is.null(plot_df) && nrow(plot_df) > 0) {
    plot_df <- plot_df[plot_df$elapsed <= 50, ]
    
    # Filter only for the asterisk markers
    marker_df <- plot_df[!is.na(plot_df$coarse) & plot_df$coarse == "*", ]
    
    if (nrow(plot_df) > 0) {
      pref_tex <- gsub("_", "\\_", pref, fixed = TRUE)
      
      p <- ggplot(plot_df, aes(x = elapsed, y = residual, color = type, linetype = type)) +
        geom_line(linewidth = 1) +
        # Add points only where coarse == "*"
        geom_point(data = marker_df, aes(x = elapsed, y = residual), 
                   inherit.aes = FALSE, color = "black", shape = 8, size = 3) +
        scale_y_log10() +
        labs(title = paste("Comparison:", pref_tex), x = "Elapsed Time (s)", y = "Residual (log)") +
        theme_minimal()
      
      #ggsave(filename = paste0("plot_", pref, ".png"), plot = p)
      out_file <- paste0("plot_", pref, ".tex")
      tikz(out_file, width = 6, height = 4)
      print(p)
      dev.off()
    }
  }
}

