library(ggplot2)
library(stringr)
library(dplyr)
library(tikzDevice)

# 1. Reading function (retained for data extraction)
read_org_table <- function(filename) {
  if (!file.exists(filename)) return(NULL)
  
  # 1. Read lines
  lines <- readLines(filename)
  # Keep lines with data (containing '|')
  data_lines <- lines[grepl("\\|", lines) & !grepl("---", lines)]
  if(length(data_lines) == 0) return(NULL)
  
  # 2. Use read.table with pipe as separator
  # We set fill=TRUE to handle missing values in the 'coarse' column
  df <- read.table(text = data_lines, header = TRUE, sep = "|", 
                   strip.white = TRUE, check.names = FALSE, fill = TRUE)
  
  # 3. Clean up the dataframe: Remove columns that are completely empty
  df <- df[, !apply(df, 2, function(x) all(is.na(x) | x == ""))]
  
  # 4. Standardize headers
  names(df) <- tolower(names(df))
  
  # Ensure coarse column exists, even if empty
  if(!"coarse" %in% names(df)) df$coarse <- ""
  df$coarse <- str_trim(as.character(df$coarse))
  
  # Return data
  return(na.omit(df[, c("elapsed", "residual", "coarse")]))
}

# 2. Identify all ML files
suffix <- "_optical_lattice.org"
#suffix <- ".org"
all_ml_files <- list.files(pattern = paste0("ml_.*_l11_depth[2345]", suffix))

# 3. Create group identifier based on scale and depth
# We want to group by 'b1000' and 'depth3' regardless of the method name
# Extract parts: ml_(METHOD)_(bNUMBER)_(l11)_(depthX).org
groups <- unique(str_extract(all_ml_files, "b[0-9]+_l11_depth[2345]"))

for (g in groups) {
  # Find all methods for this scale and depth
  matching_files <- all_ml_files[str_detect(all_ml_files, paste0("_", g, suffix))]
  
  # Initialize empty list to collect data
  plot_list <- list()
  
  for (f in matching_files) {
    # Extract method name from filename (e.g., 'frob_diff')
    method_name <- str_remove(str_remove(f, "ml_"), paste0("_", g, suffix))
    
    d <- read_org_table(f)
    if(!is.null(d) && nrow(d) > 0) {
      method_tex <- gsub("_", "\\_", method_name, fixed = TRUE)
      d$type <- method_tex
      plot_list[[f]] <- d
    }
  }
  
  # Add SL baseline
  b_val <- str_extract(g, "b[0-9]+")
  f_sl <- paste0("sl_", b_val, "_l11.org")
  d_sl <- read_org_table(f_sl)
  if(!is.null(d_sl) && nrow(d_sl) > 0) {
    d_sl$type <- "SL"
    plot_list[[f_sl]] <- d_sl
  }
  
  # Merge all methods for this scale/depth
  plot_df <- bind_rows(plot_list)
  
  if (!is.null(plot_df) && nrow(plot_df) > 0) {
    plot_df <- plot_df[plot_df$elapsed <= 30, ]
    marker_df <- plot_df[!is.na(plot_df$coarse) & plot_df$coarse == "*", ]
    
    method_names <- unique(plot_df$type[plot_df$type != "SL"])
    custom_colors <- setNames(rainbow(length(method_names)), method_names)
    custom_colors["SL"] <- "gray50" # Set SL to gray
    
    # Define a vector of 15 distinct shapes
    all_shapes <- c(15, 16, 17, 18, 3, 4, 7, 8, 9, 10, 11, 12, 13, 1, 2)
    g_tex <- gsub("_", "\\_", g, fixed = TRUE)
    
    p <- ggplot(plot_df, aes(x = elapsed, y = residual, color = type, linetype = type)) +
      geom_line(linewidth = 1) +
      # Use custom shape scale to handle > 6 categories
      geom_point(data = marker_df, aes(x = elapsed, y = residual, shape = type), 
                 inherit.aes = FALSE, size = 3) +
      scale_shape_manual(values = all_shapes) +
      scale_y_log10() +
      # Ensure SL is dashed/gray, others solid
      scale_color_manual(values = custom_colors) +
      scale_linetype_manual(values = c("SL" = "dashed", 
                                       setNames(rep("solid", length(method_names)), method_names))) +
      labs(title = paste("Comparison:", g_tex), 
           x = "Elapsed Time (s)", 
           y = "Residual (log)") +
      theme_minimal()
    
    suffix_png <- paste0(tools::file_path_sans_ext(suffix), ".png")
    ggsave(filename = paste0("plot_methods_", g, suffix_png), plot = p)
    #out_file <- paste0("plot_methods_", g, ".tex")
    #tikz(out_file, width = 6, height = 4)
    #print(p)
    #dev.off()
  }
}
