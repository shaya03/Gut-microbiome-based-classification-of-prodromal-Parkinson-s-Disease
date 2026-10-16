#B2. Exporting feature table from .biom file (in R) 
library(biomformat) 
save_biom_to_csv <- function(biom_path, output_path) { 
# Read the BIOM file 
biom_file <- read_biom(biom_path) 
# Convert to a data frame 
feature_table <- as.data.frame(as.matrix(biom_data(biom_file))) 
# Rename the first column to "Taxa"  
feature_table <- cbind(Taxa = rownames(feature_table), 
feature_table) 
rownames(feature_table) <- NULL 
# Write to CSV 
write.csv(feature_table, output_path, row.names = FALSE) 
cat("Feature table saved to:", output_path, "\n") }
