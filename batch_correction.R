#B3. Batch Effect Correction (in R) 
# Load required libraries 
library(MBECS) 
library(phyloseq) 
# Set working directory 
setwd("path_to_working_directory”) 
# Load abundance data (Taxa/ASVs as rows, samples as columns) 
abund1 <- read.delim("dataset1.tsv", row.names = 1, check.names = 
FALSE) 
abund2 <- read.delim("dataset2.tsv", row.names = 1, check.names = 
FALSE) 
# Load corresponding metadata (Samples as rows) 
meta1 <- read.delim("meta1.tsv", row.names = 1) 
meta2 <- read.delim("meta2.tsv", row.names = 1) 
# Ensure columns and rownames align 
stopifnot(all(colnames(abund1) == rownames(meta1))) 
stopifnot(all(colnames(abund2) %in% rownames(meta2))) 
# Merge and harmonize taxa 
all_taxa <- unique(c(rownames(abund1_matched), rownames(abund2))) 
abund1_full <- abund1_matched[match(all_taxa, rownames(abund1)), , 
drop = FALSE] 
abund2_full <- abund2[match(all_taxa, rownames(abund2)), , drop = 
FALSE] 
# Fill missing taxa with zeros 
abund1_full[is.na(abund1_full)] <- 0 
abund2_full[is.na(abund2_full)] <- 0 
# Restore rownames
rownames(abund1_full) <- all_taxa 
rownames(abund2_full) <- all_taxa 
# Combine abundance data 
combined_abundance <- cbind(abund1_full, abund2_full) 
# Add source label to metadata 
meta1$Dataset <- "Data1" #China 
meta2$Dataset <- "Data2" #Japan 
# Combine metadata 
combined_meta <- rbind(meta1, meta2) 
# Check alignment 
stopifnot(all(colnames(combined_abundance) %in% 
rownames(combined_meta))) 
# Create phyloseq object 
otu_mat <- otu_table(combined_abundance, taxa_are_rows = TRUE) 
meta_df <- sample_data(combined_meta) 
physeq <- phyloseq(otu_mat, meta_df) 
# Create MBECS object 
mbec_obj <- mbecProcessInput(physeq, required.col = c("Batch", 
"Group")) 
# Apply Total Sum Scaling (TSS) and Centered Log-Ratio (CLR) 
transformations 
mbec_obj <- mbecTransform(mbec_obj, method = "tss") 
mbec_obj <- mbecTransform(mbec_obj, method = "clr")  # Default CLR 
transform 
# Ensure Batch and Group columns are factors 
mbec_obj@sam_data$Batch <- as.factor(mbec_obj@sam_data$Batch) 
mbec_obj@sam_data$Group <- as.factor(mbec_obj@sam_data$Group) 
# Run all MBECS correction methods 
cor_data_all <- mbecRunCorrections( 
input.obj = mbec_obj, 
model.vars = c("Batch", "Group"), 
method = c("rbe", "bmc", "svd", "bat"), 
type = "clr") 
# Pre- and post-correction PCA plots 
mbecPCA(input.obj = mbec_obj, model.vars = c("Batch"), 
type = "clr", pca.axes = c(1, 2), return.data = FALSE) 
mbecPCA(input.obj = mbec_obj, model.vars = c("Batch"), 
type = "cor", label=’rbe’, pca.axes = c(1, 2), return.data = 
FALSE)

# Extract corrected data (example: RBE method) 
rbe_data <- mbecGetData(cor_data_all, type = "cor", label = "rbe", 
orientation = "sxf") 
write.csv(rbe_data, "rbe_combined.csv", row.names = TRUE) 
# Evaluate variance explained by batch and biological variables 
pvca.variance <- mbecModelVariance( 
input.obj = cor_data_all, 
model.vars = c("Batch", "Group"), 
method = "pvca", 
type = "clr") 
