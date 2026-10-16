#B4. Differential Abundance Analysis (in R) 
#Differential Abundance Analysis 
library(ANCOMBC) 
# 4. Run ANCOM-BC2 with batch correction via fix_formula 
results <- ancombc2( 
data = physeq, 
fix_formula = "Group + Batch",     # Adjust for batch effect 
p_adj_method = "BH",) 
# 5. Extract and filter results 
diff_abund <- results$res 
significant_taxa <- diff_abund[diff_abund$p_adj < 0.05, ] 
