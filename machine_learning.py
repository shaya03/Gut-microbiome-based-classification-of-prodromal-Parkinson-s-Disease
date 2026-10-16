#B5. Machine Learning (in Python) 
Feature Selection Pipeline 
from sklearn.feature_selection import RFE, SequentialFeatureSelector 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import cross_val_score 
import matplotlib.pyplot as plt 
import shap 
import numpy as np 
import pandas as pd 
# Step 1: Filter features based on importance threshold 
# Retain features with importance > 0.005 from initial RandomForest 
significant_features = importances[importances > 
0.005].sort_values(ascending=False)
X_significant = X_train[significant_features.index] 
# Step 2: Recursive Feature Elimination (RFE) with CV 
rfe_auc_scores = [] 
min_features = 5 
max_features = X_significant.shape[1] 
for n_features in range(min_features, max_features + 1): 
rfe = RFE(estimator=RandomForestClassifier(n_estimators=50, 
random_state=42),  
n_features_to_select=n_features) 
pipeline = Pipeline([ 
('scaler', StandardScaler()),  
('rfe', rfe),  
('rf', RandomForestClassifier(n_estimators=50, 
random_state=42)) 
]) 
scores = cross_val_score(pipeline, X_significant, y_train, cv=5, 
scoring='roc_auc') 
rfe_auc_scores.append(np.mean(scores)) 
# Determine optimal number of features using best CV score 
optimal_rfe_features = np.argmax(rfe_auc_scores) + min_features 
# Visualize performance across different feature counts 
plt.figure(figsize=(10, 6)) 
plt.plot(range(min_features, max_features + 1), rfe_auc_scores, 
marker='o') 
plt.axvline(x=optimal_rfe_features, color='r', linestyle='--') 
plt.xlabel('Number of Features') 
plt.ylabel('Mean ROC AUC Score') 
plt.title('RFE: Performance vs Number of Features') 
plt.grid() 
# Final RFE model with optimal number of features 
rfe = RFE(estimator=RandomForestClassifier(n_estimators=50, 
random_state=42), 
n_features_to_select=optimal_rfe_features) 
rfe.fit(X_significant, y_train) 
rfe_features = X_significant.columns[rfe.support_] 
# Step 3: Sequential Feature Selection (Forward & Backward) 
plt.tight_layout() 
plt.show()

# Forward selection 
sfs_forward = SequentialFeatureSelector( 
RandomForestClassifier(n_estimators=50, random_state=42), 
direction='forward', 
n_features_to_select='auto', 
scoring='roc_auc', 
cv=5, 
n_jobs=-1) 
sfs_forward.fit(X_significant, y_train) 
sfs_forward_features = 
X_significant.columns[sfs_forward.get_support()] 
# Backward selection 
sfs_backward = SequentialFeatureSelector( 
RandomForestClassifier(n_estimators=50, random_state=42), 
direction='backward', 
n_features_to_select='auto', 
scoring='roc_auc', 
cv=5, 
n_jobs=-1) 
sfs_backward.fit(X_significant, y_train) 
sfs_backward_features = 
X_significant.columns[sfs_backward.get_support()] 
# Step 4: SHAP-based Feature Importance 
# Train RandomForest on significant features 
rf = RandomForestClassifier(n_estimators=50, random_state=42) 
rf.fit(X_significant, y_train) 
# Compute SHAP values 
explainer = shap.TreeExplainer(rf) 
shap_values = explainer.shap_values(X_significant) 
# SHAP summary plot (bar) 
plt.figure(figsize=(12, 8)) 
shap.summary_plot(shap_values, X_significant, 
feature_names=X_significant.columns, plot_type="bar") 
plt.tight_layout() 
plt.show()

# Mean absolute SHAP value per feature for class 1 (positive class) 
mean_abs_shap = np.abs(shap_values[:, :, 1]).mean(axis=0) 
shap_importances = pd.DataFrame({'feature': X_significant.columns, 'shap_importance': 
mean_abs_shap}).sort_values('shap_importance', ascending=False) 
# Select top SHAP features using optimal feature count from RFE 
shap_features = 
shap_importances['feature'][:optimal_rfe_features].tolist() 
# Step 5: Combine Rankings from RFE, SHAP, SFS Forward, SFS Backward 
feature_scores = pd.DataFrame(index=X_significant.columns) 
# RFE rank (1 = most important) 
feature_scores['rfe_rank'] = rfe.ranking_ 
# SHAP rank 
feature_scores['shap_rank'] = 
shap_importances.set_index('feature')['shap_importance'].rank(ascend
ing=False) 
# SFS ranks: 1 if selected, else max rank 
max_rank = len(X_significant.columns) 
feature_scores['sfs_fwd_rank'] = [1 if f in sfs_forward_features 
else max_rank for f in X_significant.columns] 
feature_scores['sfs_bwd_rank'] = [1 if f in sfs_backward_features 
else max_rank for f in X_significant.columns]

# Average rank across all methods 
feature_scores['avg_rank'] = feature_scores[ 
['rfe_rank', 'shap_rank', 'sfs_fwd_rank', 
'sfs_bwd_rank']].mean(axis=1) 
# Rank features by average rank 
top_features = feature_scores.sort_values('avg_rank').index.tolist() 
# Step 6: Fine-Grained Model Evaluation vs Feature Count 
sorted_features = feature_scores.sort_values('avg_rank').index 
n_feature_options = np.arange(1, len(sorted_features) + 1) 
rank_auc_scores = [] 
# Evaluate AUC with increasing number of top-ranked features 
for n in n_feature_options: 
selected_features = sorted_features[:n] 
pipeline = Pipeline([('scaler', StandardScaler()), ('rf',     
RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=
1)) ]) 
scores = cross_val_score(pipeline, 
X_significant[selected_features], y_train,  
cv=5, scoring='roc_auc', n_jobs=-1) 
rank_auc_scores.append(np.mean(scores)) 
# Determine final optimal feature count 
optimal_n_features = n_feature_options[np.argmax(rank_auc_scores)] 

# Plot performance vs number of features 
plt.figure(figsize=(10, 6)) 
plt.plot(n_feature_options, rank_auc_scores, marker='.') 
plt.axvline(x=optimal_n_features, color='r', linestyle='--', 
label=f'Optimal = {optimal_n_features}') 
plt.xlabel('Number of Top Ranked Features') 
plt.ylabel('Mean CV ROC AUC') 
plt.title('Performance vs Number of Ranked Features') 
plt.legend() 
plt.grid(True) 
plt.tight_layout() 
plt.show()
