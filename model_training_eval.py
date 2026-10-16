#Model Training and Evaluation 
# Import required libraries 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import ( 
roc_curve, auc, precision_recall_curve, average_precision_score, 
confusion_matrix, precision_score, recall_score, f1_score, 
accuracy_score) 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
# Set Seaborn theme 
sns.set_theme(style='whitegrid', font_scale=1.1) 
# Hyperparameter Tuning using GridSearchCV 
param_grid = { 
'n_estimators': [50, 100, 200], 
'max_depth': [None, 5, 10, 20], 
'min_samples_split': [2, 5, 10], 
'min_samples_leaf': [1, 2, 4], 
'max_features': ['sqrt', 'log2', None], 'bootstrap': [True, 
False]} 
rf = RandomForestClassifier(random_state=42) 
grid_search = GridSearchCV( 
estimator=rf, param_grid=param_grid, 
cv=5,scoring='roc_auc',n_jobs=-1,verbose=1)
grid_search.fit(X_train_final, y_train) 
best_rf = grid_search.best_estimator_ 
# Bootstrapped Evaluation (1000 Iterations) 
n_iterations = 1000 
rng = np.random.RandomState(42) 
# Prepare containers for metrics 
roc_aucs, avg_precisions = [], [] 
precision_scores, recall_scores, f1_scores, acc_scores = [], [], [], 
[] 
conf_matrices, tprs, prcs = [], [], [] 
base_fpr = np.linspace(0, 1, 101) 
base_recall = np.linspace(0, 1, 101) 
for _ in range(n_iterations): 
indices = rng.randint(0, len(y_test), len(y_test)) 
X_bs = X_test.iloc[indices] 
y_bs = np.array(y_test)[indices] 
# Skip if only one class in bootstrap sample 
if len(np.unique(y_bs)) < 2: 
continue 
y_pred_bs = best_rf.predict(X_bs) 
y_proba_bs = best_rf.predict_proba(X_bs)[:, 1] 
# ROC and AUC 
fpr, tpr, _ = roc_curve(y_bs, y_proba_bs, pos_label='RBD') 
interp_tpr = np.interp(base_fpr, fpr, tpr) 
interp_tpr[0] = 0.0 
tprs.append(interp_tpr) 
roc_aucs.append(auc(fpr, tpr)) 
# Precision-Recall Curve 
precision, recall, _ = precision_recall_curve(y_bs, y_proba_bs, 
pos_label='RBD') 
interp_pr = np.interp(base_recall, recall[::-1], precision[::
1])  
# Ensures monotonicity 
prcs.append(interp_pr) 
avg_precisions.append(average_precision_score(y_bs, y_proba_bs, 
pos_label='RBD')) 
# Classification metrics 
precision_scores.append(precision_score(y_bs, y_pred_bs, 
pos_label='RBD')) 
recall_scores.append(recall_score(y_bs, y_pred_bs, 
pos_label='RBD')) f1_scores.append(f1_score(y_bs, y_pred_bs, pos_label='RBD')) 
acc_scores.append(accuracy_score(y_bs, y_pred_bs)) 
conf_matrices.append(confusion_matrix(y_bs, y_pred_bs, 
labels=best_rf.classes_)) 
# Plotting Results 
title_fontsize = 14 #font sizes 
label_fontsize = 12 
tick_fontsize = 10 
legend_fontsize = 11 
annot_fontsize = 11 
tprs = np.array(tprs) 
prcs = np.array(prcs) 
fig, axs = plt.subplots(2, 2, figsize=(15, 12)) 
# --- ROC Curve with 95% CI --- 
mean_tpr = tprs.mean(axis=0) 
std_tpr = tprs.std(axis=0) 
tpr_upper = np.minimum(mean_tpr + 1.96 * std_tpr, 1) 
tpr_lower = np.maximum(mean_tpr - 1.96 * std_tpr, 0) 
mean_auc = np.mean(roc_aucs) 
axs[0, 0].plot(base_fpr, mean_tpr, color='green', label=f'Mean AUC = 
{mean_auc:.3f}', lw=2) 
axs[0, 0].fill_between(base_fpr, tpr_lower, tpr_upper, 
color='green', alpha=0.2, label='95% CI') 
axs[0, 0].plot([0, 1], [0, 1], linestyle='--', color='gray') 
axs[0, 0].set_title('ROC Curve', fontsize=title_fontsize) 
axs[0, 0].set_xlabel('False Positive Rate', fontsize=label_fontsize) 
axs[0, 0].set_ylabel('True Positive Rate', fontsize=label_fontsize) 
axs[0, 0].tick_params(axis='both', labelsize=tick_fontsize) 
axs[0, 0].legend(fontsize=legend_fontsize) 
# --- Precision-Recall Curve with 95% CI --- 
mean_pr = prcs.mean(axis=0) 
std_pr = prcs.std(axis=0) 
pr_upper = np.minimum(mean_pr + 1.96 * std_pr, 1) 
pr_lower = np.maximum(mean_pr - 1.96 * std_pr, 0) 
mean_ap = np.mean(avg_precisions) 
axs[0, 1].plot(base_recall, mean_pr, color='olive', label=f'Mean AP 
= {mean_ap:.3f}', lw=2) 
axs[0, 1].fill_between(base_recall, pr_lower, pr_upper, 
color='olive', alpha=0.2, label='95% CI') 
axs[0, 1].set_title('Precision-Recall Curve', 
fontsize=title_fontsize) 
axs[0, 1].set_xlabel('Recall', fontsize=label_fontsize) 
axs[0, 1].set_ylabel('Precision', fontsize=label_fontsize) 
axs[0, 1].tick_params(axis='both', labelsize=tick_fontsize) 
axs[0, 1].legend(fontsize=legend_fontsize) 
# --- Score Distributions --- 
metric_df = pd.DataFrame({ 
'Accuracy': acc_scores, 
'Precision': precision_scores, 
'Recall': recall_scores, 
'F1-score': f1_scores}) 
sns.boxplot(data=metric_df, ax=axs[1, 0], palette='Set1') 
axs[1, 0].set_title('Score Distributions', fontsize=title_fontsize) 
axs[1, 0].set_ylim(0, 1.05) 
axs[1, 0].set_ylabel('Score', fontsize=label_fontsize) 
axs[1, 0].tick_params(axis='both', labelsize=tick_fontsize) 
# --- Average Confusion Matrix --- 
conf_avg = np.mean(conf_matrices, axis=0) 
sns.heatmap(conf_avg, annot=True, fmt='.1f', cmap='Greens', 
xticklabels=best_rf.classes_, 
yticklabels=best_rf.classes_, 
ax=axs[1, 1], annot_kws={"size": annot_fontsize}) 
axs[1, 1].set_title('Average Confusion Matrix', 
fontsize=title_fontsize) 
axs[1, 1].set_xlabel('Predicted', fontsize=label_fontsize) 
axs[1, 1].set_ylabel('Actual', fontsize=label_fontsize) 
axs[1, 1].tick_params(axis='both', labelsize=tick_fontsize) 
# --- Layout and viewing --- 
plt.suptitle('Control vs. RBD: Bootstrapped Model Performance (Test 
Set)', fontsize=16, fontweight='bold') 
plt.tight_layout(rect=[0, 0, 1, 0.96]) 
plt.show()
