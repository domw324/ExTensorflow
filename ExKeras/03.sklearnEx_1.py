import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

class_A = np.array([0, 0, 0, 0, 1, 1, 0, 0, 1, 1])
proba_A = np.array([0.05, 0.15, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.95, 0.95])

class_B = np.array([0, 0, 1, 0, 1, 0, 0, 1, 0, 1])
proba_B = np.array([0.05, 0.05, 0.15, 0.25, 0.35, 0.45, 0.65, 0.75, 0.85, 0.95])

false_positive_rate_A, true_positive_rate_A, thresholds_A = roc_curve(class_A, proba_A)
false_positive_rate_B, true_positive_rate_B, thresholds_B = roc_curve(class_B, proba_B)
roc_auc_A = auc(false_positive_rate_A, true_positive_rate_A)
roc_auc_B = auc(false_positive_rate_B, true_positive_rate_B)

plt.title('Receiver Operating Characteristic')
plt.xlabel('False Positive Rate(1 - Specificity)')
plt.ylabel('True Positive Rate(Sensitivity)')

plt.plot(false_positive_rate_A, true_positive_rate_A, 'b', label='Model A (AUC = %0.2f)'% roc_auc_A)
plt.plot(false_positive_rate_B, true_positive_rate_B, 'g', label='Model B (AUC = %0.2f)'% roc_auc_B)
plt.plot([0,1],[1,1],'y--')
plt.plot([0,1],[0,1],'r--')

plt.legend(loc='lower right')
plt.show()