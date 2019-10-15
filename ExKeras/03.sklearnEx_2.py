import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

class_A = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1])
proba_A = np.array([0.05, 0.05, 0.15, 0.15, 0.25, 0.25, 0.35, 0.35, 0.45, 0.45, 0.55, 0.55, 0.65, 0.85, 0.95])

class_B = np.array([0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1])
proba_B = np.array([0.05, 0.05, 0.15, 0.15, 0.25, 0.25, 0.25, 0.35, 0.35, 0.45, 0.55, 0.55, 0.65, 0.75, 0.95])

precision_A, recall_A, _ = precision_recall_curve(class_A, proba_A)
precision_B, recall_B, _ = precision_recall_curve(class_B, proba_B)

ap_A = average_precision_score(class_A, proba_A)
ap_B = average_precision_score(class_B, proba_B)

plt.title('Precision-Recall Graph')
plt.xlabel('Recall')
plt.ylabel('Precision')

plt.plot(recall_A, precision_A, 'b', label = 'Model A (AP = %0.2F)'%ap_A)
plt.plot(recall_B, precision_B, 'g', label = 'Model B (AP = %0.2F)'%ap_B)

plt.legend(loc='upper right')
plt.show()