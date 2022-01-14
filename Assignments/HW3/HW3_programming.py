## Q3.) a.) ROC Curve For both M1 and M2 Models.
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
probability1 = np.array([0.73,0.69,0.67,0.55,0.47,0.45,0.44,0.35,0.15,0.08])
probability2 = np.array([0.68,0.61,0.45,0.38,0.31,0.09,0.05,0.04,0.03,0.01])
classlabels1 = np.array([1,1,1,0,1,1,0,0,0,0])
classlabels2 = np.array([0,1,1,0,0,1,0,0,1,1])
fpr, tpr, _ = roc_curve(classlabels1,probability1,classlabels2,probability2)
print("fpr = ", fpr) 
print("tpr = ", tpr) 
roc_auc = auc(fpr, tpr)
print("roc_auc=", roc_auc) 
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.plot(fpr, tpr)
plt.show()