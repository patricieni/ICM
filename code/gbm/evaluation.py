import matplotlib.pyplot as plt 
import os 
from pathlib import Path
import argparse

import pandas as pd
import numpy as np
from pathlib import Path
import os 
from datetime import timedelta
from sklearn.metrics import adjusted_rand_score, accuracy_score, confusion_matrix, r2_score
from sklearn.metrics import log_loss,auc,classification_report,roc_auc_score, roc_curve
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Compute ROC curve and ROC area for each class manually as there seems to be no good library 
# to do it for a multiclass problem..
# for tpr  and fpr sum over rows
def multi_class_auc(size, y_test, y_score):

    r = size
    fpr = dict()
    tpr = dict()
    roc_auc = np.zeros(r)

    #temp = cm
    for i in range(r):
                #sum1 = np.sum(temp[:,i])
                #total_sum = np.sum(temp)

                #sum2 = np.sum(temp[:,i]) - temp[i,i]
                #lower = total_sum - np.sum(temp[0,:])


        # tpr[j,i] = temp[i,i]/sum1
        # fpr[j,i] = (lower-sum2)/(total_sum-sum1)
        tpr[i], fpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    weighted_roc_auc = np.sum(roc_auc)/r
    return weighted_roc_auc

'''
# Plot of a ROC curve for a specific class
for i in range(cm[5]):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
'''

def plot_confusion_matrix(cm, classes=None, title='Confusion matrix', norm=False, **kwargs):
    """Plots a confusion matrix."""
    heatmap_kwargs = dict(annot=True, fmt='d')
    if norm:
        cm_norm = cm/cm.sum(axis=1)[:, np.newaxis]
        heatmap_kwargs['data'] = cm_norm
        heatmap_kwargs['vmin']=0.
        heatmap_kwargs['vmax']=1.
        heatmap_kwargs['fmt']='.3f'
    else:
        heatmap_kwargs['data'] = cm
    if classes is not None:
        heatmap_kwargs['xticklabels']=classes
        heatmap_kwargs['yticklabels']=classes
    heatmap_kwargs.update(kwargs)
    sns.heatmap(**heatmap_kwargs)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')