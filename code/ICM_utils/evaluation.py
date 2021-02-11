import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from sklearn.metrics import classification_report, confusion_matrix, \
    accuracy_score, roc_curve, auc

def plot_report(y_test, y_pred, labels):
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(confusion_matrix(y_test, y_pred),
                          classes=labels, title='Confusion matrix')

    print(classification_report(y_test, y_pred, target_names=labels))
    print('Accuracy: {}'.format(accuracy_score(y_test, y_pred)))


def plot_hist(history, save_as=None):
    for metric in [k for k in history.history.keys() if 'val' not in k]:
        plt.plot(history.history[metric])
        plt.plot(history.history['val_' + metric])
        plt.title('Model ' + metric)
        plt.ylabel(metric)
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        if save_as:
            plt.savefig(save_as + '_acc.jpg')
        plt.show()


def plot_confusion_matrix(cm, classes=None, title='Confusion matrix',
                          norm=False, **kwargs):
    """Plots a confusion matrix."""
    heatmap_kwargs = dict(annot=True, fmt='d')
    if norm:
        cm_norm = cm / cm.sum(axis=1)[:, np.newaxis]
        heatmap_kwargs['data'] = cm_norm
        heatmap_kwargs['vmin'] = 0.
        heatmap_kwargs['vmax'] = 1.
        heatmap_kwargs['fmt'] = '.3f'
    else:
        heatmap_kwargs['data'] = cm
    if classes is not None:
        heatmap_kwargs['xticklabels'] = classes
        heatmap_kwargs['yticklabels'] = classes
    heatmap_kwargs.update(kwargs)
    sns.heatmap(**heatmap_kwargs)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    
def plot_confidence_interval(predictions, ground_truth, prediction_id=123, interval_in_days=181):
    r"""
    Plot the probability interval given an array of probabilities.

    Parameters
    ----------
    predictions: pandas.DataFrame / numpy.array [N, M]
        N - number of predictions
        M - probability per interval
    
    ground_truth: pandas.DataFrame / numpy.array [N]
        N - number of entries
     
    prediction_id: int
        id of the prediction < N
    
    interval_in_days: int
        number of days over which we smooth the probabilities
    """
        
    Y_pred = np.array(predictions, copy=True)  
    window_size = interval_in_days
    
    # Smooth interval
    # Savgol changes inplace, so we need the copy above
    smooth_res = savgol_filter(predictions[prediction_id], window_size, 2)

    x = smooth_res * window_size
    y = range(len(predictions[prediction_id]))

    res = ground_truth[prediction_id]
    
    # Plot probability
    plt.fill_between(y, x, 0, color='b', alpha=0.3)
    plt.plot(res, x.max(), color='r', linewidth=10)

    # Plot results
    plt.plot([res, res], [x.max() , 0], color='r')
    plt.scatter([res], [x.max()], color='r')
    
    
# Compute ROC curve and ROC area for each class manually as there seems to be
# no good library to do it for a multiclass problem..
# for tpr  and fpr sum over rows
def multi_class_auc(size, y_test, y_score):
    r = size
    fpr = dict()
    tpr = dict()
    roc_auc = np.zeros(r)

    # temp = cm
    for i in range(r):
        # sum1 = np.sum(temp[:,i])
        # total_sum = np.sum(temp)

        # sum2 = np.sum(temp[:,i]) - temp[i,i]
        # lower = total_sum - np.sum(temp[0,:])

        # tpr[j,i] = temp[i,i]/sum1
        # fpr[j,i] = (lower-sum2)/(total_sum-sum1)
        tpr[i], fpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    weighted_roc_auc = np.sum(roc_auc) / r
    return weighted_roc_auc

def confused_entries(X,y_true,y_pred,val1,val2):
    """Creates a dataframe based on what entries were predicted as val2 and should have been predicted as val1
    Perhaps it makes sense to return the diagonal entries as well
    
    Args:
        X (df): Dataset with features
        y_true (dic): True target variables for dataset X - as a dictionary!
        y_pred (array): Predicted target variables by the model
        val1 (string): The actual label value that should have been predicted (true label)
        val2 (string): The predicted label that was wrong/right (false label)
    Returns:
        pandas.dataframe: a dataframe with all val1 entries from X that were confused with val2
    """

    confused = []
    for key1,key2 in zip(y_true.keys(),y_pred):
    
        if y_true.get(key1)== val1 and key2 == val2:
            confused.append(key1)
    return X.loc[confused,:]


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