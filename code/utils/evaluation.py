import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    if save_as:
        plt.savefig(save_as + '_acc.jpg')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    if save_as:
        plt.savefig(save_as + '_loss.jpg')
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