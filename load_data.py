import pickle
import sklearn

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
#file = open('data1572202842.2083826.pkl', 'rb')

#file = open('data1572202842.2083826.pkl', 'rb')
#file = open('data1572408488.0855298.pkl', 'rb')
file = open('data1572266933.4238122.pkl', 'rb') #all data
#file = open('data1572217916.17099.pkl', 'rb') #MLP
#data1572465709.8507457 baging
#file = open('data1572202842.2083826.pkl', 'rb')
#file = open('data1572405451.3997903.pkl', 'rb')
#file = open('data1572465709.8507457.pkl', 'rb')
#file = open('data1572403567.6981306.pkl', 'rb')
#file = open('data1572404699.4220088.pkl', 'rb')
# dump information to that file
dddd = pickle.load(file)
sklearn.metrics.confusion_matrix(dddd.y_test,dddd.pred)
# close the file
file.close()


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = [0,1]#classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plot_confusion_matrix(dddd.y_test, dddd.pred, classes=[0,1],
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(dddd.y_test, dddd.pred, classes=[0,1], normalize=True,
                      title='Normalized confusion matrix')



plt.show()

x=0
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
estimators_num = len(dddd.clf.estimators_)
X = range(1, estimators_num + 1)
ax.plot(list(X), list(dddd.clf.stage_score(dddd.x_train,dddd. y_train)), label="Traing score")
ax.plot(list(X), list(dddd.clf.stage_score(dddd.x_test, dddd.y_test)), label="Testing score")
ax.set_xlabel("estimator num")
ax.set_ylabel("score")
ax.legend(loc="best")
ax.set_title("AdaBoostClassifier")
plt.show()

