import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import dataset
import itertools

modelnetDataset = dataset.ModelnetMV('data/test_0.00/')

#Generate confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=18)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=30, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)

def cmexp(n):
    resultPaths = ['results/exp{}/{}.npz'.format(n, f) for f in ('vot', 'vp', 'conc')]

    for i, resultPath in enumerate(resultPaths, 1):
        result = np.load(resultPath)
        cm = result['cm']
        desc = result['desc']
        f1 = result['f1']

        plt.subplot(1, 3, i)
        plot_confusion_matrix(cm, classes=modelnetDataset.classes, normalize=True, title='{}\nF1={:.3f}'.format(desc, f1))

        #  plt.subplots_adjust(top=0.925, bottom=0.09, left=0.03, right=0.99, hspace=0.54, wspace=0.2)
    plt.subplots_adjust(top=0.94,bottom=0.09,left=0.11,right=0.99,hspace=0.515,wspace=0.465)
    plt.show()


def exp4():

    def exp4_ind(mode, desc, ax):

        f1_cs = []
        f1s = []
        occRange = np.arange(0,0.5, 0.05)
        for occSize in occRange:
            path = 'results/exp4/{}/occ{:.2f}.npz'.format(mode, occSize)
            result = np.load(path)
            f1_c = result['f1_class']
            f1_cs.append(f1_c)
            f1s.append(result['f1'])

        #f1_cs (rows->occ_size ; columns->classes)
        f1_cs = np.array(f1_cs)
        f1s = np.array(f1s)

        plt.figure(1)
        for i, cl in enumerate(modelnetDataset.classes):
            plt.plot(occRange, f1_cs[:,i], label=cl)

        ax.set_xlabel('Occlusion size')
        ax.set_ylabel('F1 score')
        ax.set_title(desc)
        ax.legend(ncol=2)

        plt.figure(2)
        plt.plot(occRange, f1s, label=desc)
        plt.legend(loc='lower left')
        plt.xlabel('Occlusion size')
        plt.ylabel('F1 score')
        plt.axvline(x=0.3, alpha=0.1, color='red', linestyle='dashed') #Plot vertical line on the occlusion size used for training

    modes = {'vot':'Voting', 'vp':'View pooling', 'conc':'Concatenation'}

    for i, m in enumerate(modes.keys(), 1):
        plt.figure(1)
        ax = plt.subplot(1,3,i)
        exp4_ind(m, modes[m], ax)

    plt.show()

def exp5():

    def exp5_ind(mode, desc, ax):

        f1_cs = []
        f1s = []
        noiseRange = np.arange(0,0.15, 0.01)
        for noisePower in noiseRange:
            path = 'results/exp5/{}/gn{:.2f}.npz'.format(mode, noisePower)
            result = np.load(path)
            f1_c = result['f1_class']
            f1_cs.append(f1_c)
            f1s.append(result['f1'])

        #f1_cs (rows->occ_size ; columns->classes)
        f1_cs = np.array(f1_cs)
        f1s = np.array(f1s)

        plt.figure(1)
        for i, cl in enumerate(modelnetDataset.classes):
            plt.plot(noiseRange, f1_cs[:,i], label=cl)

        ax.set_xlabel('Gaussian Noise Std Dev')
        ax.set_ylabel('F1 score')
        ax.set_title(desc)
        ax.legend(ncol=2)

        plt.figure(2)
        plt.plot(noiseRange, f1s, label=desc)
        plt.legend(loc='lower left')
        plt.xlabel('Gaussian Noise Std Dev')
        plt.ylabel('F1 score')
        plt.xticks(noiseRange)
        plt.axvline(x=0.05, alpha=0.1, color='red', linestyle='dashed') #Plot vertical line on the occlusion size used for training

    modes = {'vot':'Voting', 'vp':'View pooling', 'conc':'Concatenation'}

    for i, m in enumerate(modes.keys(), 1):
        plt.figure(1)
        ax = plt.subplot(1,3,i)
        exp5_ind(m, modes[m], ax)

    plt.show()

def exp1():
    cmexp(1)

def exp2():
    cmexp(2)

def exp3():
    cmexp(3)

if __name__ == '__main__':
    'argument should be expX, where X is 1,2,..,6'
    f = sys.argv[1]
    if f not in [f'exp{i}' for i in range(1,7)]:
        print('Invalid Experiment!')
    else:
        f = eval(f)
        f()
