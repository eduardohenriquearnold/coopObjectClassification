import torch
import torch.nn as nn

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import itertools

import dataset
import model

#Load dataset
batch_size = 8
modelnetDataset = dataset.ModelnetMV('data/test_0.30_rot')
loader = torch.utils.data.DataLoader(modelnetDataset, batch_size=batch_size, shuffle=True, num_workers=4)

#Create model
mvcnn = model.MVCNN(mode='vp', numClasses=len(modelnetDataset.classes)).cuda()
mvcnn.load_state_dict(torch.load('models/imp/vp_rloss.pt'))
mvcnn.eval()

#Evaluate
ys = []
yps = []
for i, data in enumerate(loader):
	#Get data
	x, y = data
	x = x.cuda()
	y = y.cuda()

	#Predict
	yp = mvcnn(x)
	_, plabels = torch.max(yp, 1)

	#Keep ground truths and predictions
	ys.append(y)
	yps.append(plabels)

#Concatenate to one vector
ys = torch.cat(ys, 0)
yps = torch.cat(yps, 0)

#Generate stats
acc = (ys == yps).type(torch.float).mean().item()
print('Overall acc {:.4f}'.format(acc))

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
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

ys, yps = ys.cpu().numpy(), yps.cpu().numpy()
f1 = f1_score(ys, yps, average='weighted')
cm = confusion_matrix(ys, yps)
plot_confusion_matrix(cm, classes=modelnetDataset.classes, normalize=True, title='Confusion matrix, VP\nF1={:.3f}'.format(f1))
plt.show()
