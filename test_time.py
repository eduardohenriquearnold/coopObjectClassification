import torch
import torch.nn as nn

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import itertools

import dataset
import model

import time

#Load dataset
batch_size = 8
modelnetDataset = dataset.ModelnetMV('data', train=False)
loader = torch.utils.data.DataLoader(modelnetDataset, batch_size=batch_size, shuffle=True, num_workers=4)

#Create model
mvcnn = model.MVCNN(mode='vot').cuda()
mvcnn.load_state_dict(torch.load('models/mvcnn_vot.pt'))
mvcnn.eval()

#Evaluate
ys = []
yps = []
ts = []
for i, data in enumerate(loader):
	torch.cuda.synchronize()
	t0 = time.perf_counter()
	#Get data
	x, y = data
	x = x.cuda()
	y = y.cuda()

	#Predict
	yp = mvcnn(x)
	_, plabels = torch.max(yp, 1)
	
	torch.cuda.synchronize()
	t1 = time.perf_counter()

	#Keep ground truths and predictions
	ys.append(y)
	yps.append(plabels)
	ts.append(t1-t0)

#Concatenate to one vector
ys = torch.cat(ys, 0)
yps = torch.cat(yps, 0)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

etime = np.array(ts[0:-1])
etime = len(etime)*etime.mean()
print('Model contains {} parameters'.format(count_parameters(mvcnn)))
print('Classified {} samples in {:.02e}s'.format(ys.shape[0], etime))


