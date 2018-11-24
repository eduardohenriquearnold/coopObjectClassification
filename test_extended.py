import torch
import torch.nn as nn

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import itertools

import dataset
import model
import os
import subprocess
import gc

def test(mode, modelPath, resultsPath, desc, noisePower=False):
	#Load dataset
	batch_size = 32
	modelnetDataset = dataset.ModelnetMV('data', train=False)
	loader = torch.utils.data.DataLoader(modelnetDataset, batch_size=batch_size, shuffle=True, num_workers=4)

	#Create model
	mvcnn = model.MVCNN(mode=mode, numClasses=len(modelnetDataset.classes)).cuda()
	mvcnn.load_state_dict(torch.load(modelPath))
	mvcnn.eval()

	#Evaluate
	ys = []
	yps = []
	for i, data in enumerate(loader):
		#Get data
		x, y = data
		
		if noisePower:
			n = torch.zeros((x.size()[3],x.size()[4])).normal_(0, noisePower) #same noise mask to all images
			x += n
			x.clamp_(0,1)
					
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

	ys, yps = ys.cpu().numpy(), yps.cpu().numpy()
	f1_class = f1_score(ys, yps, average=None)
	f1 = f1_score(ys, yps, average='weighted')
	cm = confusion_matrix(ys, yps)

	np.savez(resultsPath, cm=cm, f1=f1, f1_class=f1_class, ys=ys, yps=yps, desc=desc)

def exp1():
	modes = ['sv0', 'sv1', 'sv2', 'vot', 'vp', 'conc']
	modelPaths = ['models/impfree/{}.pt'.format(mode) for mode in modes]
	resultPaths = ['results/exp1/{}.npz'.format(mode) for mode in modes]
	descriptions = ['SV, view 0', 'SV, view 1', 'SV, view 2', 'MV, voting', 'MV, view-pooling', 'MV, concatenation']
	
	for mode, modelPath, resultPath, desc in zip(modes, modelPaths, resultPaths, descriptions):
		test(mode, modelPath, resultPath, desc)
	
def exp2():
	modes = ['sv0', 'sv1', 'sv2', 'vot', 'vp', 'conc']
	modelPaths = ['models/impfree/{}.pt'.format(mode) for mode in modes]
	resultPaths = ['results/exp2/{}.npz'.format(mode) for mode in modes]
	descriptions = ['SV, view 0', 'SV, view 1', 'SV, view 2', 'MV, voting', 'MV, view-pooling', 'MV, concatenation']
	
	for mode, modelPath, resultPath, desc in zip(modes, modelPaths, resultPaths, descriptions):
		test(mode, modelPath, resultPath, desc)
		gc.collect()
		
def exp3():
	modes = ['sv0', 'sv1', 'sv2', 'vot', 'vp', 'conc']
	modelPaths = ['models/impfree/{}.pt'.format(mode) for mode in modes]
	resultPaths = ['results/exp3/{}.npz'.format(mode) for mode in modes]
	descriptions = ['SV, view 0', 'SV, view 1', 'SV, view 2', 'MV, voting', 'MV, view-pooling', 'MV, concatenation']
	
	for mode, modelPath, resultPath, desc in zip(modes, modelPaths, resultPaths, descriptions):
		test(mode, modelPath, resultPath, desc, noisePower=0.05)
		gc.collect()
		
def exp4():
	modes = ['vot', 'vp', 'conc']
	modelPaths = ['models/imp/{}.pt'.format(mode) for mode in modes]
	descriptions = ['MV voting', 'MV view-pooling', 'MV concatenation']
	
	occSizes = np.arange(0, 0.50, 0.05)
	for occSize in occSizes:
		#Generate test dataset
		subprocess.call(['data/generate.sh', 'data/test/', str(occSize)], env={})
		
		#Save sample image for diagnostics
		subprocess.call(['cp', 'data/test/car/1a1dcd236a1e6133860800e6696b8284-2.png', 'samples/1a1dcd236a1e6133860800e6696b8284-{}.png'.format(occSize)])
			
		#Generate test results for each mode
		for mode, modelPath, desc in zip(modes, modelPaths, descriptions):
			resultPath = 'results/exp4/{}/occ{:.2f}.npz'.format(mode, occSize)
			test(mode, modelPath, resultPath, desc)
			gc.collect()
			
def exp5():
	'''Same as 4, but without occlusion, and adding WGN'''

	modes = ['vot', 'vp', 'conc']
	modelPaths = ['models/imp/{}.pt'.format(mode) for mode in modes]
	descriptions = ['MV voting', 'MV view-pooling', 'MV concatenation']
	
	noisePowers = np.arange(0, 0.15, 0.01)
	for noisePower in noisePowers:
		#Generate test results for each mode
		for mode, modelPath, desc in zip(modes, modelPaths, descriptions):
			resultPath = 'results/exp5/{}/gn{:.2f}.npz'.format(mode, noisePower)
			test(mode, modelPath, resultPath, desc, noisePower=noisePower)
			gc.collect()

exp2()
exp3()
