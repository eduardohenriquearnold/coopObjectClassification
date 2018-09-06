import torch
import torch.nn as nn

import dataset
import model
import gc

def train(mode, modelpath, noise=False):
	epochs = 10
	batch_size = 32 if mode=='conc' else 64

	#Load dataset
	modelnetDataset = dataset.ModelnetMV('data', train=True)
	loader = torch.utils.data.DataLoader(modelnetDataset, batch_size=batch_size, shuffle=True, num_workers=4)

	#Create model
	mvcnn = model.MVCNN(mode=mode, numClasses=len(modelnetDataset.classes)).cuda()
	mvcnn.train()

	#Select optimization method and loss function
	optParams = filter(lambda p: p.requires_grad, mvcnn.parameters())#Get parameters that need optimization (are not frozen)
	opt = torch.optim.SGD(optParams, lr=1e-3, momentum=0.9)
	weights = torch.from_numpy(1/modelnetDataset.histogram()).type(torch.FloatTensor).cuda()
	lossF = torch.nn.CrossEntropyLoss(weight=weights)

	#Start training process
	for e in range(epochs):
		rloss = 0
		for b, data in enumerate(loader):
			#Get data
			x, y = data
			
			if noise:
				n = torch.zeros((x.size()[3],x.size()[4])).normal_(0, 0.05) #same noise mask to all views of the same object
				x += n
				x.clamp_(0,1)
						
			x = x.cuda()
			y = y.cuda()		

			#Zero grad
			opt.zero_grad()

			#Forward pass
			yp = mvcnn(x)

			#Calculate loss and backprop
			loss = lossF(yp, y)
			rloss += loss.item()
			loss.backward()
			opt.step()

			if b%10 == 0:
				print("Epoch {:3d}. Batch {:3d}/{:<3d}. Running loss {:8.6f}".format(e, b, len(loader), rloss/10))
				rloss = 0
			
	torch.save(mvcnn.state_dict(), modelpath)
	
def train_all(impairment=False):
	'''Train models with impairments or not. (Occlusion must be generated on the dataset with the generate.sh script)'''
	imp = 'imp' if impairment else 'impfree'
	modes = ['sv0', 'sv1', 'sv2', 'vot', 'vp', 'conc']
	modelPaths = ['models/{}/{}.pt'.format(imp, mode) for mode in modes]	
	
	for mode, modelPath in zip(modes, modelPaths):
		print('Training {}'.format(mode))
		train(mode, modelPath, noise=impairment)
		gc.collect() #Garbage collector (avoid GPU running out of memory)
		
	
		
train_all(impairment=True)
