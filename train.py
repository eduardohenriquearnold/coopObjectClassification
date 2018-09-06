import torch
import torch.nn as nn

import dataset
import model

epochs = 10
batch_size = 64

#Load dataset
modelnetDataset = dataset.ModelnetMV('data', train=True)
loader = torch.utils.data.DataLoader(modelnetDataset, batch_size=batch_size, shuffle=True, num_workers=4)

#Create model
mvcnn = model.MVCNN(mode='vp', numClasses=len(modelnetDataset.classes)).cuda()
mvcnn.train()

#Set last conv before maxpooling to be optimized
for param in mvcnn.cnn1[18].parameters():
	param.requires_grad = True

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
		
torch.save(mvcnn.state_dict(), 'models/vp.pt')
