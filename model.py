import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class MVCNN(nn.Module):
	def __init__(self, numClasses=10, mode='vp'):
		super(MVCNN, self).__init__()
		self.numClasses = numClasses

		#Instantiate pre-trained VGG
		vgg = models.vgg11(pretrained=True)

		self.cnn1 = vgg.features

		#cnn2 consists of VGG classification network without the last layer (which defines number of classes). We add our own linear layer on top
		self.cnn2 = nn.Sequential(*list(vgg.classifier.children())[:-1])
		self.cnn2.add_module('6', nn.Linear(4096, numClasses))

		#Freezes cnn1 params
		for param in self.cnn1.parameters():
			param.requires_grad = False

		#Set forward function
		if mode[0:2] == 'vp':
			self.forward = self.forward_viewpooling
		elif mode == 'vot':
			self.forward = self.forward_voting
		elif mode[0:2] == 'sv':
			self.forward = self.forward_single_view
			self.useView = int(mode[2])
		elif mode == 'conc':
			#Modify first cnn2 layer to 3 times the original input size (concatenate all feature maps)
			cnn2layers = list(self.cnn2.children())[1:]
			cnn2layers.insert(0, nn.Linear(75264, 4096))
			self.cnn2 = nn.Sequential(*cnn2layers)
			self.forward = self.forward_concatenate


	def forward_single_view(self, x):
		'''Restricts samples to a single view'''

		#Restricts the samples to a single if useSingleView is specified:
		x = x[:,self.useView]

		#Get features for all samples in batch
		f = self.cnn1(x)

		#Transform to cnn2 input format
		f = f.view(-1, 512*7*7)

		#Performs classification
		f = self.cnn2(f)

		return f

	def forward_voting(self, x):
		'''Get full classification scores for each image. Obtains the final class through a voting scheme'''

		#Assumes input size in the form of [batch_idx, view_idx, ch, h, w] (5D). Must convert it to 4D to use standard pipeline. Image input size is [3,224,224].
		bsamples = x.size()[0]
		x = x.contiguous().view(-1,3,224,224)

		#Get features for all views in batch
		f = self.cnn1(x)

		#Transform to cnn2 input format
		f = f.view(-1, 512*7*7)

		#Performs classification
		f = self.cnn2(f)

		#Transform back to batch samples, with 3 views
		f = f.view(bsamples, 3, self.numClasses)    #results in [bsamples, 3, 10]

		#Voting implemented as mean scores along 3 views (scores are not in probability format, but log)
		f = nn.LogSoftmax(dim=2)(f).mean(dim=1)

		return f

	def forward_viewpooling(self, x):
		'''Get features for each image and then performs view pooling on the 3D volume, classification is carried on the pooled volume'''

		#Assumes input size in the form of [batch_idx, view_idx, ch, h, w] (5D). Must convert it to 4D to use standard pipeline. Image input size is [3,224,224].
		bsamples = x.size()[0]
		x = x.contiguous().view(-1,3,224,224)

		#Get features for all views in batch
		f = self.cnn1(x)

		#Transform back to batch samples, with 3 views, but concatenate the 3D feature maps into 1D
		f = f.view(bsamples, 3, 512*7*7)

		#View pooling (maximum along views dimension)
		f, _ = torch.max(f, 1)

		#uses cnn2 for classification
		f = self.cnn2(f)

		return f

	def forward_concatenate(self, x):
		'''Get features for each image and then performs view pooling on the 3D volume, classification is carried on the pooled volume'''

		#Assumes input size in the form of [batch_idx, view_idx, ch, h, w] (5D). Must convert it to 4D to use standard pipeline. Image input size is [3,224,224].
		bsamples = x.size()[0]
		x = x.contiguous().view(-1,3,224,224)

		#Get features for all views in batch
		f = self.cnn1(x)

		#Transform back to batch samples, with 3 views, but concatenate the 3D feature maps into 1D
		f = f.view(bsamples, 3*512*7*7)

		#uses cnn2 for classification
		f = self.cnn2(f)

		return f


if __name__ == '__main__':
	'''DEBUG'''
	model = MVCNN(mode='conc')
	model.eval()
	model.cuda()
	dataset = torch.autograd.Variable(torch.rand((8,3,3,224,224))).cuda()
	o = model(dataset)
