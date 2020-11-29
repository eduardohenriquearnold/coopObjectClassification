import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import numpy as np
import matplotlib.pyplot as plt

import dataset

def getHist(folder, train=True):
	d = dataset.ModelnetMV(folder)

	hist = d.histogram()
	nclasses = len(d.classes)
	classes = d.classes

	width=0.3
	bins = list(map(lambda x: x-width/2,range(1,nclasses+1)))

	ax = plt.subplot(111)
	lbl = 'Train' if train else 'Test'
	ax.bar(bins, hist, width=width, label=lbl)
	ax.set_xticks(bins)
	ax.set_xticklabels(classes,rotation=45)
	ax.set_title("Classes histogram")
	plt.legend(loc='upper right')

getHist('data/train_0', True)
getHist('data/test_0.00', False)
plt.show()
