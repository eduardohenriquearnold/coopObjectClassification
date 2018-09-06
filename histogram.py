import numpy as np
import matplotlib.pyplot as plt

import dataset

def getHist(train=True):
	d = dataset.ModelnetMV('data', train=train)

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
	
getHist(True)
getHist(False)	
plt.show()

