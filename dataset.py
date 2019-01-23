import os

import torch
import torch.utils.data as data
import torchvision

import numpy as np

from PIL import Image

class ModelnetMV(torchvision.datasets.DatasetFolder):
    def __init__(self, root):

        self.numberViews = 3
        self.imageSize = (224,224)

        self.root = root

        super().__init__(self.root, self.loader, extensions=['.png'])

        #Discard multiple views of the same object, keep only the first view (ex. objid-0.png)
        self.samples = self.samples[::self.numberViews]

    def loader(self, path):
        '''Load the perspective views for each given off file'''

        imgs = []
        for i in range(self.numberViews):
            ipath = path[:-5]+'{}.png'.format(i)
            with open(ipath, 'rb') as f:
                img = Image.open(f)
                img = img.convert('RGB')
                img = img.resize(self.imageSize)
                img = torchvision.transforms.ToTensor()(img).unsqueeze(0)
                imgs.append(img)

        imgs = torch.cat(imgs, 0)
        return imgs

    def histogram(self):
        targets = [s[1] for s in self.samples]
        hist, _ = np.histogram(targets, bins=len(self.classes))
        return hist
