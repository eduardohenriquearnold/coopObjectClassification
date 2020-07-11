# Cooperative Object Classification for Driving applications 

This project aims at evaluating cooperative approaches towards object classification on monocular images. It uses 3D models to render objects at different viewpoints.
The results obtained with these experiments were published [here](https://ieeexplore.ieee.org/abstract/document/8813811).
Please cite using
```
@INPROCEEDINGS{8813811,
  author={E. {Arnold} and O. Y. {Al-Jarrah} and M. {Dianati} and S. {Fallah} and D. {Oxtoby} and A. {Mouzakitis}},
  booktitle={2019 IEEE Intelligent Vehicles Symposium (IV)}, 
  title={Cooperative Object Classification for Driving Applications}, 
  year={2019},
  volume={},
  number={},
  pages={2484-2489}
}
```

Dataset
==========

Dataset is generated from 3D models of the Shapenet dataset with the Blender render engine. The folder **render_scripts** has scripts to generate the dataset, which will reside in the **data** folder.

Models
========

All models are stored in the **models** folder. Models on the main model folder were trained on impairment free data, whilst the ones residing in **impairments** were trained with occlusion and additive gaussian noise.


Scripts
==========

train.py and test.py are used to individually train models of specific configurations. In order to train and evaluate multiple models in a batch use train_extended.py and test_extended.py. Note that test_extended.py will generate raw results, including predicted probabilities, confusion matrix and F1 scores, on npy format to the **results** folder. This data can be used to evaluate different metrics without having to re-evaluate the dataset. Please note that the results in that folder are specific to a experiment.


Experiments Description
=========================

1. Train and test on impairment free data
2. Train on impairment free, evaluate with fixed occlusion size 0.3
3. Train on impairment free, evaluate with fixed occlusion 0.3 and additive gaussian noise sigma 0.05
4. Train on impaired data, evaluate on different occlusion sizes
5. Train on impaired data, evaluate on different gaussian noise powers
6. Train on impaired data, evaluate with same occlusion level (0.3) but with random object orientation
