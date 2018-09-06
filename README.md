# MonoMultiView

This project aims at evaluating cooperative approaches towards object classification on monocular images. It uses 3D models to render objects at different viewpoints.

Dataset
==========

Dataset is generated from 3D models with the Blender render engine. The folder **render_scripts** has scripts to generate the dataset, which will reside in the **data** folder.

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
