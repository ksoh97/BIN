# Born-Identity-Network
Tensorflow implementation of [Born Identity Network: Multi-way Counterfactual Map Generation to Explain a Classifier's Decision](https://arxiv.org/abs/2011.10381).

### Network Archiecture
![Group 1387](https://user-images.githubusercontent.com/57162425/141602585-4a434cd5-530b-4ebe-b873-df11e98b8f81.png)

### Requirements
tensorflow (2.2.0)\
tensorboard (2.2.2)\
tensorflow-addons (0.11.0)\
tqdm (4.48.0)\
matplotlib (3.3.0)\
numpy (1.19.0)\
scikit-learn (0.23.2)


### Datasets
Place them into "data_path" on each Config.py
1. [HandWritten digits data (MNIST)](http://yann.lecun.com/exdb/mnist/)
2. [3D Geometric shape data](https://github.com/deepmind/3d-shapes)
3. [Alzheimer’s Disease Neuroimaging Initiative (ADNI)](http://adni.loni.usc.edu/)


### How to run
Mode:\
#0 Pre-training a classifier\
#1 Training the counterfactual map generator

1. Pre-training a classifier
>- `training.py --mode=0`

2. Training the counterfactual map generator
>- Set the classifier and encoder weight for training (freeze)
>- Change the mode from 0 to 1 on Config.py
  >- `training.py --mode=1`


### Config.py of each dataset 
data_path = Raw dataset path\
save_path = Storage path to save results such as tensorboard event files, model weights, etc.\
cls_weight_path = Pre-trained classifier weight path obtained in mode#0 setup\
enc_weight_path = Pre-trained encoder weight path obtained in mode#0 setup
