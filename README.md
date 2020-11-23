# Born-Identity-Network
Tensorflow implementation of [Born Identity Network: Multi-way Counterfactual Map Generation to Explain a Classifier's Decision](https://arxiv.org/abs/2011.10381).

### Requirements
tensorflow (2.2.0)\
tensorboard (2.2.2)\
tensorflow-addons (0.11.0)\
tqdm (4.48.0)\
matplotlib (3.3.0)\
numpy (1.19.0)\
scikit-learn (0.23.2)

### Data sets
1. [HandWritten digits data (MNIST)](http://yann.lecun.com/exdb/mnist/)
2. [3D Geometric shape data](https://github.com/deepmind/3d-shapes)

### How to run


### Config.py of each dataset 
data_path = Raw dataset path\
save_path = Storage path to save results such as tensorboard event files, model weights, etc.\
cls_weight_path = Pre-trained classifier weight path obtained in mode0 setup\
enc_weight_path = Pre-trained encoder weight path obtained in mode0 setup
