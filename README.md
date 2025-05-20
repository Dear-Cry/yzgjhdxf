# Project2 of Neural Network and Deep Learning
The code for this experiment consists of the following Python files, which have been uploaded to the GitHub repository (link provided at the end):

## Data
- __init__.py
- loaders.py # CIFAR10 data loading
- cifar-10-python.tar.gz # CIFAR10 dataset
To obtain the CIFAR10 dataset, run loaders.py in the data folder.

## Network
- figures # Images
- saved_model # Saved models
- loaders.py # CIFAR10 data loading
- model.py # Model definition
- train.py # Model training and visualization
- test.py # Model testing and visualization
- visualize.py # Visualization functions
Training: Modify necessary parameters in train.py and execute it.
The trained model will be saved as a binary file named best_model.pth by default.
Testing: Run test.py, which loads the model from best_model.pth and performs testing.
Pre-trained Models: The saved_model folder contains the following models (uploaded to Google Drive, link at the end):
TFT.pth (Section 2.1), 
Dropout.pth (Section 2.1.1), 
SGD.pth, SGDSTEP.pth, RMSProp.pth (Section 2.2.2), 
GELU.pth, LeakyReLU.pth, Sigmoid.pth, Tanh.pth (Section 2.2.3), 
MORE.pth (Section 2.2.5).

## VGG_BatchNorm
- data # Data loading and preprocessing
- figures # Images
- losses # Loss files
- models # Model definitions
- utils # Model initialization utilities
- VGG_Loss_Landscape.py # Model training and visualization
Run VGG_Loss_Landscape.py.
Note: To generate the loss landscape plot (e.g., Figure 8), you must:
1. Train models with different learning rates to obtain loss files.
2. Comment out the training section before plotting.

## Report \& Resources

The experiment report has been uploaded to elearning.

Github Repository:
- https://github.com/Dear-Cry/yzgjhdxf.git

Google Drive Link:
- https://drive.google.com/drive/folders/1-gT9ZiGitG4lnURBLZ7YECFeWFZh1vSF?usp=sharing
