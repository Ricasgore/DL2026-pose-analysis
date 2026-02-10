# Method_Automatic_pose_recognition

The goal of our project is to try to predict someone’s pose in a real time video. 

## Prediction Models
We created a CNN model from scratch and also used a pretrained ResNet50 model. We first trained our models on the LSPPE Dataset (referred to as 'first dataset' in the comments of our code), and then we created additional models using the MPII Dataset.

We compared the four models, as shown in the "loss_graph_plot" and "auc" folders.

We also tested the OpenPose model (https://github.com/CMU-Perceptual-Computing-Lab/openpose?ref=blog.roboflow.com) on both datasets. You can see the AUC results to evaluate its performance.

## Configuration 

- You need to install PyTorch

- Download or Clone our repository

- If you want to test or train with the LSPPE Datasets, you need to install it using this link https://github.com/axelcarlier/lsp . We put all the files in a "lsp_dataset" folder.
- If you want to test or train with the MPII Datasets, you need to install it using this link https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/software-and-datasets/mpii-human-pose-dataset/download . We put all the files in a "mpii_dataset" folder.

- If you want to use one of our four trained models, you need to install the checkpoints which are in the release.

## Real time Prediction 

Run the human_detection.py file (you must choose the model you want to use) (we advise you to use the Resnet50 model trained with the 2nd dataset).
