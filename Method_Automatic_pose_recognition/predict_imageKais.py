import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from datasetKais import *
from models import *
from utils import *
from metrics import *
from config import device
import os
import matplotlib.pyplot as plt

image_path = '/home/amine_tsp/DL2026/Method_Automatic_pose_recognition/predict_one_image/image_test.png'

image = Image.open(image_path).convert('RGB')

def predict_dataset1(model, SavingName, image):
    print(SavingName)
    model.load_state_dict(torch.load(SavingName, map_location=torch.device('cpu')))

    tr = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    image = tr(image).unsqueeze(0)  
    image = image.to(device)

    model.eval()
    with torch.no_grad():
        output = model(image).squeeze().cpu().numpy()
        pred = np.array(output).reshape(14,3)

    return pred[:,0], pred[:,1]

def predict_dataset2(model, SavingName, image):
    model.load_state_dict(torch.load(SavingName, map_location=torch.device('cpu')))

    tr = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    image = tr(image).unsqueeze(0)  
    image = image.to(device)

    model.eval()
    with torch.no_grad():
        output = model(image).round()
        pred = output.squeeze().cpu().numpy()
        image = image.cpu().numpy()
        pred_x = []
        pred_y = []
        for j in range(len(pred)):
            if j%2==0:
                pred_x.append(pred[j])
            if j%2==1 : 
                pred_y.append(pred[j])
    return pred_x, pred_y


#########################################
# model = 0  => own cnn (not pretrained)#
# model = 1  => resnet                  #
# dataset = 1  => first dataset         #
# dataset = 2  => second dataset        #
#########################################

(model,dataset) = (0,1)

if (model,dataset) == (0,1):
    CNNI = CNN(num_classes=42).to(device)
    CNNI = CNNI.to(device)
    pred_x, pred_y = predict_dataset1(model=CNNI, SavingName='/home/amine_tsp/DL2026/Method_Automatic_pose_recognition/checkpoints/CNN-pose-Basic_dataset1.ckpt', image=image)

elif (model, dataset) == (0,2):
    CNNI = CNN(num_classes=32).to(device)
    CNNI = CNNI.to(device)
    pred_x, pred_y = predict_dataset1(model=CNNI, SavingName='./checkpoints/CNN-pose-Basic_dataset2.ckpt', image=image)

elif (model, dataset) == (1,1):
    sel = 0
    list_models = ['resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet', 'inception']
    MODEL = list_models[sel]
    CNNI, input_size = get_pretrained_models(model_name = list_models[sel], num_classes= 42, freeze_prior = False, 
                                            use_pretrained = True)
    CNNI = CNNI.to(device)
    pred_x, pred_y = predict_dataset1(model=CNNI, SavingName='./checkpoints/CNN-pose-Resnet50_dataset1.ckpt', image=image)

elif (model,dataset) == (1,2):
    sel = 0
    list_models = ['resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet', 'inception']
    MODEL = list_models[sel]
    CNNI, input_size = get_pretrained_models(model_name = list_models[sel], num_classes= 32, freeze_prior = False, 
                                            use_pretrained = True)
    CNNI = CNNI.to(device)
    pred_x, pred_y = predict_dataset1(model=CNNI, SavingName='./checkpoints/CNN-pose-Resnet50_dataset2.ckpt', image=image)

 

height, width = image.height , image.width

# Fit predicted points to original box size
scale_x = height / 128  # Scaling factor for x
scale_y = width / 128  # Scaling factor for y

pred_x = np.array(pred_x) * scale_x
pred_y = np.array(pred_y) * scale_y

if dataset == 1 : 
    #Joints For the FIRST Dataset
    joint_order = ['r ankle','r knee','2r hip','l hip','4l knee','l ankle','6r wrist','r elbow','8r shoulder','l shoulder','10l elbow','l wrist','12neck','Head top']
    joint_connexion=[[0,1],[1,2],[2,3],[3,4],[4,5],[6,7],[7,8],[8,12],[9,12],[9,10],[10,11],[12,13]]

elif dataset == 2 :
    #Joints For the SECOND Dataset
    joint_order = ['r ankle','r knee','r hip','l hip','l knee','l ankle','pelvis','thorax','upper neck','head top','r wrist','r elbow','r shoulder','l shoulder','l elbow','l wrist']
    joint_connexion=[[0,1],[1,2],[2,6],[6,3],[3,4],[4,5],[6,7],[7,8],[8,9],[8,12],[12,11],[11,10],[8,13],[13,14],[14,15]]


plt.imshow(image)
plt.scatter(pred_x, pred_y)
for connexion in joint_connexion:
    plt.plot([pred_x[connexion[0]],pred_x[connexion[1]]],[pred_y[connexion[0]],pred_y[connexion[1]]])
plt.savefig('/home/amine_tsp/DL2026/Method_Automatic_pose_recognition/predict_one_image/test_plot_Basic_dataset1.png')
plt.close()