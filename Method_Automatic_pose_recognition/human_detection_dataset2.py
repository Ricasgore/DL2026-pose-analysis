import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from dataset import *
from models import *
from utils import *
from metrics import *
from config import device
import os

joint_order = ['r ankle','r knee','r hip','l hip','l knee','l ankle','pelvis','thorax','upper neck','head top','r wrist','r elbow','r shoulder','l shoulder','l elbow','l wrist']
joint_connexion=[[0,1],[1,2],[2,6],[6,3],[3,4],[4,5],[6,7],[7,8],[8,9],[8,12],[12,11],[11,10],[8,13],[13,14],[14,15]]

#joint_order1 = ['r ankle','r knee','2r hip','l hip','4l knee','l ankle','6r wrist','r elbow','8r shoulder','l shoulder','10l elbow','l wrist','12neck','Head top']
#joint_connexion1=[[0,1],[1,2],[2,3],[3,4],[4,5],[6,7],[7,8],[8,12],[9,12],[9,10],[10,11],[12,13]]

"""
def predict(model, SavingName, image):
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

    return pred
"""

def predict(model, SavingName, image):
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




'''

model_name = input('What model do you want to use ? (Resnet, CNN)')
dataset = input('Which dataset do you want to use ? 1 or 2')

if model_name == Resnet and dataset == 2 :
    CNNI, input_size = get_pretrained_models(model_name = 'resnet', num_classes= 32, freeze_prior = False, 
                                            use_pretrained = True)
    joint_order = joint_order2
    joint_connexion = joint_order2

elif model_name== Resnet and dataset == 1 :
    CNNI, input_size = get_pretrained_models(model_name = 'resnet', num_classes= 48, freeze_prior = False, 
                                            use_pretrained = True)
    joint_order = joint_order1
    joint_connexion = joint_connexion1

elif model_name == CNN and dataset ==1 :
    CNNI = CNN()
    joint_order = joint_order1
    joint_connexion = joint_connexion1

else :
    CNNI = CNN()
    join_order = joint_order2 
    joint_connexion = joint_connexion2

'''
    








CNNI, input_size = get_pretrained_models(model_name = 'resnet', num_classes= 32, freeze_prior = False, 
                                            use_pretrained = True)

CNNI = CNNI.to(device)

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()

#Test the model on a video already filmed
video_path = "video_test.MOV"
cap = cv2.VideoCapture(video_path)

# Récupération des dimensions de la vidéo d'entrée
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

# the output will be written to output.avi
out = cv2.VideoWriter(
    'output.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    frame_rate,
    (frame_width, frame_height)
)

'''
# open webcam video stream
cap = cv2.VideoCapture(0)

# the output will be written to output.avi
out = cv2.VideoWriter(
    'output.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (640, 480)
)
'''

while True:
    # reading the frame
    ret, frame = cap.read()

    # resizing for faster detection
    #frame = cv2.resize(frame, (640, 480))

    # detect people in the image
    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))
    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    for (xA, yA, xB, yB) in boxes:
        # Extraire la région détectée
        person_frame = frame[yA:yB, xA:xB]

        # Convertir en PIL Image pour la transformation
        person_image = Image.fromarray(cv2.cvtColor(person_frame, cv2.COLOR_BGR2RGB))

        # Prédire les joints
        '''
        if dataset == 1 :
            pred_points_x, pred_points_y = predict1(model=CNNI, SavingName="./CNN-pose-resnet.ckpt0.ckpt", image=person_image)
        '''
        pred_points_x, pred_points_y = predict(model=CNNI, SavingName="checkpoints/CNN-pose-Resnet50_dataset2.ckpt", image=person_image)
        
        
        # Ajuster les points prédits à la taille originale de la boîte
        scale_x = (xB - xA) / 128  # Facteur d'échelle pour les x
        scale_y = (yB - yA) / 128  # Facteur d'échelle pour les y

        pred_points_x = np.array(pred_points_x) * scale_x + xA
        pred_points_y = np.array(pred_points_y )* scale_y + yA

        #print(pred_points)

        # Afficher les points prédits
        for k in range(len(pred_points_y)):
            cv2.circle(frame, (int(pred_points_x[k]), int(pred_points_y[k])), 4, (0, 0, 255), -1)
            cv2.putText(frame, joint_order[k], (int(pred_points_x[k]), int(pred_points_y[k])), 1, 1, (0, 0, 0))

        # Affichez les boîtes détectées
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
        # Tracer une ligne entre les deux points sur chaque frame
        for connexion in joint_connexion :
            cv2.line(
                frame,
                (int(pred_points_x[connexion[0]]),int(pred_points_y[connexion[0]])),
                (int(pred_points_x[connexion[1]]),int(pred_points_y[connexion[1]])),
                (0, 255, 0),
                thickness = 2
            )
    # Write the output video
    out.write(frame.astype('uint8'))
    

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
