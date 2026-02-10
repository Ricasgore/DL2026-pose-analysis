import torch
import torch.nn as nn
from utils import set_parameter_requires_grad
from config import device
import torch.nn.functional as F

class FC(nn.Module):
    def __init__(self,inputNode=561,hiddenNode = 256, outputNode=1):   
        super(FC, self).__init__()     
        #Define Hyperparameters
        self.inputLayerSize = inputNode
        self.outputLayerSize = outputNode
        self.hiddenLayerSize = hiddenNode
        
        # weights
        self.Linear1 = nn.Linear(self.inputLayerSize, self.hiddenLayerSize)
        self.Linear2 = nn.Linear(self.hiddenLayerSize, self.outputLayerSize)
        
    def forward(self, X):
        
        self.z2 = self.Linear1(X) # 3 X 3 ".dot" does not broadcast in PyTorch
        self.a2 = self.sigmoid(self.z2) # activation function
        self.z3 = self.Linear2(self.a2)
        return self.z3
        
    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+torch.exp(-z))
    
    def loss(self, yHat, y):
        J = 0.5*sum((y-yHat)**2)
        

class CNN(nn.Module):
    def __init__(self, num_classes=1):
        super(CNN, self).__init__()

        self.conv11 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.conv12 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        self.fc = nn.Linear(64 * 32 * 32, num_classes)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2)

        self.relu = nn.ReLU()

    def forward(self, x):
        out11 = self.maxpool(self.relu(self.conv11(x)))
        
        #print('out1' , out11.shape)
        out12 = self.maxpool(self.relu(self.conv12(out11)))
        
        #print('out2', out12.shape)
        
        out = out12.reshape(out12.size(0), -1)
        out = self.fc(out)

        return out
    
    

class CNN1D(nn.Module):
    def __init__(self, num_classes=1):
        super(CNN1D, self).__init__()

        self.conv11 = nn.Conv1d(1, 16, kernel_size=256, stride=1, padding=2)
        self.conv12 = nn.Conv1d(16, 32, kernel_size=128, stride=1, padding=1)

        self.fc = nn.Linear(32 * 2936, num_classes)
        #self.fc = nn.Linear(32 * 4, num_classes)

        self.maxpool = nn.MaxPool1d(kernel_size=16, stride=2)

        self.relu = nn.ReLU()

    def forward(self, x):
        #print(x.shape)
        out11 = self.maxpool(self.relu(self.conv11(x)))
        
        #print(out11.shape)
        out12 = self.maxpool(self.relu(self.conv12(out11)))
        
        #print(out12.shape)
        
        out = out12.reshape(out12.size(0), -1)
        out = self.fc(out)

        return out
    

def get_pretrained_models(model_name = 'resnet', num_classes = 2, freeze_prior = True, use_pretrained=True):
    
    from torchvision import models
    from utils import set_parameter_requires_grad
    
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet50"""
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, freeze_prior)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, freeze_prior)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, freeze_prior)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, freeze_prior)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, freeze_prior)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, freeze_prior)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()
        
    if True:
        from torchsummary import summary
        model_ft = model_ft.to(device)  # Assurez-vous que le modèle est sur le bon appareil
        summary(model_ft, (3, input_size, input_size), device=device.type)


    return model_ft.to(device), input_size

class CNNLSTM(nn.Module):
    def __init__(self, num_classes=101):
        super(CNNLSTM, self).__init__()
        
        self.cnn = CNN(num_classes=num_classes)
        self.cnn.fc = nn.Identity()
        
        self.cnn_output_size = 64 * 32 * 32 
        
        self.lstm = nn.LSTM(input_size=self.cnn_output_size, 
                            hidden_size=256, 
                            num_layers=2, 
                            batch_first=True)
        
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x_3d):
        batch_size, time_steps, C, H, W = x_3d.size()
        
        c_in = x_3d.reshape(batch_size * time_steps, C, H, W)
        
        c_out = self.cnn(c_in)
        
        r_in = c_out.reshape(batch_size, time_steps, -1)
        
        lstm_out, _ = self.lstm(r_in)
        
        last_output = lstm_out[:, -1, :]
        
        x = F.relu(self.fc1(last_output))
        x = self.fc2(x)
        
        return x

from torchvision import models
from torchvision.models import ResNet50_Weights

class ResNet50LSTM(nn.Module):
    def __init__(self, num_classes=101, hidden_size=512, num_layers=2):
        super(ResNet50LSTM, self).__init__()
        
        # 1. Backbone: ResNet-50 pré-entraîné
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        #set_parameter_requires_grad(resnet, True)  # Geler les poids pré-entraînés
        
        # On retire la couche de classification (fc)
        # resnet.fc.in_features est 2048 pour ResNet-50
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # 2. Modélisation Temporelle: LSTM
        # input_size=2048 (features venant de ResNet), hidden_size=512
        self.lstm = nn.LSTM(input_size=2048, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=True)
        
        # 3. Classificateur final
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x arrive en : [Batch, Channels (3), Time (16), H, W]
        
        # 1. On permute pour mettre le temps en 2ème position
        x = x.permute(0, 2, 1, 3, 4) # [Batch, Time, Channels, H, W]
        
        # 2. On récupère les dimensions
        batch_size, time_steps, C, H, W = x.size()
        
        # 3. On utilise reshape au lieu de view (plus robuste après un permute)
        # Ou alors on ajoute .contiguous() avant le .view()
        x = x.reshape(batch_size * time_steps, C, H, W)
        
        # 4. Extraction de caractéristiques spatiales
        with torch.no_grad(): 
            x = self.feature_extractor(x) 
        
        # 5. On aplatit et on redonne la forme pour le LSTM
        x = torch.flatten(x, 1) 
        x = x.reshape(batch_size, time_steps, -1) 
        
        # 6. Passage dans le LSTM
        out, (hn, cn) = self.lstm(x)
        
        # 7. Sortie finale
        x = self.fc(out[:, -1, :]) 
        return x
    
    # --- MODÈLE DE FUSION ---
class TwoStreamFusion(nn.Module):
    def __init__(self, num_classes=101):
        super(TwoStreamFusion, self).__init__()
        
        # --- BRANCHE 1 : VIDÉO (RGB) ---
        # On réutilise ton architecture ResNet50LSTM
        self.video_stream = ResNet50LSTM(num_classes=num_classes)
        # ASTUCE : On remplace la dernière couche (classifieur) par une identité
        # pour récupérer le vecteur de caractéristiques (taille 512) au lieu des classes.
        self.video_stream.fc = nn.Identity() 
        
        # --- BRANCHE 2 : POSE (Squelette) ---
        # Entrée : 38 (19 points x 2 coordonnées x,y)
        self.pose_lstm = nn.LSTM(input_size=38, hidden_size=256, num_layers=2, batch_first=True, dropout=0.3)
        
        # --- FUSION ---
        # Entrée combinée : 512 (Vidéo) + 256 (Pose) = 768
        self.classifier = nn.Sequential(
            nn.Linear(512 + 256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x_video, x_pose):
        """
        x_video : [Batch, 3, 16, 224, 224]
        x_pose  : [Batch, 16, 38]
        """
        
        # 1. Passage dans la branche Vidéo
        # v_feat shape : [Batch, 512]
        v_feat = self.video_stream(x_video)
        
        # 2. Passage dans la branche Pose
        # Le LSTM attend [Batch, Time, Features] -> C'est déjà bon !
        _, (h_n, _) = self.pose_lstm(x_pose)
        # On prend le dernier état caché : [Batch, 256]
        p_feat = h_n[-1]
        
        # 3. Concaténation (Fusion)
        # combined shape : [Batch, 768]
        combined = torch.cat((v_feat, p_feat), dim=1)
        
        # 4. Classification finale
        output = self.classifier(combined)
        
        return output