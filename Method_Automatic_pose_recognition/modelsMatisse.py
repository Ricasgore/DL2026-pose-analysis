import torch
import torch.nn as nn
from config import device

class FC(nn.Module):
    def __init__(self,inputNode=561,hiddenNode = 256, outputNode=1):   
        super(FC, self).__init__()     
        #Define Hyperparameters
        self.inputLayerSize = inputNode
        self.outputLayerSize = outputNode
        self.hiddenLayerSize = hiddenNode
        
        self.relu = nn.ReLU()

        # weights
        self.Linear1 = nn.Linear(self.inputLayerSize, self.hiddenLayerSize)
        self.Linear2 = nn.Linear(self.hiddenLayerSize, self.outputLayerSize)
        
    def forward(self, X):
        #print(X.shape,self.Linear1)
        X = torch.flatten(X, start_dim=1)

        #print(X.shape,self.Linear1)
        self.z2 = self.Linear1(X) # 3 X 3 ".dot" does not broadcast in PyTorch
        self.a2 = self.relu(self.z2) # activation function
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
        """ Resnet50
        """
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
