import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from datasetAlex import *
from models import *
from utils import * 
from metrics import * 
from config import device
import os

def train(model = None,SavingName=None, train_loader = None, val_loader=None, optimizer = None):
    # training
    print('training')
    losses = []
    total_step = len(train_loader)
    #print(total_step)

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):

            #print(images.shape, labels.shape)

            images = images.to(device)
            labels = labels.to(device)

            labels_f = labels.reshape(labels.size(0), -1)
            
            # Forward pass
            outputs = model.forward(images)
            
            loss = torch.sqrt(torch.mean((outputs - labels_f) ** 2))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 2 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
                losses.append(loss.item())
                np.save('/home/amine_tsp/DL2026/Method_Automatic_pose_recognition/loss_save_during_training/lossCNN_Basic_dataset1.npy',losses)
                
            
            image = images.cpu().numpy()

            #plotting
            """
            for k in range(images.shape[0]):
                labels_f_k = np.array(labels_f[k]).reshape(14,3)
                outputs_k = np.array(outputs[k].detach().numpy()).reshape(14,3)
                image = np.array(images[k])
                image = image.transpose(1, 2, 0) #(128, 128, 3)
                plt.imshow(image)
                plt.scatter(outputs_k[:,0], outputs_k[:,1], color='red')
                plt.scatter(labels_f_k[:,0], labels_f_k[:,1], color='blue')
                plt.legend()
                plt.title("test")
                plt.show()
                    #print(pred)
            """   
            
            #do validations every 10 epoch 
            if i%10 == 0:
                with torch.no_grad():
                    
                    model.eval()        
                    pred,gt = [],[]
                    
                    for imagesV, labelsV in val_loader:
                        
                        imagesV = imagesV.to(device)
                        labelsV = labelsV.to(device)

                        labelsV = labelsV.reshape(labelsV.size(0), -1)
                        
                        # Forward pass
                        outputsV = model(imagesV).round()

                        #ploting                    
                        """
                        for k in range(imagesV.shape[0]):
                            gt_k = np.array(labelsV[k]).reshape(14,3)
                            pred_k = np.array(outputsV[k]).reshape(14,3)
                            image = np.array(imagesV[k])
                            image = image.transpose(1, 2, 0) #(128, 128, 3)
                            print(pred_k)
                            print(gt_k)
                            plt.imshow(image)
                            plt.scatter(pred_k[:,0], pred_k[:,1], color='red')
                            plt.scatter(gt_k[:,0], gt_k[:,1], color='blue')
                            plt.legend()
                            plt.title("test")
                            plt.show()
                        """

                        gt.extend(labelsV.cpu().numpy())
                        pred.extend(outputsV.cpu().numpy())
                    
                    gt = np.asarray(gt,np.float32)
                    pred = np.asarray(pred)

                    imagesV = imagesV.cpu().numpy()
                                    
                    print('loss : ', np.sum(np.sqrt(np.square(gt - pred))))
                    
                model.train()

                checkDirMake(os.path.dirname(SavingName))
                torch.save(model.state_dict(), SavingName)

    
def plott():
    a = np.load('/home/amine_tsp/DL2026/Method_Automatic_pose_recognition/loss_save_during_training/lossCNN_Basic_dataset1.npy', allow_pickle=True)
    plt.plot(a)
    plt.savefig('/home/amine_tsp/DL2026/Method_Automatic_pose_recognition/loss_graph_plot/lossCNN_Basic_dataset1.png')
    plt.close()
    print('finish')

def test(model = None,SavingName=None, test_loader=None):
    #load the model
    print('start testing')
    model.load_state_dict(torch.load(SavingName))
    print('data loaded')
    model.eval()  # eval mode
    print('model in eval mode')
    with torch.no_grad():
        print('testing')
        pred,gt = [],[]
        
        for i, (images, labels) in enumerate(test_loader):

            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images).round()
            labels = labels.reshape(labels.size(0), -1)
            
            gt.extend(labels.squeeze().cpu().numpy())
            pred.extend(outputs.squeeze().cpu().numpy())

            images = images.cpu().numpy()

            #plot the prediction and save
            """
            for k in range(images.shape[0]):
                pred_k = pred[k]
                gt_k = gt[k]

                pred_k_x = []
                pred_k_y = []
                for j in range(len(pred_k)):
                    if j%3==0:
                        pred_k_x.append(pred_k[j])
                    if j%3==1 : 
                        pred_k_y.append(pred_k[j])
                plt.imshow(images[k].transpose(1, 2, 0))
                plt.scatter(pred_k_x,pred_k_y, color='red')
                plt.scatter(gt_k[:,0], gt_k[:,1], color='blue')
                plt.title("test")
                plt.savefig(f'./test_plot_CNN_dataset1/im{i}.png')
                plt.close()
            """
        

        gt = np.asarray(gt,np.float32)
        pred = np.asarray(pred)

        rmse = RMSE(gt,pred)
        print(rmse)

        # .argmax(dim=1) picks the class with the highest score for each image
        print('Test Accuracy of the model on test images: {} %'.format(accuracy(pred.argmax(axis=1), gt.argmax(axis=1))))

def predict_single_video(model, video_path, dataset_obj, device):
    model.eval()
    with torch.no_grad():
        # 1. Charger et transformer les frames (on réutilise ta logique interne)
        # On simule ce que fait __getitem__
        video_tensor = dataset_obj._load_frames(video_path) 
        
        # 2. Ajouter la dimension de Batch : [C, T, H, W] -> [1, C, T, H, W]
        video_tensor = video_tensor.unsqueeze(0).to(device)
        
        # 3. Passer dans le modèle
        outputs = model(video_tensor)
        _, predicted = torch.max(outputs, 1)
        
        # 4. Traduire l'index en nom de classe
        class_idx = predicted.item()
        class_name = dataset_obj.idx_to_class[class_idx]
        
        return class_name, class_idx

    # 1. Convertir le tenseur PyTorch en tableau numpy
    if hasattr(final_cm, 'cpu'):
        final_cm = final_cm.cpu().numpy()
    plt.figure(figsize=(25, 20)) # Grande taille pour 101 classes
    sns.heatmap(final_cm, annot=False, cmap='Blues', fmt='g')
    plt.title(f'Matrice de Confusion - Epoch {50}')
    plt.xlabel('Classes Prédites')
    plt.ylabel('Classes Réelles')
    
    # 4. Sauvegarde
    plt.savefig("/home/amine_tsp/DL2026/Method_Automatic_pose_recognition/matrice_confusion.png")
    plt.close()
    print(f"Matrice de confusion sauvegardée sous : {'/home/amine_tsp/DL2026/Method_Automatic_pose_recognition/matrice_confusion.png'}")


if __name__ == '__main__':

    # --- HYPERPARAMETERS ---
    batch_size = 12  # 16 videos per batch (Adjust if you get "Out of Memory")
    learning_rate = 1e-5
    num_epochs = 5
    num_classes = 101  # UCF101 has 101 actions
    
    # 0 = Train, 1 = Test, 2 = predire
    operation = 2

    # --- 1. SETUP MODEL ---
    print("Initializing 3D ResNet-18...")
    # Load Pre-trained weights (trained on Kinetics-400)
    CNNI = FC(inputNode=2408448 ,hiddenNode = 256, outputNode=101).to(device)
    CNNI = CNNI.to(device)
    SavingName = '/home/amine_tsp/DL2026/Method_Automatic_pose_recognition/checkpoints/FC_UCF101_Matisse.ckpt'

    # --- 2. SETUP DATASET ---
    # Standard transforms for ResNet video models (Resize to 112x112 or 128x128)
    tr = transforms.Compose([
        #transforms.Resize((112, 112)), 
        transforms.ToTensor()
    ])
        
if __name__ == '__main__':

    #########################################
    # model = 0  => own cnn (not pretrained)#
    # model = 1  => resnet                  #
    # dataset = 1  => first dataset         #
    # dataset = 2  => second dataset        #
    #########################################
    (model, dataset) = (0,1)
    print(f'selected model: {model}, dataset: {dataset}')
    

    if (model,dataset) == (0,1):
        print('using basic CNN model')
        CNNI = CNN(num_classes=42).to(device)
        CNNI = CNNI.to(device)
        SavingName = '/home/amine_tsp/DL2026/Method_Automatic_pose_recognition/checkpoints/CNN-pose-Basic_dataset1.ckpt'
    elif (model, dataset) == (0,2):
        CNNI = CNN(num_classes=32).to(device)
        CNNI = CNNI.to(device)
        SavingName = '/home/amine_tsp/DL2026/Method_Automatic_pose_recognition/checkpoints/CNN-pose-Basic_dataset2.ckpt'
    elif (model, dataset) == (1,1):
        sel = 0
        list_models = ['resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet', 'inception']
        MODEL = list_models[sel]
        CNNI, input_size = get_pretrained_models(model_name = list_models[sel], num_classes= 42, freeze_prior = False, 
                                                use_pretrained = True)
        CNNI = CNNI.to(device)
        SavingName = './checkpoints/CNN-pose-Resnet50_dataset1.ckpt'
    elif (model,dataset) == (1,2):
        sel = 0
        list_models = ['resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet', 'inception']
        MODEL = list_models[sel]
        CNNI, input_size = get_pretrained_models(model_name = list_models[sel], num_classes= 32, freeze_prior = False, 
                                                use_pretrained = True)
        CNNI = CNNI.to(device)
        SavingName = './checkpoints/CNN-pose-Resnet50_dataset2.ckpt'
 
    
    tr = transforms.Compose([
        transforms.ToTensor()
        ])
    
    batch_size = 50
    
    if dataset == 1 :
        print('using dataset 1')
        PoseTrain = LSPPE(transform=tr, theType='train')
        train_loader = torch.utils.data.DataLoader(dataset=PoseTrain,
                                                batch_size=batch_size,
                                                shuffle=True)
        
        PoseVal = LSPPE(transform=tr, theType='val')
        val_loader = torch.utils.data.DataLoader(dataset=PoseVal,
                                                batch_size=batch_size,
                                                shuffle=True)

        PoseTest = LSPPE(transform=tr, theType='test')
        test_loader = torch.utils.data.DataLoader(dataset=PoseTest,
                                                batch_size=batch_size)
    elif dataset == 2 :
        PoseTrain = MPII(transform=tr, theType='train')
        train_loader = torch.utils.data.DataLoader(dataset=PoseTrain,
                                                batch_size=batch_size,
                                                shuffle=True)
        
        PoseVal = MPII(transform=tr, theType='val')
        val_loader = torch.utils.data.DataLoader(dataset=PoseVal,
                                                batch_size=batch_size,
                                                shuffle=True)

        PoseTest = MPII(transform=tr, theType='test')
        test_loader = torch.utils.data.DataLoader(dataset=PoseTest,
                                                batch_size=batch_size)
    
    learning_rate = .0001
    num_epochs = 999999999
    optimizer = torch.optim.Adam(CNNI.parameters(), lr=learning_rate)

    operation = 1
    
    if operation ==0 or operation==2: 
        print('start training')
        train(model = CNNI,SavingName=SavingName, train_loader = train_loader, val_loader=val_loader, optimizer = optimizer)
    if operation ==1 or operation==2: 
        print('start testing')
        test(model = CNNI,SavingName=SavingName, test_loader=test_loader)
    if operation == 3: #predire
        PoseTest = UCF101Dataset(baseDir=path, transform=tr, theType='test', split=1)
        test_loader = torch.utils.data.DataLoader(dataset=PoseTest, batch_size=batch_size, shuffle=False, num_workers=4)
        video_path = "/home/amine_tsp/DL2026/Datasets/UCF101/frames_img_224/Basketball/v_Basketball_g01_c01"
        nom_predit, id_predit = predict_single_video(CNNI, video_path, PoseTest, device)
        print(f"Verdict : {nom_predit} (ID: {id_predit})")
    if operation == 4:
        plott()
        
        