import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.video import r3d_18, R3D_18_Weights
import numpy as np
import os
import matplotlib.pyplot as plt
from config import device
from datasetMatisseUCF import *
from modelsMatisse import *
from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassF1Score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch.nn.functional as F

# IMPORTS FROM YOUR SPECIFIC FILE
from datasetMatisseUCF import UCF101Dataset 

# CONFIGURATION
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model=None, SavingName=None, train_loader=None, val_loader=None, optimizer=None, num_epochs=50):
    print('Start Training on UCF101 (Video)...')
    
    # Classification Loss
    criterion = nn.CrossEntropyLoss() 
    losses = []
    total_step = len(train_loader)

    for epoch in range(num_epochs):
        model.train() # Set model to training mode
        for i, (videos, labels) in enumerate(train_loader):

            # videos shape: [Batch, 3, 16, 112, 112] (Correct for 3D ResNet)
            videos = videos.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(videos) # Output shape: [Batch, 101]
            loss = criterion(outputs, labels)
            # print(outputs, labels)
            # exit()
            
            # Backward and optimize
            optimizer.zero_grad()   
            loss.backward()
            optimizer.step()

            # Logging
            if (i + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
                losses.append(loss.item())
                
                # Save loss curve data
                if not os.path.exists('./loss_save_during_training'):
                    os.makedirs('./loss_save_during_training')
                np.save('./loss_save_during_training/loss_UCF101.npy', losses)

        # Validation every 5 epochs
        if (epoch + 1) % 5 == 0:
            validate(model, val_loader)
            
            # Save Checkpoint
            directory = os.path.dirname(SavingName)
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save(model.state_dict(), SavingName)
            print(f"Checkpoint saved to {SavingName}")

def validate(model, val_loader):
    print("Validating...")
    model.eval() # Set model to eval mode
    correct = 0
    total = 0
    
    with torch.no_grad():
        for videos, labels in val_loader:
            videos = videos.to(device)
            labels = labels.to(device)

            outputs = model(videos)
            
            # Get the predicted class (index with max probability)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    acc = 100 * correct / total
    print(f'Validation Accuracy: {acc:.2f} %')
    model.train() # Switch back to train mode

def test(model=None, SavingName=None, test_loader=None):
    # Load the model weights
    print(f"Loading model from {SavingName}")
    model.load_state_dict(torch.load(SavingName))
    model.eval()

    # Initialisation des métriques PyTorch
    # num_classes=101 pour UCF101
    f1_metric = MulticlassF1Score(num_classes=101, average='macro').to(device)
    conf_matrix_metric = MulticlassConfusionMatrix(num_classes=101).to(device)
    
    correct = 0
    total = 0
    
    print("Testing on full test set...")
    with torch.no_grad():
        for i, (videos, labels) in enumerate(test_loader):
            videos = videos.to(device)
            labels = labels.to(device)
            
            outputs = model(videos)
            _, predicted = torch.max(outputs.data, 1)

            # Mise à jour des métriques à chaque batch
            f1_metric.update(predicted, labels)
            conf_matrix_metric.update(predicted, labels)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Final Test Accuracy: {100 * correct / total:.2f} %')
    final_f1 = f1_metric.compute()
    final_cm = conf_matrix_metric.compute() # C'est ta matrice de confusion
    print(final_cm.shape)
    print(f'Final F1-Score: {final_f1:.4f}')
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


def predict_top5_video(model, video_path, dataset_obj, device):
    model.eval()
    with torch.no_grad():
        # 1. Chargement (Logique UCF101)
        video_tensor = dataset_obj._load_frames(video_path) 
        
        # 2. Batch & Permutation [1, Time, Channels, H, W]
        # Attention: On garde la permutation qui corrigeait ton erreur précédente
        video_tensor = video_tensor.unsqueeze(0).to(device)
        
        # 3. Forward pass
        outputs = model(video_tensor)
        
        # 4. Conversion en probabilités (Softmax)
        probs = F.softmax(outputs, dim=1)
        
        # 5. Récupérer les 5 meilleurs scores et leurs indices
        # topk renvoie (valeurs, indices)
        top5_probs, top5_indices = torch.topk(probs, k=5)
        
        # 6. Construire la liste des résultats
        results = []
        for i in range(5):
            idx = top5_indices[0][i].item()
            score = top5_probs[0][i].item()
            
            # Récupérer le nom de la classe
            if hasattr(dataset_obj, 'idx_to_class'):
                class_name = dataset_obj.idx_to_class[idx]
            else:
                class_name = f"Class {idx}"
                
            results.append((class_name, score))
            
        return results


   

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
    
    # Path to your data on the server
    path = '/home/amine_tsp/DL2026/Datasets/UCF101'

    if operation == 0: # TRAIN MODE
        print("Loading Train Dataset...")
        PoseTrain = UCF101Dataset(baseDir=path, transform=tr, theType='train', split=1)
        train_loader = torch.utils.data.DataLoader(dataset=PoseTrain, batch_size=batch_size, shuffle=True, num_workers=4)
        """"print(len(train_loader))
        exit()"""
        # Using Test set as Validation during training
        PoseVal = UCF101Dataset(baseDir=path, transform=tr, theType='test', split=1)
        val_loader = torch.utils.data.DataLoader(dataset=PoseVal, batch_size=batch_size, shuffle=False, num_workers=4)
        
        # Run Training
        optimizer = torch.optim.Adam(CNNI.parameters(), lr=learning_rate)
        train(model=CNNI, SavingName=SavingName, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, num_epochs=num_epochs)

    elif operation == 1: # TEST MODE
        print("Loading Test Dataset...")
        PoseTest = UCF101Dataset(baseDir=path, transform=tr, theType='test', split=1)
        test_loader = torch.utils.data.DataLoader(dataset=PoseTest, batch_size=batch_size, shuffle=False, num_workers=4)
        """print(len(test_loader))
        exit()"""
        # Run Testing
        test(model=CNNI, SavingName=SavingName, test_loader=test_loader)
    
    elif operation == 2: #predire
        print("--- Mode Prédiction (Top 5) ---")
        
        # 1. Charger les poids
        if os.path.exists(SavingName):
            print(f"Chargement du modèle : {SavingName}")
            CNNI.load_state_dict(torch.load(SavingName))
        else:
            print("ERREUR : Checkpoint introuvable.")
            exit()

        # 2. Dataset pour les noms de classes
        PoseTest = UCF101Dataset(baseDir=path, transform=tr, theType='test', split=1)
        
        # 3. Vidéo cible
        video_path = "/home/amine_tsp/DL2026/Datasets/UCF101/frames_img_224/Basketball/v_Basketball_g01_c01"
        
        try:
            # Appel de la nouvelle fonction
            top5_results = predict_top5_video(CNNI, video_path, PoseTest, device)
            
            print("\n" + "="*40)
            print(f"Vidéo analysée : {os.path.basename(video_path)}")
            print("="*40)
            print(f"{'RANG':<5} | {'CLASSE':<20} | {'CONFIANCE':<10}")
            print("-" * 40)
            
            for rank, (name, score) in enumerate(top5_results, 1):
                # On affiche le score en pourcentage (score * 100)
                print(f"{rank:<5} | {name:<20} | {score*100:.2f}%")
                
            print("="*40 + "\n")
            
        except Exception as e:
            print(f"Erreur prédiction : {e}")
            # C'est souvent utile d'imprimer le traceback complet en cas de bug
            import traceback
            traceback.print_exc()

    # Chemin vers votre fichier
    chemin = '/home/amine_tsp/DL2026/Method_Automatic_pose_recognition/checkpoints/R3D18_UCF101.ckpt'

    # Charger le fichier (map_location='cpu' est important si vous n'avez pas de GPU sur la machine qui lit)
    donnees = torch.load(chemin, map_location=torch.device('cpu'))

    # 1. Voir quelles sont les clés (noms des couches)
    #print("Liste des couches enregistrées :")
    #print(donnees.keys())

    # 2. Voir la forme d'un poids spécifique (ex: la première couche de convolution)
    print("\nForme du tenseur 'stem.0.weight' :")
    print(donnees['stem.0.weight'].shape) 
    # Devrait afficher quelque chose comme torch.Size([64, 3, 3, 7, 7])

    # 3. Voir les valeurs réelles de la dernière couche (classification)
    print("\nPoids de la dernière couche (fc.weight) :")
    print(donnees['fc.weight'])