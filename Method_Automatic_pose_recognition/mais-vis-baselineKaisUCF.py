import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import os
import matplotlib.pyplot as plt
from config import device
from datasetMatisseUCF import UCF101Dataset 
import seaborn as sns
from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassF1Score


# --- IMPORT DE VOTRE MODÈLE ---
from models import CNNLSTM 

def train(model=None, SavingName=None, train_loader=None, val_loader=None, optimizer=None, num_epochs=50):
    print('Start Training on UCF101 (CNN-LSTM)...')
    
    criterion = nn.CrossEntropyLoss() 
    losses = []
    total_step = len(train_loader)

    for epoch in range(num_epochs):
        model.train() 
        for i, (videos, labels) in enumerate(train_loader):
            
            # 1. Charger sur GPU
            videos = videos.to(device)
            labels = labels.to(device)

            # 2. PERMUTATION DES DIMENSIONS
            # Le DataLoader sort : [Batch, Channel, Time, Height, Width] (ex: 16, 3, 16, 128, 128)
            # Votre CNNLSTM veut : [Batch, Time, Channel, Height, Width] (ex: 16, 16, 3, 128, 128)
            videos = videos.permute(0, 2, 1, 3, 4)

            # 3. Forward pass
            outputs = model(videos) 
            loss = criterion(outputs, labels)

            # 4. Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            if (i + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
                losses.append(loss.item())
                
                # Save loss curve
                save_path = '/home/amine_tsp/DL2026/Method_Automatic_pose_recognition/loss_save_during_training'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                np.save(os.path.join(save_path, 'loss_UCF101_CNNLSTM.npy'), losses)

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
    model.eval() 
    correct = 0
    total = 0
    
    with torch.no_grad():
        for videos, labels in val_loader:
            videos = videos.to(device)
            labels = labels.to(device)

            # PERMUTATION (Même chose qu'au train)
            videos = videos.permute(0, 2, 1, 3, 4)

            outputs = model(videos)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    acc = 100 * correct / total
    print(f'Validation Accuracy: {acc:.2f} %')
    model.train() 

def plott():
    a = np.load('/home/amine_tsp/DL2026/Method_Automatic_pose_recognition/loss_save_during_training/loss_UCF101_CNNLSTM.npy', allow_pickle=True)
    plt.plot(a)
    plt.savefig('/home/amine_tsp/DL2026/Method_Automatic_pose_recognition/loss_graph_plot/loss_UCF101_CNNLSTM.png')
    plt.close()
    print('finish')

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
            videos = videos.permute(0, 2, 1, 3, 4)
            
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
    plt.savefig("/home/amine_tsp/DL2026/Method_Automatic_pose_recognition/matrice_confusion_Kais.png")
    plt.close()
    print(f"Matrice de confusion sauvegardée sous : {'/home/amine_tsp/DL2026/Method_Automatic_pose_recognition/matrice_confusion.png'}")

import torch.nn.functional as F # Assure-toi d'avoir cet import

def predict_top5_video(model, video_path, dataset_obj, device):
    model.eval()
    with torch.no_grad():
        # 1. Chargement (Logique UCF101)
        video_tensor = dataset_obj._load_frames(video_path) 
        
        # 2. Batch & Permutation [1, Time, Channels, H, W]
        # Attention: On garde la permutation qui corrigeait ton erreur précédente
        video_tensor = video_tensor.unsqueeze(0).to(device)
        video_tensor = video_tensor.permute(0, 2, 1, 3, 4)
        
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
    # CNN-LSTM consomme plus de mémoire que ResNet3D parfois (à cause des gradients stockés sur le temps)
    # Si "Out of Memory", réduisez batch_size à 8 ou 4.
    batch_size = 16    
    learning_rate = 1e-4 
    num_epochs = 999999999
    num_classes = 101
    
    # 0 = Train, 1 = Test
    operation = 4

    # --- 1. SETUP MODEL ---
    print("Initializing Custom CNN-LSTM...")
    
    # Instanciation de VOTRE modèle
    CNNI = CNNLSTM(num_classes=num_classes)
    CNNI = CNNI.to(device)
    
    # Nom de sauvegarde différent pour ne pas écraser le ResNet
    SavingName = '/home/amine_tsp/DL2026/Method_Automatic_pose_recognition/checkpoints/CNNLSTM_UCF101_Kais.ckpt'

    # --- 2. SETUP DATASET ---
    # CRITIQUE : Resize à (128, 128) pour correspondre à 64*32*32 dans votre model.py
    tr = transforms.Compose([
        transforms.Resize((128, 128)), 
        transforms.ToTensor()
    ])
    
    path = '/home/amine_tsp/DL2026/Datasets/UCF101'

    if operation == 0: # TRAIN
        print("Loading Train Dataset...")
        PoseTrain = UCF101Dataset(baseDir=path, transform=tr, theType='train', split=1)
        train_loader = torch.utils.data.DataLoader(dataset=PoseTrain, batch_size=batch_size, shuffle=True, num_workers=4)
        
        PoseVal = UCF101Dataset(baseDir=path, transform=tr, theType='test', split=1)
        val_loader = torch.utils.data.DataLoader(dataset=PoseVal, batch_size=batch_size, shuffle=False, num_workers=4)
        
        optimizer = torch.optim.Adam(CNNI.parameters(), lr=learning_rate)
        train(model=CNNI, SavingName=SavingName, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, num_epochs=num_epochs)

    elif operation == 1: # TEST
        print("Loading Test Dataset...")
        PoseTest = UCF101Dataset(baseDir=path, transform=tr, theType='test', split=1)
        test_loader = torch.utils.data.DataLoader(dataset=PoseTest, batch_size=batch_size, shuffle=False, num_workers=4)
        
        test(model=CNNI, SavingName=SavingName, test_loader=test_loader)
        
    elif operation == 2:
        plott()
    elif operation == 4: # PREDIRE TOP 5
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