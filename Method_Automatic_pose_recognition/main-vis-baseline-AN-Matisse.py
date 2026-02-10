import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassF1Score

# Imports des modèles et config
from models import ResNet50LSTM
from config import device

# IMPORT DU DATASET ACTIVITYNET
from datasetActivityNet import ActivityNetDataset

# --- FONCTION DE PLOT ---
def plott():
    path = '/home/amine_tsp/DL2026/Method_Automatic_pose_recognition/loss_save_during_training/loss_AN_Alex.npy'
    if os.path.exists(path):
        data = np.load(path, allow_pickle=True)
        plt.figure(figsize=(10, 5))
        plt.plot(data, label='Training Loss')
        plt.title("ActivityNet Training Loss")
        plt.xlabel("Steps (x10)")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig('/home/amine_tsp/DL2026/Method_Automatic_pose_recognition/loss_graph_plot/loss_AN_Alex.png')
        plt.close()
        print('Graphique sauvegardé.')
    else:
        print(f"Fichier introuvable : {path}")

# --- FONCTION D'ENTRAÎNEMENT ---
def train(model=None, SavingName=None, train_loader=None, val_loader=None, optimizer=None, num_epochs=50):
    print(f'Start Training on ActivityNet (Device: {device})...')
    
    # ActivityNet est souvent Multi-class, on utilise CrossEntropy
    # Si tes labels sont One-Hot, utilise BCEWithLogitsLoss
    criterion = nn.CrossEntropyLoss() 
    
    losses = []
    total_step = len(train_loader)

    # Dossier pour sauvegarder la loss
    loss_save_path = '/home/amine_tsp/DL2026/Method_Automatic_pose_recognition/loss_save_during_training/loss_AN_Alex.npy'
    os.makedirs(os.path.dirname(loss_save_path), exist_ok=True)

    for epoch in range(num_epochs):
        model.train() # Mode entraînement
        
        for i, batch_data in enumerate(train_loader):
            # ActivityNetDataset renvoie souvent : (videos, labels, video_ids)
            # On déballe proprement
            videos = batch_data[0].to(device)
            labels = batch_data[1].to(device)

            # --- CORRECTION ICI ---
            # On passe de [Batch, Time, Channel, H, W] -> [Batch, Channel, Time, H, W]
            # On échange la dimension 1 (Time=16) et 2 (Channel=3)
            videos = videos.permute(0, 2, 1, 3, 4) 
            # ----------------------

            # Forward pass
            outputs = model(videos) 
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            if (i + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
                losses.append(loss.item())
                
                # Sauvegarde intermédiaire de la loss
                np.save(loss_save_path, losses)

        # Validation et Sauvegarde Checkpoint (tous les 1 epoch vu la taille du dataset, ou 5)
        if (epoch + 1) % 1 == 0: 
            # On sauvegarde à chaque epoch car ActivityNet est long
            directory = os.path.dirname(SavingName)
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save(model.state_dict(), SavingName)
            print(f"Checkpoint saved to {SavingName}")
            
            # Optionnel : Validation (peut être long sur ActivityNet)
            # validate(model, val_loader) 

def validate(model, val_loader):
    print("Validating...")
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_data in val_loader:
            videos = batch_data[0].to(device)
            labels = batch_data[1].to(device)

            # --- CORRECTION ICI AUSSI ---
            videos = videos.permute(0, 2, 1, 3, 4)
            # ----------------------------

            outputs = model(videos)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    acc = 100 * correct / total
    print(f'Validation Accuracy: {acc:.2f} %')
    model.train()

def test(model=None, SavingName=None, test_loader=None):
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
    plt.savefig("/home/amine_tsp/DL2026/Method_Automatic_pose_recognition/matrice_confusion.pngAN")
    plt.close()
    print(f"Matrice de confusion sauvegardée sous : {'/home/amine_tsp/DL2026/Method_Automatic_pose_recognition/matrice_confusion.ANpng'}")

# --- MAIN ---
if __name__ == '__main__':

    # --- HYPERPARAMETERS ---
    # Attention : ActivityNet utilise des images 224x224.
    # 16 frames * 224 * 224 est lourd pour la VRAM.
    # Si "Out of Memory", réduis le batch_size à 8 ou 4.
    batch_size = 8    
    learning_rate = 1e-4
    num_epochs = 50
    clip_len = 16 # Nombre de frames par vidéo (doit matcher ton modèle)

    # --- PATHS ---
    json_path = '/home/amine_tsp/DL2026/Datasets/ActivityNet/Evaluation/data/activity_net.v1-3.min.json'
    video_root = '/home/amine_tsp/DL2026/Datasets/ActivityNet/raw_clips'
    
    # Checkpoint de sauvegarde
    SavingName = '/home/amine_tsp/DL2026/Method_Automatic_pose_recognition/checkpoints/ResNet50LSTM_ActivityNet_Alex.ckpt'

    # 0 = Train, 2 = Plot Loss
    operation = 0

    if operation == 0: # TRAIN MODE
        print("Loading ActivityNet Dataset...")
        
        # Initialisation du Dataset
        # Note : frame_size=(224, 224) est important pour ResNet50
        train_ds = ActivityNetDataset(json_path, video_root, split='training', clip_len=clip_len, frame_size=(224, 224))
        
        # Vérification critique
        if len(train_ds) == 0:
            print("ERREUR : Le dataset est vide. Vérifie les chemins json_path et video_root.")
            exit()

        # Récupération automatique du nombre de classes (200 normalement)
        num_classes = len(train_ds.classes)
        print(f"Dataset chargé. {len(train_ds)} vidéos. {num_classes} classes détectées.")

        # DataLoader
        train_loader = torch.utils.data.DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
        
        # Validation Loader (On utilise 'validation' split du json)
        val_ds = ActivityNetDataset(json_path, video_root, split='validation', clip_len=clip_len, frame_size=(224, 224))
        val_loader = torch.utils.data.DataLoader(dataset=val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

        # --- SETUP MODEL ---
        print(f"Initializing ResNet-50 + LSTM Hybrid for {num_classes} classes...")
        model = ResNet50LSTM(num_classes=num_classes)
        model = model.to(device)

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Lancement
        train(model=model, SavingName=SavingName, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, num_epochs=num_epochs)
    
    elif operation == 1:
        # --- SETUP MODEL ---
        print(f"Initializing ResNet-50 + LSTM Hybrid for {num_classes} classes...")
        model = ResNet50LSTM(num_classes=num_classes)
        model = model.to(device)

        # Initialisation du Dataset
        # Note : frame_size=(224, 224) est important pour ResNet50
        test_ds = ActivityNetDataset(json_path, video_root, split='validation', clip_len=clip_len, frame_size=(224, 224))
        
        # DataLoader
        test_loader = torch.utils.data.DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=True, num_workers=4)
        



    elif operation == 2: # PLOT MODE
        plott()