import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Imports locaux
from datasetActivityNet import ActivityNetDataset
from models import CNNLSTM 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, SavingName, train_loader, val_loader, optimizer, num_epochs):
    print(f'Start Training... Device: {device}')
    
    # Loss pour Multi-Label / Classification One-Hot
    criterion = nn.BCEWithLogitsLoss() 
    
    # Gestion dossiers sauvegarde
    if not os.path.exists(os.path.dirname(SavingName)): os.makedirs(os.path.dirname(SavingName))
    
    loss_dir = '/home/amine_tsp/DL2026/Method_Automatic_pose_recognition/loss_save_during_training/'
    if not os.path.exists(loss_dir): os.makedirs(loss_dir)
    loss_file_path = os.path.join(loss_dir, 'loss_ActivityNet.npy')

    # Chargement historique loss
    losses = []
    if os.path.exists(loss_file_path):
        try: losses = list(np.load(loss_file_path, allow_pickle=True))
        except: pass
        
    total_step = len(train_loader)

    for epoch in range(num_epochs):
        model.train() 
        for i, (videos, labels, _) in enumerate(train_loader):
            
            # Transfert GPU
            videos = videos.to(device) # [Batch, Time, C, H, W]
            labels = labels.to(device) 

            # Forward
            optimizer.zero_grad()
            outputs = model(videos) 
            loss = criterion(outputs, labels)
            
            # Backward
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            # Log tous les 10 steps
            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')
                np.save(loss_file_path, losses)
        
        # Validation rapide et Sauvegarde Checkpoint tous les 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), SavingName)
            print(f"Checkpoint saved to {SavingName}")

def plott():
    path = '/home/amine_tsp/DL2026/Method_Automatic_pose_recognition/loss_save_during_training/loss_ActivityNet.npy'
    if os.path.exists(path):
        plt.plot(np.load(path, allow_pickle=True))
        plt.title("Training Loss")
        plt.show()

if __name__ == '__main__':
    # --- HYPERPARAMETRES ---
    batch_size = 4        # Petit batch car images HD (224x224) + Vidéo
    learning_rate = 1e-4 
    num_epochs = 999999999
    clip_len = 16         # Nombre de frames par vidéo
    
    # 0 = Train, 2 = Plot
    operation = 0 

    # --- PATHS ---
    json_path = '/home/amine_tsp/DL2026/Datasets/ActivityNet/Evaluation/data/activity_net.v1-3.min.json'
    video_root = '/home/amine_tsp/DL2026/Datasets/ActivityNet/raw_clips'
    
    SavingName = '/home/amine_tsp/DL2026/Method_Automatic_pose_recognition/checkpoints/ActivityNet_CNNLSTM.ckpt'

    if operation == 2:
        # --- 1. SETUP DATASET ---
        print("Loading Dataset...")
        # On demande explicitement du 224x224 pour la qualité
        train_ds = ActivityNetDataset(json_path, video_root, split='training', clip_len=clip_len, frame_size=(224, 224))
        
        if len(train_ds) == 0:
            print("ERREUR CRITIQUE: Aucune vidéo trouvée. Vérifie tes dossiers.")
            exit()
            
        num_classes = len(train_ds.classes)
        print(f"Classes détectées: {num_classes}")
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)

        # --- 2. SETUP MODEL ---
        print("Init Model...")
        model = CNNLSTM(num_classes=num_classes).to(device)
        
        # Reprise training si existe
        if os.path.exists(SavingName):
            print(f"Reprise du checkpoint : {SavingName}")
            model.load_state_dict(torch.load(SavingName))

        # --- 3. TRAIN ---
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        train(model, SavingName, train_loader, train_loader, optimizer, num_epochs)
        
    elif operation == 2:
        plott()