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

def test(model=None, SavingName=None, test_loader=None, num_classes=200):
    print(f"Loading model from {SavingName}")
    
    # Chargement des poids
    try:
        model.load_state_dict(torch.load(SavingName))
    except FileNotFoundError:
        print(f"Erreur : Le fichier {SavingName} n'existe pas encore.")
        return

    model.eval()
    
    # Initialisation des métriques
    # 'task' est implicite avec les classes, mais on précise num_classes
    f1_metric = MulticlassF1Score(num_classes=num_classes, average='macro').to(device)
    conf_matrix_metric = MulticlassConfusionMatrix(num_classes=num_classes).to(device)
    
    correct = 0
    total = 0
    
    print("Testing on ActivityNet Validation Set...")
    
    with torch.no_grad():
        for i, batch_data in enumerate(test_loader):
            # 1. Récupération des données
            videos = batch_data[0].to(device)
            labels = batch_data[1].to(device)
            
            # --- CORRECTION ONE-HOT (AJOUTER CECI) ---
            # Si les labels sont en format [Batch, 200], on les convertit en [Batch]
            if labels.dim() > 1 and labels.size(1) > 1:
                _, labels = torch.max(labels, 1) # On prend l'index du 1
            # -----------------------------------------
            
            # --- CORRECTION DIMENSIONS (CRUCIAL) ---
            # [Batch, Time, Channel, H, W] -> [Batch, Channel, Time, H, W]
            videos = videos.permute(0, 2, 1, 3, 4)
            # ---------------------------------------

            # 2. Prédiction
            outputs = model(videos)
            _, predicted = torch.max(outputs.data, 1)

            # 3. Mise à jour des métriques
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            f1_metric.update(predicted, labels)
            conf_matrix_metric.update(predicted, labels)

            # Petit log pour savoir si ça avance
            if (i+1) % 10 == 0:
                print(f"Step [{i+1}/{len(test_loader)}] processed...")

    # --- RÉSULTATS FINAUX ---
    final_acc = 100 * correct / total
    final_f1 = f1_metric.compute().item()
    final_cm = conf_matrix_metric.compute().cpu().numpy() # On passe en CPU pour le dessin

    print(f'\n=== RÉSULTATS ACTIVITYNET ===')
    print(f'Final Test Accuracy: {final_acc:.2f} %')
    print(f'Final F1-Score: {final_f1:.4f}')

    # --- SAUVEGARDE MATRICE DE CONFUSION ---
    print("Génération de la matrice de confusion (200x200)...")
    
    # On sauvegarde les données brutes au cas où
    np.save("/home/amine_tsp/DL2026/Method_Automatic_pose_recognition/matrice_brute_AN.npy", final_cm)

    # Création de l'image (TRES GRANDE pour 200 classes)
    plt.figure(figsize=(40, 35), facecolor='white') 
    
    # Heatmap sans annotation (trop de chiffres sinon)
    sns.heatmap(final_cm, annot=False, cmap='viridis', fmt='g', cbar=True)
                
    plt.title(f'Matrice de Confusion ActivityNet (Acc: {final_acc:.1f}%)', fontsize=30)
    plt.xlabel('Classes Prédites', fontsize=20)
    plt.ylabel('Classes Réelles', fontsize=20)
    
    output_img = "/home/amine_tsp/DL2026/Method_Automatic_pose_recognition/matrice_confusion_AN_Alex.png"
    plt.savefig(output_img, bbox_inches='tight')
    plt.close()
    
    print(f"Matrice sauvegardée sous : {output_img}")


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
    operation = 1

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

    elif operation == 1: # TEST MODE
        print("Loading ActivityNet Validation Set...")
        
        # On utilise le split 'validation' pour le test
        test_ds = ActivityNetDataset(json_path, video_root, split='validation', clip_len=clip_len, frame_size=(224, 224))
        test_loader = torch.utils.data.DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False, num_workers=4)
        
        # Récupération nombre de classes
        num_classes = len(test_ds.classes)
        print(f"Nombre de classes détectées : {num_classes}")

        # Initialisation du modèle vide (il sera rempli par load_state_dict)
        model = ResNet50LSTM(num_classes=num_classes).to(device)

        # Lancement du test
        test(model=model, SavingName=SavingName, test_loader=test_loader, num_classes=num_classes)

    elif operation == 2: # PLOT MODE
        plott()