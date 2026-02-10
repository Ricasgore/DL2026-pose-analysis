import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassF1Score

# Imports locaux
from models import TwoStreamFusion, ResNet50LSTM
from config import device
from datasetUCFAlex import UCF101Dataset 

# --- 1. FONCTIONS DE LOGS ---

def plott():
    path = '/home/amine_tsp/DL2026/Method_Automatic_pose_recognition/loss_save_during_training/loss_UCF101_Alex.npy'
    if os.path.exists(path):
        a = np.load(path, allow_pickle=True)
        plt.plot(a)
        plt.title("Loss Training - Two-Stream Fusion")
        plt.savefig('/home/amine_tsp/DL2026/Method_Automatic_pose_recognition/loss_graph_plot/loss_UCF101_Alex.png')
        plt.close()
        print('finish')

# --- 2. FONCTIONS DE TRAINING / VAL / TEST ---

def train(model=None, SavingName=None, train_loader=None, val_loader=None, optimizer=None, num_epochs=50):
    print('Début de l\'entraînement du modèle FUSION (Vidéo + Pose)...')
    criterion = nn.CrossEntropyLoss() 
    losses = []
    total_step = len(train_loader)

    for epoch in range(num_epochs):
        model.train()
        for i, (data) in enumerate(train_loader):
            videos = data[0].to(device)
            labels = data[1].to(device)
            poses  = data[2].to(device) # Keypoints [Batch, 16, 38]

            # Forward pass fusionné
            outputs = model(videos, poses) 
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')
                losses.append(loss.item())
                
                save_dir = '/home/amine_tsp/DL2026/Method_Automatic_pose_recognition/loss_save_during_training'
                os.makedirs(save_dir, exist_ok=True)
                np.save(os.path.join(save_dir, 'loss_UCF101_Alex.npy'), losses)

        # Validation et sauvegarde toutes les 5 époques
        if (epoch + 1) % 5 == 0:
            validate(model, val_loader)
            os.makedirs(os.path.dirname(SavingName), exist_ok=True)
            torch.save(model.state_dict(), SavingName)
            print(f"Checkpoint Fusion sauvegardé : {SavingName}")

def validate(model, val_loader):
    print("Validating Fusion Model...")
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data in val_loader:
            # CORRECTION : On déballe les 3 éléments
            videos, labels, poses = data[0].to(device), data[1].to(device), data[2].to(device)
            # CORRECTION : On donne les 2 entrées au modèle
            outputs = model(videos, poses)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    print(f'Validation Accuracy: {100 * correct / total:.2f} %')
    model.train()

def test(model=None, SavingName=None, test_loader=None):
    print(f"Loading Fusion model from {SavingName}")
    model.load_state_dict(torch.load(SavingName))
    model.eval()

    f1_metric = MulticlassF1Score(num_classes=101, average='macro').to(device)
    conf_matrix_metric = MulticlassConfusionMatrix(num_classes=101).to(device)
    correct, total = 0, 0
    
    with torch.no_grad():
        for i, (data) in enumerate(test_loader):
            # Déballage des keypoints obligatoire pour le test fusion
            videos, labels, poses = data[0].to(device), data[1].to(device), data[2].to(device)
            outputs = model(videos, poses)
            _, predicted = torch.max(outputs.data, 1)

            f1_metric.update(predicted, labels)
            conf_matrix_metric.update(predicted, labels)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Final Fusion Accuracy: {100 * correct / total:.2f} %')
    final_cm = conf_matrix_metric.compute().cpu().numpy()
    
    plt.figure(figsize=(25, 20))
    sns.heatmap(final_cm, annot=False, cmap='viridis')
    plt.savefig("/home/amine_tsp/DL2026/Method_Automatic_pose_recognition/matrice_confusion_Fusion_Alex.png")
    plt.close()
    print("Matrice de confusion sauvegardée.")

def predict_top5_video(model, video_path, dataset_obj, device):
    model.eval()
    with torch.no_grad():
        # 1. Chargement Vidéo + Pose
        video_tensor = dataset_obj._load_frames(video_path).unsqueeze(0).to(device)
        pose_tensor = dataset_obj._load_keypoints(video_path).unsqueeze(0).to(device)
        
        # 2. Forward pass Fusion
        outputs = model(video_tensor, pose_tensor)
        probs = F.softmax(outputs, dim=1)
        top5_probs, top5_indices = torch.topk(probs, k=5)
        
        results = []
        for i in range(5):
            idx = top5_indices[0][i].item()
            score = top5_probs[0][i].item()
            class_name = dataset_obj.idx_to_class[idx]
            results.append((class_name, score))
        return results

# --- 3. MAIN ---

if __name__ == '__main__':
    batch_size = 16
    learning_rate = 1e-4
    num_epochs = 50

    # Initialisation du modèle Fusion
    model = TwoStreamFusion(num_classes=101).to(device)
    
    # Nouveau nom de sauvegarde pour la Fusion
    SavingName = '/home/amine_tsp/DL2026/Method_Automatic_pose_recognition/checkpoints/TwoStreamFusion_UCF101_Alex.ckpt'

    # --- INJECTION OPTIONNELLE : Charger ta Baseline 72% ---
    baseline_path = '/home/amine_tsp/DL2026/Method_Automatic_pose_recognition/checkpoints/ResNet50LSTM_UCF101_Alex.ckpt'
    if os.path.exists(baseline_path):
        print("Injection des poids de la Baseline (72%) dans la branche vidéo...")
        baseline_state = torch.load(baseline_path, map_location=device)
        filtered_state = {k: v for k, v in baseline_state.items() if not k.startswith('fc')}
        model.video_stream.load_state_dict(filtered_state, strict=False)

    tr = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    path = '/home/amine_tsp/DL2026/Datasets/UCF101'

    # 0=Train, 1=Test, 3=Predict

    operation = 1

    if operation == 0:
        PoseTrain = UCF101Dataset(path, tr, 'train', returnKeypoint=True)
        train_loader = torch.utils.data.DataLoader(PoseTrain, batch_size, shuffle=True, num_workers=4)
        
        PoseVal = UCF101Dataset(path, tr, 'test', returnKeypoint=True)
        val_loader = torch.utils.data.DataLoader(PoseVal, batch_size, shuffle=False)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        train(model, SavingName, train_loader, val_loader, optimizer, num_epochs)

    elif operation == 1:
        PoseTest = UCF101Dataset(path, tr, 'test', returnKeypoint=True)
        test_loader = torch.utils.data.DataLoader(PoseTest, batch_size, shuffle=False)
        test(model, SavingName, test_loader)

    elif operation == 3:
        # Prediction Top 5 avec Fusion
        if os.path.exists(SavingName):
            model.load_state_dict(torch.load(SavingName))
            ds = UCF101Dataset(path, tr, 'test', returnKeypoint=True)
            v_path = "/home/amine_tsp/DL2026/Datasets/UCF101/frames_img_224/Basketball/v_Basketball_g01_c01"
            top5 = predict_top5_video(model, v_path, ds, device)
            
            print("\nRésultats Fusion Top 5 :")
            for r, (name, score) in enumerate(top5, 1):
                print(f"{r}. {name} ({score*100:.2f}%)")