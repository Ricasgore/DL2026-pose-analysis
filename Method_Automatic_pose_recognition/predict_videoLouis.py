'''import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.video import r3d_18, R3D_18_Weights
import cv2
import numpy as np
from PIL import Image

# 1. CONFIGURATION
MODEL_PATH = './checkpoints/R3D18_UCF101.ckpt' # Votre fichier sauvegardé
VIDEO_PATH = '/home/amine_tsp/DL2026/Datasets/UCF101/videos/Archery/v_Archery_g01_c01.avi' # Mettez le chemin d'une vidéo à tester ici
CLASS_IND_PATH = '/home/amine_tsp/DL2026/Datasets/UCF101/ucfTrainTestlist/classInd.txt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. CHARGEMENT DES CLASSES
idx_to_class = {}
with open(CLASS_IND_PATH, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            idx_to_class[int(parts[0]) - 1] = parts[1]

# 3. CHARGEMENT DU MODÈLE
print("Chargement du modèle...")
model = r3d_18(weights=None) # Pas besoin des poids ImageNet, on charge les vôtres
model.fc = nn.Linear(model.fc.in_features, 101) # 101 classes
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# 4. FONCTION DE PRÉPARATION VIDÉO
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        print("Erreur lecture vidéo")
        return None

    # On prend 16 frames réparties équitablement
    indices = np.linspace(0, total_frames-1, 16, dtype=int)
    frames = []
    
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor()
    ])

    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img = transform(img)
            frames.append(img)
        else:
            # Padding si erreur
            frames.append(torch.zeros(3, 112, 112))
            
    cap.release()
    
    # Stack et format [Batch, C, T, H, W]
    video_tensor = torch.stack(frames).permute(1, 0, 2, 3)
    return video_tensor.unsqueeze(0) # Ajoute la dimension Batch

# 5. PRÉDICTION
print(f"Analyse de {VIDEO_PATH}...")
input_tensor = process_video(VIDEO_PATH)

if input_tensor is not None:
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        # Softmax pour avoir des pourcentages
        probs = torch.nn.functional.softmax(outputs, dim=1)
        
        # Récupérer le Top 3
        top3_prob, top3_idx = torch.topk(probs, 3)
        
    print("\n--- RÉSULTATS ---")
    for i in range(3):
        idx = top3_idx[0][i].item()
        score = top3_prob[0][i].item() * 100
        name = idx_to_class.get(idx, "Inconnu")
        print(f"{i+1}. {name}: {score:.2f}%")'''