import ssl
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from collections import deque

from models import ResNet50LSTM

# --- 1. CONFIGURATION DU PÉRIPHÉRIQUE (ACCÉLÉRATION MAC/NVIDIA) ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(">>> Succès : Utilisation du GPU Apple (MPS) pour une inférence rapide.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(">>> Succès : Utilisation du GPU NVIDIA (CUDA).")
else:
    device = torch.device("cpu")
    print(">>> Attention : Utilisation du CPU (Cela risque d'être lent).")

ssl._create_default_https_context = ssl._create_unverified_context

# --- 2. DICTIONNAIRE DES CLASSES UCF101 (Index 0 à 100) ---
UCF101_CLASSES = {
    0: 'ApplyEyeMakeup', 1: 'ApplyLipstick', 2: 'Archery', 3: 'BabyCrawling',
    4: 'BalanceBeam', 5: 'BandMarching', 6: 'BaseballPitch', 7: 'Basketball',
    8: 'BasketballDunk', 9: 'BenchPress', 10: 'Biking', 11: 'Billiards',
    12: 'BlowDryHair', 13: 'BlowingCandles', 14: 'BodyWeightSquats', 15: 'Bowling',
    16: 'BoxingPunchingBag', 17: 'BoxingSpeedBag', 18: 'BreastStroke', 19: 'BrushingTeeth',
    20: 'CleanAndJerk', 21: 'CliffDiving', 22: 'CricketBowling', 23: 'CricketShot',
    24: 'CuttingInKitchen', 25: 'Diving', 26: 'Drumming', 27: 'Fencing',
    28: 'FieldHockeyPenalty', 29: 'FloorGymnastics', 30: 'FrisbeeCatch', 31: 'FrontCrawl',
    32: 'GolfSwing', 33: 'Haircut', 34: 'Hammering', 35: 'HammerThrow',
    36: 'HandstandPushups', 37: 'HandstandWalking', 38: 'HeadMassage', 39: 'HighJump',
    40: 'HorseRace', 41: 'HorseRiding', 42: 'HulaHoop', 43: 'IceDancing',
    44: 'JavelinThrow', 45: 'JugglingBalls', 46: 'JumpingJack', 47: 'JumpRope',
    48: 'Kayaking', 49: 'Knitting', 50: 'LongJump', 51: 'Lunges',
    52: 'MilitaryParade', 53: 'Mixing', 54: 'MoppingFloor', 55: 'Nunchucks',
    56: 'ParallelBars', 57: 'PizzaTossing', 58: 'PlayingCello', 59: 'PlayingDaf',
    60: 'PlayingDhol', 61: 'PlayingFlute', 62: 'PlayingGuitar', 63: 'PlayingPiano',
    64: 'PlayingSitar', 65: 'PlayingTabla', 66: 'PlayingViolin', 67: 'PoleVault',
    68: 'PommelHorse', 69: 'PullUps', 70: 'Punch', 71: 'PushUps',
    72: 'Rafting', 73: 'RockClimbingIndoor', 74: 'RopeClimbing', 75: 'Rowing',
    76: 'SalsaSpin', 77: 'ShavingBeard', 78: 'Shotput', 79: 'SkateBoarding',
    80: 'Skiing', 81: 'Skijet', 82: 'SkyDiving', 83: 'SoccerJuggling',
    84: 'SoccerPenalty', 85: 'StillRings', 86: 'SumoWrestling', 87: 'Surfing',
    88: 'Swing', 89: 'TableTennisShot', 90: 'TaiChi', 91: 'TennisSwing',
    92: 'ThrowDiscus', 93: 'TrampolineJumping', 94: 'Typing', 95: 'UnevenBars',
    96: 'VolleyballSpiking', 97: 'WalkingWithDog', 98: 'WallPushups', 99: 'WritingOnBoard',
    100: 'YoYo'
}


def load_action_model(checkpoint_path):
    print(f"Chargement du modèle depuis {checkpoint_path}...")
    # Initialiser le modèle (101 classes pour UCF101)
    model = ResNet50LSTM(num_classes=101)

    # Charger les poids
    # Note : map_location assure que ça charge sur le CPU avant d'envoyer au bon device
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()  # Mode évaluation (fige le Dropout/Batchnorm)
    return model


# Transformation spécifique pour ResNet50 (224x224 + Normalisation ImageNet)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 3. PARAMÈTRES PRINCIPAUX ---
window_size = 16  # Le modèle a besoin de 16 frames
checkpoint_path = "/Users/maurel/PycharmProjects/ArtificialIntelligenceProject/ResNet50LSTM_UCF101_AlexV2.ckpt"
skip_frames = 4  # OPTIMISATION : On ne fait l'IA que toutes les 5 images pour fluidifier

# Charger le modèle
try:
    model = load_action_model(checkpoint_path)
except FileNotFoundError:
    print(f"ERREUR : Le fichier {checkpoint_path} est introuvable.")
    exit()

# Démarrer la webcam
cap = cv2.VideoCapture(0)

# Initialiser le buffer et les variables
frames_buffer = deque(maxlen=window_size)
current_label = "Initialisation..."
frame_count = 0

print("Lancement de la reconnaissance d'action... (Pressez 'q' pour quitter)")

while True:
    # 1. Lire l'image
    ret, frame = cap.read()

    # 2. Vérifier si l'image est valide
    if not ret:
        print("Fin du flux vidéo ou erreur webcam.")
        break

    frame_count += 1

    # --- A. PRÉTRAITEMENT ---
    try:
        # Convertir BGR (OpenCV) -> RGB (PIL)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(frame_rgb)

        # Appliquer les transformations
        image_tensor = transform(image_pil)

        # Ajouter au buffer
        frames_buffer.append(image_tensor)
    except Exception as e:
        print(f"Erreur de prétraitement : {e}")
        continue
    #print(len(frames_buffer))
    # --- B. PRÉDICTION (Optimisée avec skip_frames) ---
    if len(frames_buffer) == window_size and (frame_count % skip_frames == 0):
        # Préparation des données [Batch, Channels, Time, H, W]

        input_sequence = torch.stack(list(frames_buffer))
        #print(input_sequence.shape)
        input_sequence = input_sequence.permute(1, 0, 2, 3)  # [C, T, H, W]
        input_sequence = input_sequence.unsqueeze(0).to(device)  # [1, C, T, H, W]

        with torch.no_grad():
            outputs = model(input_sequence)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

            idx = predicted_idx.item()
            conf = confidence.item()

            if conf > 0.5:
                # Récupération du nom de la classe
                action_name = UCF101_CLASSES[idx] if idx in UCF101_CLASSES else str(idx)
                current_label = f"{action_name} ({conf * 100:.1f}%)"
            else:
                current_label = f"Incertain... ({conf * 100:.1f}%)"

    # --- C. AFFICHAGE ---
    # Rectangle noir pour le fond du texte
    cv2.rectangle(frame, (0, 0), (450, 40), (0, 0, 0), -1)

    # Texte de prédiction
    cv2.putText(frame, current_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)

    # Affichage de la fenêtre
    cv2.imshow('Live Action Recognition', frame)

    # Quitter avec 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Nettoyage
cap.release()
cv2.destroyAllWindows()
