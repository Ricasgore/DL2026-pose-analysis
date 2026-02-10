import torch
from torchvision import transforms
from PIL import Image
from _operator import truediv
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import glob


class UCF101Dataset(torch.utils.data.Dataset):
    def __init__(self, baseDir, transform=None, theType='train', split=1, num_frames=16, returnKeypoint=False):
        self.frames_dir = os.path.join(baseDir, 'frames_img_224') 
        self.returnkeypoint = returnKeypoint
        self.baseDir = baseDir # --- MODIFICATION 1 : On stocke le chemin de base ---
        self.annotation_dir = os.path.join(baseDir, 'ucfTrainTestlist')
    
        self.transform = transform
        self.num_frames = num_frames
        self.data_paths = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {} 

        # 1. Chargement des noms de classes
        class_ind_path = os.path.join(self.annotation_dir, 'classInd.txt')
        with open(class_ind_path, 'r') as f:
            for line in f:
                parts = line.strip().split() # Utilisation de .split() robuste
                if len(parts) >= 2:
                    idx = int(parts[0]) - 1
                    name = parts[1]
                    self.class_to_idx[name] = idx
                    self.idx_to_class[idx] = name

        # 2. Sélection du fichier split
        fname = f'trainlist{split:02d}.txt' if theType == 'train' else f'testlist{split:02d}.txt'
        list_file = os.path.join(self.annotation_dir, fname)

        # 3. Récupération des chemins
        with open(list_file, 'r') as f:
            for line in f:
                parts = line.strip().split() # Utilisation de .split() robuste
                if not parts: continue
            
                video_avi_path = parts[0] # ex: Archery/v_Archery_g01_c01.avi
                video_folder_name = video_avi_path.replace('.avi', '')
                full_path = os.path.join(self.frames_dir, video_folder_name)
            
                # On n'ajoute la vidéo que si le dossier existe vraiment
                if os.path.exists(full_path):
                    self.data_paths.append(full_path)
                
                    if theType == 'train' and len(parts) > 1:
                        # On prend le label écrit dans le fichier
                        self.labels.append(int(parts[1]) - 1)
                    else:
                        # Mode test : on extrait le nom de la classe depuis le chemin
                        # "Archery/v_Archery..." -> "Archery"
                        class_name = video_avi_path.split('/')[0]
                        self.labels.append(self.class_to_idx.get(class_name, 0))

        print(f"Dataset prêt : {len(self.data_paths)} vidéos chargées. (Keypoints={self.returnkeypoint})")
 
    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        # On récupère le chemin du dossier d'images
        video_dir_path = self.data_paths[idx]
        label = self.labels[idx]
        
        # On appelle la fonction qui charge les JPG
        video_tensor = self._load_frames(video_dir_path)

        # --- MODIFICATION 2 : Chargement des Keypoints ---
        if self.returnkeypoint:
            keypoints = self._load_keypoints(video_dir_path)
            # Retourne : (Vidéo, Label, Points)
            return video_tensor, torch.tensor(label, dtype=torch.long), keypoints
        else:
            # Retourne : (Vidéo, Label)
            return video_tensor, torch.tensor(label, dtype=torch.long)

    def _load_keypoints(self, video_dir_path):
        """
        Charge le fichier .npy associé à la vidéo, nettoie les données,
        sélectionne 16 frames et normalise les coordonnées.
        """
        # 1. Reconstruire le chemin vers le .npy
        # video_dir_path = .../frames_img_224/Archery/v_Archery_g01_c01
        video_name = os.path.basename(video_dir_path)
        class_name = os.path.basename(os.path.dirname(video_dir_path))
        
        # Chemin supposé : .../Datasets/UCF101/openpose_results/Archery/v_Archery_g01_c01.npy
        npy_path = os.path.join(self.baseDir, 'openpose_results', class_name, f"{video_name}.npy")

        # Fallback : Si le fichier n'existe pas, on renvoie des zéros
        # Shape finale attendue : [16 frames, 38 coordonnées (19x2)]
        if not os.path.exists(npy_path):
            return torch.zeros((self.num_frames, 38), dtype=torch.float32)

        try:
            data = np.load(npy_path, allow_pickle=True)
        except:
            return torch.zeros((self.num_frames, 38), dtype=torch.float32)

        total_frames = len(data)
        if total_frames == 0:
            return torch.zeros((self.num_frames, 38), dtype=torch.float32)

        # 2. Sélectionner les mêmes indices temporels que pour les images
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

        sampled_keypoints = []
        for i in indices:
            frame_points = data[i] # Liste de 19 points (ou None) pour cette frame
            flat_points = []
            
            # OpenPose renvoie 19 points. Chaque point est (x, y) ou None.
            for point in frame_points:
                if point is not None:
                    # Normalisation : on divise par 224 (taille de l'image) pour avoir des valeurs entre 0 et 1
                    norm_x = float(point[0]) / 224.0
                    norm_y = float(point[1]) / 224.0
                    flat_points.extend([norm_x, norm_y])
                else:
                    # Si le point n'est pas détecté -> 0.0
                    flat_points.extend([0.0, 0.0])
            
            sampled_keypoints.append(flat_points)

        # Conversion en Tenseur PyTorch [16, 38]
        return torch.tensor(sampled_keypoints, dtype=torch.float32)

    def _load_frames(self, video_dir_path):
        """
        Fonction légère : Liste les JPG et les empile.
        """
        # Trouve toutes les images .jpg dans le dossier
        all_frames = sorted(glob.glob(os.path.join(video_dir_path, "*.jpg")))
        total_frames = len(all_frames)
    
        if len(all_frames) == 0:
            return torch.zeros((3, self.num_frames, 224, 224))
        
        # On choisit 16 indices équitablement répartis
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        frames = []
        for i in indices:
            img_path = all_frames[i]
            try:
                # Ouverture rapide avec PIL
                with Image.open(img_path) as img:
                    img = img.convert('RGB')
                    if self.transform:
                        img = self.transform(img)
                    frames.append(img)
            except Exception:
                # Si image corrompue, on met une frame noire
                frames.append(torch.zeros((3, 224, 224)))
        
        # Padding de sécurité
        while len(frames) < self.num_frames:
            frames.append(frames[-1] if len(frames) > 0 else torch.zeros((3, 224, 224)))

        # Empilement : [Temps, Canaux, Hauteur, Largeur]
        video_tensor = torch.stack(frames) 
        
        # Permutation finale pour ResNet3D : [Canaux, Temps, Hauteur, Largeur]
        return video_tensor.permute(1, 0, 2, 3)

    
if __name__ == '__main__':
    
    tr = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
        ])
    
    # 2. Création du Dataset AVEC Keypoints
    print("Test du Dataset avec Keypoints...")
    ucf_dataset = UCF101Dataset(
        baseDir='/home/amine_tsp/DL2026/Datasets/UCF101', 
        transform=tr, 
        theType='train', 
        split=1, 
        returnKeypoint=True  # <--- ON ACTIVE ICI
    )

    # 3. DataLoader
    train_loader = torch.utils.data.DataLoader(ucf_dataset, batch_size=1, shuffle=True)

    # 4. Récupération d'un exemple
    try:
        # Note : data contient maintenant 3 éléments
        data = next(iter(train_loader))
        images = data[0]
        labels = data[1]
        keypoints = data[2]

        print("\n--- Vérification ---")
        print(f"Shape des images   : {images.shape}")   # Ex: [1, 3, 16, 224, 224]
        print(f"Shape des Keypoints: {keypoints.shape}") # Ex: [1, 16, 38] (16 frames, 19*2 coords)
        print(f"Label              : {labels}")

        # Vérifions si les keypoints ne sont pas tous à zéro
        if keypoints.sum() == 0:
            print("⚠️ Attention : Les keypoints sont tous à 0. Vérifiez que les fichiers .npy existent bien dans openpose_results.")
        else:
            print(f"Exemple de keypoints (Frame 0) : {keypoints[0, 0, :4]} ...")
            print("✅ Chargement réussi !")

    except Exception as e:
        print(f"Erreur lors du chargement : {e}")

    exit(0)