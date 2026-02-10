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
    def __init__(self, baseDir, transform=None, theType='train', split=1, num_frames=16):
        self.frames_dir = os.path.join(baseDir, 'frames_img_224') 
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
        
        print(f"Dataset prêt : {len(self.data_paths)} vidéos chargées.")

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        # On récupère le chemin du dossier d'images
        video_dir_path = self.data_paths[idx]
        label = self.labels[idx]
        
        #nouveau
        video_name = os.path.basename(video_dir_path)

        # On appelle la fonction qui charge les JPG
        video_tensor = self._load_frames(video_dir_path)
        
        return video_tensor, torch.tensor(label, dtype=torch.long), video_name

    def _load_frames(self, video_dir_path):
        
        # Trouve toutes les images .jpg dans le dossier
        all_frames = sorted(glob.glob(os.path.join(video_dir_path, "*.jpg")))
        total_frames = len(all_frames)
        
        # DIAGNOSTIC 1 : Est-ce que glob trouve les fichiers ?
        #print(f"Dossier : {video_dir_path} | Frames trouvées : {len(all_frames)}")
    
        if len(all_frames) == 0:
            return torch.zeros((3, self.num_frames, 224, 224))
        # Si dossier vide (erreur lors de l'extraction), on renvoie du noir
        if total_frames == 0:
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
                print("exeption")
                frames.append(torch.zeros((3, 224, 224)))
        
        # Padding de sécurité (si on a moins de 16 frames récupérées)
        while len(frames) < self.num_frames:
            frames.append(frames[-1] if len(frames) > 0 else torch.zeros((3, 224, 224)))

        # Empilement : [Temps, Canaux, Hauteur, Largeur]
        video_tensor = torch.stack(frames) 
        
        # DIAGNOSTIC 2 : Quelles sont les valeurs réelles dans le tenseur ?
        #print(f"Valeur Max : {video_tensor.max():.2f}, Valeur Moyenne : {video_tensor.mean():.2f}")
        # Permutation finale pour ResNet3D : [Canaux, Temps, Hauteur, Largeur]
        return video_tensor.permute(1, 0, 2, 3)

    
if __name__ == '__main__':
    
    
    tr = transforms.Compose([
        transforms.Resize((224,224)),
        #transforms.Resize((64,64)),
        transforms.ToTensor()
        ])
    
    # 2. Création du Dataset
    # Assurez-vous que baseDir pointe bien vers le dossier contenant 'videos' et 'ucfTrainTestlist'
    ucf_dataset = UCF101Dataset(baseDir='/home/amine_tsp/DL2026/Datasets/UCF101', transform=tr, theType='train', split=1)

    # 3. Création du DataLoader (C'est lui qui permet de faire 'next(iter(...))')
    # batch_size=1 pour récupérer une seule vidéo à la fois, comme dans votre exemple
    train_loader = torch.utils.data.DataLoader(ucf_dataset, batch_size=1, shuffle=True)

    # 4. Récupération d'un exemple
    # images aura la forme : [Batch, Channels, Frames, Height, Width]
    images, labels, video_names = next(iter(train_loader))



    print("--- Vérification ---")
    print(f"Shape des images : {images.shape}") # Ex: torch.Size([1, 3, 16, 128, 128])
    print(f"Label : {labels}")                  # Ex: tensor([15])
    
    # Pour info : 
    # 1 = Batch size (une vidéo)
    # 3 = Couleurs (RGB)
    # 16 = Nombre de frames (num_frames par défaut)
    # 128, 128 = Taille de l'image

    
    print("Récupération d'une vidéo...")

    # Chemin absolu (si tu veux stocker ailleurs sur le serveur)
    dossier_sortie = "/home/amine_tsp/DL2026/Datasets/UCF101/mes_frames_video"

    # 1. On s'assure que le dossier existe (Python le crée s'il n'existe pas)
    os.makedirs(dossier_sortie, exist_ok=True)
    
    # 1. Récupération des données du batch existant
    video_tensor = images[0]          # Shape : [3, 16, 128, 128]
    label_idx = labels[0].item()      # L'index numérique (ex: 55)
    action_name_estimated = ucf_dataset.idx_to_class[label_idx] # On prend toutes les couleurs (:), la frame 8, toute la hauteur (:), toute la largeur (:)
    
    print("\n" + "="*50)
    print(f" Début de l'extraction des frames pour l'action : {action_name_estimated}")
    print(f" Dossier de destination : {dossier_sortie}")

    # --- 3. Boucle sur les 16 frames ---
    # video_tensor.shape[1] donne le nombre de frames (16)
    nb_frames = video_tensor.shape[1]

    for i in range(nb_frames):
        # Extraction de la frame numéro 'i'
        # [C, Frame_i, H, W] -> [C, H, W]
        frame_tensor = video_tensor[:, i, :, :]
        
        # Conversion [C,H,W] vers [H,W,C] pour l'enregistrement
        display_image = frame_tensor.permute(1, 2, 0).numpy()

        # Construction du nom de fichier unique
        # :02d permet d'écrire 01, 02... 16 (plus facile à trier dans le dossier)
        nom_fichier = f"VERIFICATION_{action_name_estimated}_frame_{i:02d}.png"
        chemin_complet = os.path.join(dossier_sortie, nom_fichier)

        # Sauvegarde
        plt.imsave(chemin_complet, display_image)
        # (Optionnel : petit print pour suivre)
        # print(f"  -> Sauvegardé : {nom_fichier}")

    print("Terminé ! Toutes les images sont générées. Tu peux vérifier qu'elles correspondent bien à la classe donnée.")
    print("="*50 + "\n")

    print(f"Visualisation terminée pour le label : {labels[0].item()}")

    exit(0)