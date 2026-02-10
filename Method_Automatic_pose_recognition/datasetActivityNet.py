import os
import json
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class ActivityNetDataset(Dataset):
    def __init__(self, json_path, root_dir, split='training', clip_len=16, frame_size=(224, 224)):
        """
        frame_size=(224, 224) : Standard ImageNet pour une bonne qualité.
        """
        self.root_dir = root_dir
        self.split = split
        self.clip_len = clip_len
        self.frame_size = frame_size
        
        # 1. Charger JSON
        with open(json_path, 'r') as f:
            self.data = json.load(f)['database']
            
        self.samples = []
        self.classes = set()
        
        print(f"Indexation du dataset ({split})...")
        
        # 2. Parcourir le JSON pour trouver les vidéos correspondantes sur le disque
        for video_id, info in self.data.items():
            if info['subset'] == split:
                if len(info['annotations']) > 0:
                    label_str = info['annotations'][0]['label']
                    
                    # Ton script de download a remplacé les espaces par des underscores
                    folder_label = label_str.replace(' ', '_')
                    self.classes.add(label_str)
                    
                    # Chemin supposé: .../raw_clips/training/Archery/v_xyz.mp4
                    # Note: yt-dlp utilise souvent l'ID video comme nom
                    video_filename = f"{video_id}.mp4" 
                    
                    # Chemin complet
                    full_path = os.path.join(self.root_dir, split, folder_label, video_filename)
                    
                    # Vérif mp4, sinon mkv
                    if not os.path.exists(full_path):
                        full_path = os.path.join(self.root_dir, split, folder_label, f"{video_id}.mkv")

                    if os.path.exists(full_path):
                        self.samples.append((full_path, label_str))

        # Trier les classes pour avoir toujours le même ordre
        self.classes = sorted(list(self.classes))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        print(f"Trouvé {len(self.samples)} vidéos valides pour {split}.")

        # 3. Transformations (Normalisation standard ImageNet)
        self.transform = transforms.Compose([
    transforms.Resize(frame_size),
    # Ajouts pour éviter le surapprentissage/stagnation :
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10), 
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    # Fin des ajouts
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label_str = self.samples[idx]
        label_idx = self.class_to_idx[label_str]
        
        # Lecture vidéo avec OpenCV
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            # Vidéo corrompue : renvoyer tenseur noir
            return torch.zeros((self.clip_len, 3, *self.frame_size)), torch.zeros(len(self.classes)), video_path

        # Sampling uniforme des frames
        indices = np.linspace(0, total_frames - 1, self.clip_len).astype(int)
        
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img_tensor = self.transform(img)
                frames.append(img_tensor)
            else:
                # Frame illisible -> noir
                frames.append(torch.zeros(3, *self.frame_size))
                
        cap.release()
        
        # Empiler : [Time, C, H, W]
        video_tensor = torch.stack(frames) 
        
        # Label One-Hot
        target = torch.zeros(len(self.classes))
        target[label_idx] = 1.0
        
        return video_tensor, target, video_path