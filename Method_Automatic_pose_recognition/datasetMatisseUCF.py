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


class LSPPE(torch.utils.data.Dataset):
    '''def __init__(self, dataDir='./lsp_dataset', transform=None, crossNum=None,crossIDs=None, theType='train'):

        self.labels = []
        self.data = []

        # First load all images data

        listImage = os.listdir(dataDir + '/images')
        listImage = sorted(listImage)

        #print(len(listImage))

        data = loadmat(dataDir + '/joints.mat')
        joints = data['joints']  

        if theType == 'train':
            listImage_len = range(int(len(listImage) * 0.8))
        elif theType == 'val':
            listImage_len = range(int(len(listImage) * 0.6), int(len(listImage) * 0.8))
        else:
            listImage_len = range(int(len(listImage) * 0.8), len(listImage))

        #print(joints.shape) #Print the shape of joints

        for i in listImage_len:
            img = listImage[i]
            joint = joints[:,:,i]
            
            self.transform = transform
                
            data = dataDir+'/images/'+img
            lbl = joint

            data = Image.open(data).convert('RGB')

            x_joint = lbl[:, 0]
            y_joint = lbl[:, 1]

            """
            plt.imshow(data)
            plt.scatter(x_joint,y_joint)
            plt.legend()
            plt.title("First image")
            plt.show()
            """

            #crop the image using min_x, min_y and max_y, max_x
            min_x = max(0,np.min(x_joint) - 15)
            min_y = max(0,np.min(y_joint) - 15)
            max_x = max(0,np.max(x_joint) + 15)
            max_y = max(0,np.max(y_joint) + 15)

            #print((min_x,min_y,max_x,max_y))

            data_crop = data.crop((min_x,min_y,max_x,max_y))

            #adjust the ground truth using min_x and min_y
            lbl_crop = np.copy(lbl)

            lbl_crop[:, 0] = lbl[:, 0] - min_x
            lbl_crop[:, 1] = lbl[:, 1] - min_y

            """
            plt.imshow(data_crop)
            plt.scatter(lbl_crop[:,0], lbl_crop[:,1])
            plt.legend()
            plt.title("crop")
            plt.show()
            """

            #resize the image using conventional size of 128 and 128
            SIZE_X = 128
            SIZE_Y = 128

            data_resize = data_crop.resize((SIZE_X,SIZE_Y))

            lbl_resize = np.copy(lbl_crop)
            lbl_resize[:,0] = (lbl_crop[:,0]/data_crop.width) * SIZE_X
            lbl_resize[:,1] = (lbl_crop[:,1]/data_crop.height) * SIZE_Y
                
            """
            plt.imshow(data_resize)
            plt.scatter(lbl_resize[:,0], lbl_resize[:,1])
            plt.legend()
            plt.title("resize")
            plt.show()
            """

            if self.transform is not None:
                data_resize = self.transform(data_resize)
                
            self.data.append(data_resize)
            self.labels.append(lbl_resize)

            #print(len(self.data))

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    
    def __len__(self):
        #print(len(self.labels))
        return len(self.data)

class MPII(torch.utils.data.Dataset):
    def __init__(self, dataDir='./mpii_dataset', transform=None, crossNum=None, crossIDs=None, theType='test'):

        print("start init")
        self.labels = []
        self.data = []
        self.max_joints = 16
        self.transform = transform

        listImage = os.listdir(dataDir + '/images')
        listImage = sorted(listImage)

        data = loadmat(dataDir+'/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat')
        annotations = data['RELEASE']

        annolist = annotations['annolist'][0, 0]
        num_images = annolist.shape[1]

        if theType == 'train':
            listImage = range(int(num_images * 0.6))
            #print(listImage)
        elif theType == 'val':
            listImage = range(int(num_images * 0.6), int(num_images * 0.8))
            #print(listImage)
        else:
            listImage = range(int(num_images * 0.8), num_images)
            #print(listImage)

        for i in listImage:
            img_struct = annolist['image'][0, i][0]
            img_name = img_struct['name'][0][0]
            if annolist['annorect'][0, i].dtype.names is None:
                continue
            elif 'annopoints' not in annolist['annorect'][0, i].dtype.names:
                continue
            else:
                annorect = annolist['annorect'][0, i]['annopoints']
                for person in annorect[0]:
                    if person.size == 0:
                        continue
                    else:
                        if person['point'][0, 0][0].shape == (16,):
                            image_path = os.path.join(dataDir, 'images', img_name)
                            image = Image.open(image_path).convert('RGB')

                            x_joint = np.array([point['x'][0, 0] for point in person['point'][0, 0][0]])
                            y_joint = np.array([point['y'][0, 0] for point in person['point'][0, 0][0]])
                            joint_id = np.array([point['id'][0, 0] for point in person['point'][0, 0][0]])

                            min_x = max(0, np.min(x_joint) - 15)
                            min_y = max(0, np.min(y_joint) - 15)
                            max_x = min(image.width, np.max(x_joint) + 15)
                            max_y = min(image.height, np.max(y_joint) + 15)

                            data_crop = image.crop((min_x, min_y, max_x, max_y))

                            x_joint -= min_x
                            y_joint -= min_y

                            SIZE_X, SIZE_Y = 128, 128
                            data_resize = data_crop.resize((SIZE_X, SIZE_Y))

                            x_joint = (x_joint / data_crop.width) * SIZE_X
                            y_joint = (y_joint / data_crop.height) * SIZE_Y

                            lbl_resize = np.array([x_joint, y_joint]).T
                            # Sort by Joint ID
                            lbl_resize = np.array(sorted(zip(x_joint, y_joint, joint_id), key=lambda x: x[2]))[:, :2]

                            if self.transform is not None:
                                data_resize = self.transform(data_resize)
                            
                            self.data.append(data_resize)
                            self.labels.append(torch.tensor(lbl_resize))
                                   
            if i % 100 == 0:
                print('init = '+str((i/len(listImage)*100))+'%')  

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)
'''


class UCF101Dataset(torch.utils.data.Dataset):
    def __init__(self, baseDir, transform=None, theType='train', split=1, num_frames=16):
        """
        Version optimisée pour lire les FRAMES (JPG) et non les VIDÉOS (AVI).
        """
        print(f"Initialisation UCF101 (Lecture JPG) - Split: {split} - Type: {theType}")
        
        # Cible le dossier frames_img (généré par votre script de preprocessing)
        self.frames_dir = os.path.join(baseDir, 'frames_img_224') 
        self.annotation_dir = os.path.join(baseDir, 'ucfTrainTestlist')
        
        self.transform = transform
        self.num_frames = num_frames
        self.data_paths = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {} 

        # 1. Chargement des noms de classes (Archery = 0, etc.)
        class_ind_path = os.path.join(self.annotation_dir, 'classInd.txt')
        if not os.path.exists(class_ind_path):
            raise FileNotFoundError(f"Fichier introuvable : {class_ind_path}")

        with open(class_ind_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    idx = int(parts[0]) - 1
                    name = parts[1]
                    self.class_to_idx[name] = idx
                    self.idx_to_class[idx] = name

        # 2. Sélection du fichier split (trainlist01.txt ou testlist01.txt)
        fname = f'trainlist{split:02d}.txt' if theType == 'train' else f'testlist{split:02d}.txt'
        list_file = os.path.join(self.annotation_dir, fname)

        # 3. Récupération des chemins
        with open(list_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                
                parts = line.split(' ')
                video_avi_name = parts[0] # ex: Archery/v_Archery_g01_c01.avi
                
                # IMPORTANT : On enlève '.avi' pour trouver le dossier d'images correspondant
                video_folder_name = video_avi_name.split('.')[0]
                full_path = os.path.join(self.frames_dir, video_folder_name)
                
                self.data_paths.append(full_path)
                
                # Gestion du label
                if theType == 'train' and len(parts) > 1:
                    self.labels.append(int(parts[1]) - 1)
                else:
                    # Pour le test, on devine le label via le nom du dossier
                    class_name = video_folder_name.split('/')[0]
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
        
        return video_tensor, torch.tensor(label, dtype=torch.long), video_name #nouveau

    def _load_frames(self, video_dir_path):
        """
        Fonction légère : Liste les JPG et les empile.
        """
        # Trouve toutes les images .jpg dans le dossier
        all_frames = sorted(glob.glob(os.path.join(video_dir_path, "*.jpg")))
        total_frames = len(all_frames)
        
        # Si dossier vide (erreur lors de l'extraction), on renvoie du noir
        if total_frames == 0:
            return torch.zeros((3, self.num_frames, 112, 112))

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
                frames.append(torch.zeros((3, 112, 112)))
        
        # Padding de sécurité (si on a moins de 16 frames récupérées)
        while len(frames) < self.num_frames:
            frames.append(frames[-1] if len(frames) > 0 else torch.zeros((3, 112, 112)))

        # Empilement : [Temps, Canaux, Hauteur, Largeur]
        video_tensor = torch.stack(frames) 
        
        # Permutation finale pour ResNet3D : [Canaux, Temps, Hauteur, Largeur]
        return video_tensor.permute(1, 0, 2, 3)
    
    
if __name__ == '__main__':
    
    
    tr = transforms.Compose([
        #transforms.Resize((128,128)),
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
    images, labels = next(iter(train_loader))



    print("--- Vérification ---")
    print(f"Shape des images : {images.shape}") # Ex: torch.Size([1, 3, 16, 128, 128])
    print(f"Label : {labels}")                  # Ex: tensor([15])
    
    # Pour info : 
    # 1 = Batch size (une vidéo)
    # 3 = Couleurs (RGB)
    # 16 = Nombre de frames (num_frames par défaut)
    # 128, 128 = Taille de l'image

    '''
    lsppe = LSPPE(transform=tr,crossNum=5, crossIDs=[5])
    #print(len(lsppe))
    
    images,labels = next(iter(lsppe))'''
    #print(images,labels)
    #print(images.shape)
    #print(labels.shape)

    '''for i, (imaages,labels) in enumerate(train_loader):
        print(images.shape)
        print(labels.shape)'''
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

    #   affiche les 16 prmeiers frame 
    # verifier l'histoire des 16 frames 