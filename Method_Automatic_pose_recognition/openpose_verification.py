import cv2 as cv
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

# On réutilise les connexions pour dessiner le squelette
BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

def verify_video_keypoints(frames_path, npy_path):
    # 1. Charger les points .npy
    # Shape attendue : (Nb_Frames, 19, 2)
    data = np.load(npy_path, allow_pickle=True)
    
    # 2. Lister les images JPG correspondantes
    images = sorted(glob.glob(os.path.join(frames_path, "*.jpg")))
    
    if len(images) == 0 or len(data) == 0:
        print("Erreur : Images ou données introuvables.")
        return

    print(f"Visualisation de {len(data)} frames...")

    # On va créer un dossier pour enregistrer la vérification
    output_debug = "debug_keypoints"
    os.makedirs(output_debug, exist_ok=True)

    # 3. Boucle sur les frames (on en prend 5 au hasard ou les 5 premières pour vérifier)
    for i in range(min(5, len(data))):
        frame = cv.imread(images[i])
        points = data[i] # Points de la frame i
        
        # Dessiner les points et les lignes
        for pair in POSE_PAIRS:
            partFrom = pair[0]
            partTo = pair[1]
            idFrom = BODY_PARTS[partFrom]
            idTo = BODY_PARTS[partTo]

            pFrom = points[idFrom]
            pTo = points[idTo]

            if pFrom is not None and pTo is not None:
                # Dessiner la ligne du squelette
                cv.line(frame, tuple(pFrom), tuple(pTo), (0, 255, 0), 2)
                # Dessiner les articulations
                cv.circle(frame, tuple(pFrom), 3, (0, 0, 255), cv.FILLED)
                cv.circle(frame, tuple(pTo), 3, (0, 0, 255), cv.FILLED)

        # Sauvegarder l'image de vérification
        cv.imwrite(f"{output_debug}/check_frame_{i:03d}.jpg", frame)
    
    print(f"Vérification terminée. Regarde dans le dossier '{output_debug}' pour voir les images.")

# --- EXÉCUTION ---
if __name__ == '__main__':

    video_folder = "/home/amine_tsp/DL2026/Datasets/UCF101/frames_img_224/PlayingDhol/v_PlayingDhol_g19_c08"
    npy_file = "/home/amine_tsp/DL2026/Datasets/UCF101/openpose_results/PlayingDhol/v_PlayingDhol_g19_c08.npy"
    
    verify_video_keypoints(video_folder, npy_file)