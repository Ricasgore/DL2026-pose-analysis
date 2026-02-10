import os
import cv2
import glob
from tqdm import tqdm

# CONFIGURATION
SOURCE_VIDEO_DIR = '/home/amine_tsp/DL2026/Datasets/UCF101/videos'
TARGET_FRAME_DIR = '/home/amine_tsp/DL2026/Datasets/UCF101/frames_img_224'

# TAILLE CIBLE (Plus petit = Moins de place)
RESIZE_DIM = (224, 224)

def extract_frames_from_video(video_path, output_folder):
    if os.path.exists(output_folder) and len(os.listdir(output_folder)) > 0:
        return

    os.makedirs(output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 1. REDIMENSIONNEMENT (Gain de place énorme)
        frame = cv2.resize(frame, RESIZE_DIM)
        
        # 2. SAUVEGARDE COMPRESSÉE (Qualité 60 au lieu de 95 par défaut)
        filename = os.path.join(output_folder, f"frame_{count:05d}.jpg")
        cv2.imwrite(filename, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        
        count += 1
    
    cap.release()

def main():
    class_folders = sorted(os.listdir(SOURCE_VIDEO_DIR))
    print(f"Extraction LÉGÈRE vers {TARGET_FRAME_DIR}...")

    for class_name in tqdm(class_folders):
        class_path_src = os.path.join(SOURCE_VIDEO_DIR, class_name)
        if not os.path.isdir(class_path_src): continue
            
        class_path_dst = os.path.join(TARGET_FRAME_DIR, class_name)
        os.makedirs(class_path_dst, exist_ok=True)
        
        video_files = glob.glob(os.path.join(class_path_src, "*.avi"))
        
        for video_path in video_files:
            video_name = os.path.basename(video_path).split('.')[0]
            output_folder = os.path.join(class_path_dst, video_name)
            extract_frames_from_video(video_path, output_folder)

if __name__ == '__main__':
    main()






