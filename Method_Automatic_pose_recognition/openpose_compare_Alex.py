################################################################################################################################
# We reused and drew inspiration from the work of misbah4064 : "https://github.com/misbah4064/human-pose-estimation-opencv"    #
################################################################################################################################

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from metrics import *
import os
import glob


BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]



# 2. FONCTION DE PRÉDICTION INDIVIDUELLE (Optimisée pour les fichiers)
def predict_openpose_frame(frame_path, net):
    inWidth, inHeight = 368, 368
    threshold = 0
    
    frame = cv.imread(frame_path)
    if frame is None: return None

    frameWidth, frameHeight = frame.shape[1], frame.shape[0]
    
    # Transformation de l'image pour le réseau
    blob = cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False)
    net.setInput(blob)
    out = net.forward()
    out = out[:, :19, :, :]

    points = []
    for i in range(19):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > threshold else None)
    
    return points

# 3. FONCTION PRINCIPALE : TRAITEMENT AUTOMATIQUE DE TOUT LE DATASET
def process_all_ucf101_openpose(base_dir):
    """
    Parcourt récursivement TOUS les sous-dossiers de UCF101.
    """
    print("Chargement du modèle OpenPose...")
    net = cv.dnn.readNetFromTensorflow("/home/amine_tsp/DL2026/Method_Automatic_pose_recognition/openpose/graph_opt.pb")
    
    # Dossier de sortie pour les fichiers .npy
    output_base = "/home/amine_tsp/DL2026/Datasets/UCF101/openpose_results"
    os.makedirs(output_base, exist_ok=True)

    # Parcours de l'arborescence (ApplyEyeMakeup, BalanceBeam, etc.)
    for root, dirs, files in os.walk(base_dir):
        # On cherche les dossiers qui contiennent des frames .jpg
        jpg_files = sorted([f for f in files if f.endswith(".jpg")])
        
        if len(jpg_files) > 0:
            video_name = os.path.basename(root)
            class_name = os.path.basename(os.path.dirname(root))
            
            print(f"\nTraitement : {class_name} -> {video_name} ({len(jpg_files)} frames)")
            
            video_points = []
            for f in jpg_files:
                img_path = os.path.join(root, f)
                points = predict_openpose_frame(img_path, net)
                video_points.append(points)
            
            # Sauvegarde organisée par classe
            save_dir = os.path.join(output_base, class_name)
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, f"{video_name}.npy"), video_points)
            print(f"Sauvegardé : {video_name}.npy")

# --- EXÉCUTION ---
if __name__ == '__main__':
    # Le chemin vers la racine de tes images JPG
    UCF_ROOT_IMAGES = "/home/amine_tsp/DL2026/Datasets/UCF101/frames_img_224"
    
    process_all_ucf101_openpose(UCF_ROOT_IMAGES)







def verify(base_dir):
    """
    Parcourt récursivement TOUS les sous-dossiers de UCF101.
    """
    print("Chargement du modèle OpenPose...")
    net = cv.dnn.readNetFromTensorflow("/home/amine_tsp/DL2026/Method_Automatic_pose_recognition/openpose/graph_opt.pb")
    
    # Dossier de sortie pour les fichiers .npy
    output_base = "/home/amine_tsp/DL2026/Datasets/UCF101/openpose_results"
    os.makedirs(output_base, exist_ok=True)

    # Parcours de l'arborescence (ApplyEyeMakeup, BalanceBeam, etc.)
    for root, dirs, files in os.walk(base_dir):
        # On cherche les dossiers qui contiennent des frames .jpg
        jpg_files = sorted([f for f in files if f.endswith(".jpg")])
        
        if len(jpg_files) > 0:
            video_name = os.path.basename(root)
            class_name = os.path.basename(os.path.dirname(root))
            
            print(f"\nTraitement : {class_name} -> {video_name} ({len(jpg_files)} frames)")
            
            video_points = []
            for f in jpg_files:
                img_path = os.path.join(root, f)
                points = predict_openpose_frame(img_path, net)
                video_points.append(points)
            








'''
############################################
#  If input = 0 => open comera and predict #
#  or put as input an image                #
############################################
def predict_openpose(input=0):

    inWidth = 368   #default size
    inHeight = 368  #default size

    threshold = 0  #default threshold=0.2 we choose 0 in order to have all the points

    net = cv.dnn.readNetFromTensorflow(".openpose/graph_opt.pb")

    cap = cv.VideoCapture(input)

    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            #print('fin')
            break

        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        
        net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
        out = net.forward()
        out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

        assert(len(BODY_PARTS) == out.shape[1])

        points = []
        for i in range(len(BODY_PARTS)):
            # Slice heatmap of corresponging body's part.
            heatMap = out[0, i, :, :]

            # Originally, we try to find all the local maximums. To simplify a sample
            # we just find a global one. However only a single pose at the same time
            # could be detected this way.
            _, conf, _, point = cv.minMaxLoc(heatMap)
            x = (frameWidth * point[0]) / out.shape[3]
            y = (frameHeight * point[1]) / out.shape[2]
            # Add a point if it's confidence is higher than threshold.
            points.append((int(x), int(y)) if conf > threshold else None)

        for pair in POSE_PAIRS:
            partFrom = pair[0]
            partTo = pair[1]
            assert(partFrom in BODY_PARTS)
            assert(partTo in BODY_PARTS)

            idFrom = BODY_PARTS[partFrom]
            idTo = BODY_PARTS[partTo]

            if points[idFrom] and points[idTo]:
                cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
                cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
                cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

        t, _ = net.getPerfProfile()
        freq = cv.getTickFrequency() / 1000
        cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        #print in a new window
        #cv.imshow('OpenPose using OpenCV', frame)

    return points

#test
#filepath = 'image.jpg'
#predict_openpose(filepath)

# for each folder, get the name of the folder 
    #call test on dataset given directory name as input and directory+npy name as output
#listFile = ['v_ApplyEyeMakeup_g01_c01.avi', 'v_ApplyEyeMakeup_g01_c02.avi']  #example for UCF101 dataset
######################################
#test openpose on the UCF images #
######################################
def test_on_dataset_UCF(folderName):
    #datatest = np.load('../mpii_test_results_openpose/datatest.npy', allow_pickle=True) #we saved the images in a datatest.npy file but we can also use directly the image folder by modifying this line
    train_results_points = []

    # get the list of the files i the folder and put them as a variable listFile

    for k in range(len(listFile)):
        test = datatest[k].permute(1, 2, 0).numpy()  # now : (128, 128, 3) and convert to numpy
        test = (test * 255).astype(np.uint8) #Converts values to uint8 (0-255)

        # Convert RGB en BGR for OpenCV
        test_bgr = cv.cvtColor(test, cv.COLOR_RGB2BGR)

        output_path = "temporary.jpg"
        cv.imwrite(output_path, test_bgr) #save the file temporary
        points = predict_openpose("temporary.jpg")
        train_results_points.append(points)
        print(k/len(datatest)*100, '%')

    np.save('folderName'+'.npy', train_results_points)

#############################################
# Compare the predictions with their labels #
#############################################
def compare_labels_dataset1():
    pred = np.load('../lsp_test_results_openpose/train_results_points.npy', allow_pickle=True)
    labelstest = np.load('../lsp_test_results_openpose/labelstest.npy', allow_pickle=True)

    preds_test_rearranged = []

    # Ensure labelstest elements are converted to NumPy and reshape properly
    labelstest_modfied = np.zeros((len(labelstest), 13, 2))
    for k in range(len(labelstest)):
        if isinstance(labelstest[k], torch.Tensor):
            labelstest[k] = labelstest[k].numpy()  # Convert from tensor to NumPy
        
        labelstest_modfied[k] = labelstest[k][:-1, :2]  # Remove last row and keep only the first two columns (x, y)


    # Ensure labelstest is a proper 3D NumPy array
    labelstest = np.array(labelstest.tolist())

    for k in range(len(pred)):
        keep_points = []
        for j in range(len(body_parts_to_dataset1)):
            if isinstance(body_parts_to_dataset1[j], str):  # Skip unavailable keypoints
                continue
            else:
                keep_points.append(pred[k][body_parts_to_dataset1[j]])

        preds_test_rearranged.append(keep_points)

    # Convert preds_test_rearranged to a NumPy array
    preds_test_rearranged = np.array(preds_test_rearranged, dtype=np.float32)

    #print("Label Test Example:")
    #print(labelstest.shape)
    #print(labelstest[0])
    #print("Predictions Example:")
    #print(preds_test_rearranged.shape)

    print(RMSE(preds_test_rearranged,labelstest_modfied))
    print('Test AUC of the model on test images: {} %'.format(auc(preds_test_rearranged, labelstest_modfied, num_keypoints=13)))
    



######################################
# Test OpenPose sur les images JPG   #
######################################
def test_on_dataset_UCF_folder(folder_path):
    train_results_points = []

    # 1. Lister tous les fichiers .jpg dans le dossier et les trier numériquement
    # glob.glob permet de trouver tous les fichiers correspondant au motif
    image_files = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))
    
    if not image_files:
        print(f"Erreur : Aucune image trouvée dans {folder_path}")
        return

    print(f"Début du traitement de {len(image_files)} images...")

    # 2. Parcourir chaque fichier image trouvé
    for k, img_path in enumerate(image_files):
        # On passe directement le chemin du fichier JPG à predict_openpose
        # predict_openpose utilise cv.VideoCapture(input), qui accepte un chemin de fichier
        points = predict_openpose(img_path)
        
        train_results_points.append(points)
        
        # Affichage de la progression
        if k % 10 == 0 or k == len(image_files) - 1:
            print(f"Progression : {(k + 1) / len(image_files) * 100:.1f} %")

    # 3. Sauvegarder les keypoints extraits
    output_filename = os.path.basename(folder_path.strip('/')) + '_points.npy'
    np.save(output_filename, train_results_points)
    print(f"Résultats sauvegardés dans {output_filename}")
'''

########################
# Plot the prediction  #
########################
def plot_predic(image,points):
    plt.imshow(image.permute(1, 2, 0).numpy())
    plt.scatter(points[:,0],points[:,1])
    plt.show()


