from _operator import truediv
import matplotlib.pyplot as plt
import numpy as np
import os
import numpy as np
import torch

#expecting 1D np array of predictions and labels. 
def accuracy(predictions,labels):
    total = len(labels)
    correct = (predictions == labels).sum().item()
    return 100*truediv(correct,total)

def euclidian(predictions,labels):
    distance = np.sum((predictions - labels) ** 2)
    return np.mean(distance)


def auc(predictions, labels, threshold=10.0, num_keypoints=16):
 
    save_path = "./save/"
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    # Conversion to numpy 
    predictions = np.array(predictions)
    labels = np.array(labels)

    distances = np.sqrt((predictions - labels) ** 2) 

    thresholds = np.linspace(0, 50, 100)  
    ratios = []  

    for threshold in thresholds:
        correct_predictions = np.sum(distances < threshold)
        total_keypoints = predictions.shape[0] * num_keypoints*3
        ratio = correct_predictions / total_keypoints
        ratios.append(ratio)


    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, ratios, label="Ratio (AUC)", color="blue", linewidth=2)
    plt.xlabel("Threshold", fontsize=14)
    plt.ylabel("Ratio (AUC)", fontsize=14)
    plt.title("Ratio vs Threshold", fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=12)
    #recording results
    if save_path:
        plt.savefig(os.path.join(save_path, "global_auc.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    auc_area = np.trapz(ratios, thresholds)

    # Calculating ratios by keypoint
    # Rearrange into 14 1D arrays
    num_subtables = 16
    elements_per_subtable = 3

    # Rearrange and flatten each sub-table
    preds_by_keypoint = [predictions[:, i * elements_per_subtable : (i + 1) * elements_per_subtable].flatten() for i in range(num_subtables)]
    labels_by_keypoint = [labels[:, i * elements_per_subtable : (i + 1) * elements_per_subtable].flatten() for i in range(num_subtables)]

    ratios_per_keypoint = []  #List to store ratios for each keypoint
    for i in range(num_keypoints):
        preds_keypoint = predictions[:, i * elements_per_subtable : (i + 1) * elements_per_subtable]
        labels_keypoint = labels[:, i * elements_per_subtable : (i + 1) * elements_per_subtable]
        distances_keypoint = np.sqrt(np.sum((preds_keypoint - labels_keypoint) ** 2, axis=1))

        ratios_keypoint = []
        for threshold in thresholds:
            correct_predictions = np.sum(distances_keypoint < threshold)
            total_keypoints = predictions.shape[0]
            ratio = correct_predictions / total_keypoints
            ratios_keypoint.append(ratio)
    
        ratios_per_keypoint.append(ratios_keypoint)
    
    #Plot the curves for each keypoint
    plt.figure(figsize=(10, 7))
    for i, ratios in enumerate(ratios_per_keypoint):
        plt.plot(thresholds, ratios, label=f"Keypoint {i+1}", linewidth=2)

    # Graphic configuration
    plt.xlabel("Threshold", fontsize=14)
    plt.ylabel("Ratio (AUC)", fontsize=14)
    plt.title("Ratio vs Threshold for each Keypoint", fontsize=16)
    plt.legend(fontsize=10, loc='lower right')
    plt.grid(True)
    #recording results
    if save_path:
        plt.savefig(os.path.join(save_path, "keypoint_auc.png"), dpi=300, bbox_inches='tight')

    plt.show()

    # AUC calculation for each keypoint
    auc_per_keypoint = [np.trapz(ratios, thresholds) for ratios in ratios_per_keypoint]
    print(f"AUC for each keypoint : {auc_per_keypoint}")

    #curve area records 
    if save_path:
        with open(os.path.join(save_path, "auc_results.txt"), "w") as f:
            f.write(f"AUC Globale : {auc_area:.4f}\n")
            for i, auc_kp in enumerate(auc_per_keypoint, start=1):
                f.write(f"AUC Keypoint {i} : {auc_kp:.4f}\n")

    return auc_area, auc_per_keypoint

def MSE(predictions, labels):
    MSE = []
    for k in range(len(predictions)):
        image_mse = 0
        for j in range(len(predictions[k])):
            image_mse += (predictions[k][j] - labels[k][j])**2
        image_mse = image_mse/len(predictions[k])
        MSE.append(image_mse)
    print(MSE)
    MSE = np.mean(MSE)
    return MSE

def RMSE(predictions, labels):
    predictions = np.asarray(predictions, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)

    errors = np.sqrt(np.mean((predictions - labels) ** 2, axis=1))  # RMSE per image
    return np.mean(errors)