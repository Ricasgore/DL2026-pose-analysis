################################################################################################################################
# We reused and drew inspiration from the work of misbah4064 : "https://github.com/misbah4064/human-pose-estimation-opencv"    #
################################################################################################################################

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from metrics import *

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]



#joints and points to keep for the first dataset
body_parts_to_dataset1 = [10, 9, 8, 11, 12, 13, 4, 3, 2, 5, 6, 7, 1, 'Head top' ]
joint_order_dataset1 = ['r ankle','r knee','2r hip','l hip','4l knee','l ankle','6r wrist','r elbow','8r shoulder','l shoulder','10l elbow','l wrist','12neck','Head top']
joint_connexion_dataset1 =[[0,1],[1,2],[2,3],[3,4],[4,5],[6,7],[7,8],[8,12],[9,12],[9,10],[10,11],[12,13]]

#joints and points to keep for the second dataset
body_parts_to_dataset2 = [10, 9, 8, 11, 12, 13, 'pelvis', 'thorax', 1 , 0 , 4, 3, 2, 5, 6, 7 ]
joint_order_dataset2 = ['r ankle','r knee','r hip','l hip','l knee','l ankle','pelvis','thorax','upper neck','head top','r wrist','r elbow','r shoulder','l shoulder','l elbow','l wrist']
joint_connexion_dataset2 =[[0,1],[1,2],[2,6],[6,3],[3,4],[4,5],[6,7],[7,8],[8,9],[8,12],[12,11],[11,10],[8,13],[13,14],[14,15]]



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


######################################
#test openpose on the second dataset #
######################################
def test_on_dataset2():
    datatest = np.load('../mpii_test_results_openpose/datatest.npy', allow_pickle=True) #we saved the images in a datatest.npy file but we can also use directly the image folder by modifying this line
    train_results_points = []

    for k in range(len(datatest)):
        test = datatest[k].permute(1, 2, 0).numpy()  # now : (128, 128, 3) and convert to numpy
        test = (test * 255).astype(np.uint8) #Converts values to uint8 (0-255)

        # Convert RGB en BGR for OpenCV
        test_bgr = cv.cvtColor(test, cv.COLOR_RGB2BGR)

        output_path = "temporary.jpg"
        cv.imwrite(output_path, test_bgr) #save the file temporary
        points = predict_openpose("temporary.jpg")
        train_results_points.append(points)
        print(k/len(datatest)*100, '%')

    np.save('../mpii_test_results_openpose/train_results_points.npy', train_results_points)

#############################################
# Compare the predictions with their labels #
#############################################
def compare_labels_dataset2():
    pred = np.load('../mpii_test_results_openpose/train_results_points.npy', allow_pickle=True)
    labelstest = np.load('../mpii_test_results_openpose/labelstest.npy', allow_pickle=True)

    preds_test_rearranged = []

    # Ensure labelstest elements are converted to NumPy and reshape properly
    for k in range(len(labelstest)):
        if isinstance(labelstest[k], torch.Tensor):
            labelstest[k] = labelstest[k].detach().cpu().numpy()  # Convert from tensor to NumPy
        
        labelstest[k] = np.delete(labelstest[k], [6, 7], axis=0)  # Remove unwanted keypoints

    # Ensure labelstest is a proper 3D NumPy array
    labelstest = np.array(labelstest.tolist())

    for k in range(len(pred)):
        keep_points = []
        for j in range(len(body_parts_to_dataset2)):
            if isinstance(body_parts_to_dataset2[j], str):  # Skip unavailable keypoints
                continue
            else:
                keep_points.append(pred[k][body_parts_to_dataset2[j]])

        preds_test_rearranged.append(keep_points)

    # Convert preds_test_rearranged to a NumPy array
    preds_test_rearranged = np.array(preds_test_rearranged, dtype=np.float32)

    #print("Label Test Example:")
    #print(labelstest.shape)
    #print(labelstest[0])
    #print("Predictions Example:")
    #print(preds_test_rearranged.shape)

    print(RMSE(preds_test_rearranged, labelstest))
    print('Test AUC of the model on test images: {} %'.format(auc(preds_test_rearranged, labelstest, num_keypoints=14)))



    



############################
#test on the first  dataset#
############################
def test_on_dataset():
    datatest = np.load('../lsp_test_results_openpose/datatest.npy', allow_pickle=True) #we saved the images in a datatest.npy file but we can also use directly the image folder by modifying this line
    train_results_points = []

    for k in range(len(datatest)):
        #print(datatest[0])
        test = datatest[k].permute(1, 2, 0).numpy()  # now : (128, 128, 3) and convert to numpy
        test = (test * 255).astype(np.uint8) #Converts values to uint8 (0-255)

        # Convert RGB en BGR for OpenCV
        test_bgr = cv.cvtColor(test, cv.COLOR_RGB2BGR)

        output_path = "temporary.jpg"
        cv.imwrite(output_path, test_bgr) #save the file temporary
        points = predict_openpose("temporary.jpg")
        train_results_points.append(points)
        print(k/len(datatest)*100, '%')

    np.save('../lsp_test_results_openpose/train_results_points.npy', train_results_points)


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
    


########################
# Plot the prediction  #
########################
def plot_predic(image,points):
    plt.imshow(image.permute(1, 2, 0).numpy())
    plt.scatter(points[:,0],points[:,1])
    plt.show()



############
#   MAIN   #
############
if __name__ == '__main__':

    #real time prediction
    #predict_openpose(input=0)

    #test_on_dataset2()
    #test_on_dataset()

    compare_labels_dataset2()
    #compare_labels_dataset1()


    #datatest2 = np.load('../mpii_test_results_openpose/datatest.npy', allow_pickle=True)
    #train_results_ploints2 = np.load('../mpii_test_results_openpose/train_results_points.npy', allow_pickle=True)

    #datatest1 = np.load('../lsp_test_results_openpose/datatest.npy', allow_pickle=True)
    #train_results_ploints1 = np.load('../lsp_test_results_openpose/train_results_points.npy', allow_pickle=True)

    #for k in range(len(datatest1)):
    #   plot_predic(datatest1[k],train_results_ploints1[k])
