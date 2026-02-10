import torch
from torchvision import transforms
from PIL import Image
from _operator import truediv
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import os

class LSPPE(torch.utils.data.Dataset):
    def __init__(self, dataDir='./lsp_dataset', transform=None, crossNum=None,crossIDs=None, theType='train'):

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
    def __init__(self, dataDir='/home/amine_tsp/DL2026/Datasets/mpii', transform=None, crossNum=None, crossIDs=None, theType='test'):

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

if __name__ == '__main__':
    
    
    tr = transforms.Compose([
        #transforms.Resize((64,64)),
        transforms.ToTensor()
        ])

    
    lsppe = LSPPE(transform=tr,crossNum=5, crossIDs=[5])
    #print(len(lsppe))

    images,labels = next(iter(lsppe))
    #print(images,labels)
    #print(images.shape)
    #print(labels.shape)

    #for i, (imaages,labels) in enumerate(lsppe):
    #    print(images.shape)
    #    print(labels.shape)

    exit(0)