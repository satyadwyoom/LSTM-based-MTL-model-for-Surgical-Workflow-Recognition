from __future__ import print_function, division
from collections import OrderedDict
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from torch import nn, optim, as_tensor
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from PIL import Image
import os
import pickle
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.models.resnet import resnet18, resnet34, resnet50

device = torch.device("cpu")

frame_data = pd.read_csv('dataset_path/Frames.csv', index_col = 0) 
seg_data = pd.read_csv('dataset_path/Segmentation.csv', index_col = 0)

class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means, sds):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means).cpu()
        self.sds = torch.tensor(sds).cpu()

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means)/sds

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
    
norm_layer = NormalizeLayer(means=[0.485, 0.456, 0.406],
                            sds=[0.229, 0.224, 0.225])

resnet50_imagenet = resnet50(pretrained=False).cpu()
resnet50_img_weight_path = 'pretrained_model/checkpoint.pth.tar'
checkpoint = torch.load(resnet50_img_weight_path, map_location=torch.device('cpu'))
new_state_dict = OrderedDict()

for k, v in checkpoint['state_dict'].items():
    if k[:1]!=str(0):
        name = k[9:] # remove `module.`
        new_state_dict[name] = v

resnet50_imagenet.load_state_dict(new_state_dict)
resnet50_imagenet.fc = Identity()
resnet50_imagenet = torch.nn.Sequential(norm_layer, resnet50_imagenet)

for param50 in resnet50_imagenet.parameters():
    param50.requires_grad = False
    
print('resnet 50 loaded and layers freezed')



def read_image(df, idx, resnet50):
    # Load the image file
    transform = T.Compose([T.Resize((224, 224)),
                            T.ToTensor()])
    image = Image.open(df.iloc[idx]['file_path'])
    x = transform(image)
    x = x.unsqueeze_(0)
    out = resnet50(x)

    return out.squeeze(0).numpy()


##Main Code to iterate the directory and make encoded frames for all video/segmentation frames

for i in range(2):
    if i==0:
        data_frames = frame_data
        main_path = '/arc/project/st-anaray02-1/skumar40/petraw_data/Training/Encoded_Frames'
    else:
        data_frames = seg_data
        main_path = '/arc/project/st-anaray02-1/skumar40/petraw_data/Training/Encoded_Segmentation'
        
    for idx in range(len(data_frames)):
        file_path = data_frames.iloc[idx]['file_path']
        file_name = file_path.split('/')[-1].split('.')[0]
        file_folder_name = str(data_frames.iloc[idx]['folder_name'])
        encoded_frame = read_image(data_frames, idx, resnet50_imagenet)
        
        save_file_folder_path = os.path.join(main_path, file_folder_name)
        if not os.path.isdir(save_file_folder_path):
            os.makedirs(save_file_folder_path)
            
        new_file_path = os.path.join(save_file_folder_path, file_name + '.npy')
        np.save(new_file_path, encoded_frame)
        print('\ridx: {}'.format(idx+1), end="")
    print('\nMoving to next data_frame')

            
        
        
        