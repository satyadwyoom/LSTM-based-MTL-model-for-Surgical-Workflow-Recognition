#### Load Libraries ####
import sys
import glob
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
import sys
from PIL import Image

import os
import pickle
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.models.resnet import resnet50
import torch.backends.cudnn as cudnn
import cv2  

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


path_of_output = sys.argv[4]
path_of_video = sys.argv[1]
path_of_kinm = sys.argv[2]


test_video = [i for i in sorted(os.listdir(path_of_video))]
test_kin = [i for i in sorted(os.listdir(path_of_kinm))]

assert len(test_video) > 0, "No video files were found in the input folder provided!!"
assert len(test_kin) > 0, "No kinematic files were found in the input folder provided!!"

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
        self.means = torch.tensor(means).cuda()
        self.sds = torch.tensor(sds).cuda()

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
    
    
# norm_layer = NormalizeLayer(means=[0.485, 0.456, 0.406],
#                             sds=[0.229, 0.224, 0.225])

# resnet50_imagenet = resnet50(pretrained=False).cuda()
# resnet50_img_weight_path = 'pretrained_model/checkpoint.pth.tar'
# checkpoint = torch.load(resnet50_img_weight_path)
# new_state_dict = OrderedDict()

# for k, v in checkpoint['state_dict'].items():
#     if k[:1]!=str(0):
#         name = k[9:] # remove `module.`
#         new_state_dict[name] = v

# resnet50_imagenet.load_state_dict(new_state_dict)
# resnet50_imagenet.fc = Identity()
# resnet50_imagenet = torch.nn.Sequential(norm_layer, resnet50_imagenet)

# for param50 in resnet50_imagenet.parameters():
#     param50.requires_grad = False
    
# print('resnet 50 loaded and layers freezed')
# resnet50_imagenet.cuda()
# print('Moved to GPU')
resnet50_imagenet = torch.load('./our_model_weights/model_petraw_resnet50_ex_t_depth_5_ex.pth.tar')




class petraw_model(nn.Module):
    def __init__(self, imagenet_extractor=None, input_size=28, output_size=None, time_depth=0):
        super(petraw_model, self).__init__()
        
        self.imagenet_extractor = imagenet_extractor
        
        
        self.lin_kin = nn.Linear(input_size, 64)
        self.linear = nn.Linear(64, 32)
        hidden_size = 32
        
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        
        
        ### Video Layers
        self.fc_ex1 = nn.Linear(2048, 256)
        self.fc_ex2 = nn.Linear(256, 64)
        ex_size = 64

        

        ## Video Frame-> Imagenet_extractor-> 2048 -> Fc_ex1 -> Fc_ex2 -> Fc_ex3 -> Ourt 1
        
        ### output-layers
        self.fc_F = nn.Linear(hidden_size + ex_size, output_size)
        self.drop1 = nn.Dropout(p=0.3)
        self.drop2 = nn.Dropout(p=0.3)
        self.drop3 = nn.Dropout(p=0.3)
        self.drop4 = nn.Dropout(p=0.3)
        


    def forward(self, input_seq, input_image):
        
        lin_out = self.lin_kin(input_seq)
        lin_out = self.relu1(lin_out)
        lin_out = self.drop1(lin_out)
        lin_out = self.linear(lin_out)
        lin_out = self.relu2(lin_out)
        lin_out = self.drop2(lin_out)
            
        if input_image is not None:
#             x = self.imagenet_extractor(input_image)
            x = self.fc_ex1(input_image)
            x = self.relu3(x)
            x = self.drop3(x)
            x = self.fc_ex2(x)
            x = self.relu4(x)
            x = self.drop4(x)
            a = torch.cat((lin_out, x), dim=1)
            y_f = self.fc_F(a)
        else:
            a = self.drop1(lin_out)
            y_f = self.fc_F(a) 

        return y_f



model_path = './our_model_weights/model_petraw_resnet50_ex_t_depth_5.pth.tar'
model_list = torch.load(model_path)

output_dict_list = [{0 :'Idle', 1: 'Transfer Left to Right', 2: 'Transfer Right to Left'},
                    {0: 'Idle', 1: 'Block 1 L2R', 2: 'Block 2 L2R', 3: 'Block 3 L2R', 4: 'Block 4 L2R', 5: 'Block 5 L2R', 6: 'Block 6 L2R', 7: 'Block 1 R2L', 8: 'Block 2 R2L', 9: 'Block 3 R2L', 10:'Block 4 R2L', 11:'Block 5 R2L', 12:'Block 6 R2L'},
                    {0: 'Idle', 1:'Catch', 2:'Extract', 3: 'Hold', 4: 'Drop', 5: 'Touch', 6: 'Insert'},
                    {0: 'Idle', 1:'Catch', 2:'Extract', 3: 'Hold', 4: 'Drop', 5: 'Touch', 6: 'Insert'}]



f_transform = T.Compose([T.Resize((224, 224)),
                         T.ToTensor()])
    
def read_video_kin_path(v_path, k_path, model_list, output_dict_list):

    output_list = [[],[],[],[]]
    kin_input = pd.read_csv(k_path, delimiter = '\t', index_col = 0)
    cap= cv2.VideoCapture(v_path)
    i=0
    t_dep_list = [[],[]]
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = f_transform(frame).unsqueeze(0)
        frame = resnet50_imagenet(frame).detach()
        kmatic = torch.from_numpy(kin_input.iloc[i].to_numpy()).unsqueeze(0).float()
        t_dep_list[0].append(kmatic)
        t_dep_list[1].append(frame)
        t_dep_list[0] = t_dep_list[0][-5:]
        t_dep_list[1] = t_dep_list[1][-5:]
        
        if len(t_dep_list[0])==5 and len(t_dep_list[1])==5:
            with torch.no_grad():
                for m in range(len(model_list)):
                    o_dict = output_dict_list[m]
                    model, _ = model_list[m]
                    model.to(device).eval()
                    kmatic_list = torch.cat(t_dep_list[0], dim=0)
                    frame_list = torch.cat(t_dep_list[1], dim=0)
    
                    out = model(kmatic_list, frame_list)
                    predicted_class = nn.Softmax(dim=1)(out).max(dim=1)[1]
                    predicted_value = [o_dict[k.cpu().item()] for k in predicted_class]
                    output_list[m].extend(predicted_value)
        else:
            for m in range(len(model_list)):
                o_dict = output_dict_list[m]
                predicted_class = torch.zeros(len(frame))
                predicted_value = [o_dict[k.cpu().item()] for k in predicted_class]
                output_list[m].extend(predicted_value)
                      
        i+=1
        
    cap.release()
    cv2.destroyAllWindows()
    
    return output_list


for (v, k) in zip(test_video, test_kin):
    v_path = os.path.join(path_of_video, v)
    k_path = os.path.join(path_of_kinm, k)
    predicted_output_val = read_video_kin_path(v_path, k_path, model_list, output_dict_list)
    predicted_output_dict = {'Frame': [i for i in range(len(predicted_output_val[0]))],
                             'Phase': predicted_output_val[0],
                             'Step': predicted_output_val[1],
                             'Verb_Left': predicted_output_val[2],
                             'Verb_Right': predicted_output_val[3]}
    predicted_output_df = pd.DataFrame.from_dict(predicted_output_dict)
    predicted_output_df.to_csv(os.path.join(path_of_output, v.split('.')[0]+'.txt'), index=False, sep='\t')
