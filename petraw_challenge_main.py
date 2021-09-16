#### Load Libraries ####
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
import sys

from PIL import Image
import os
import pickle
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.models.resnet import resnet50
import torch.backends.cudnn as cudnn

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_gpu = torch.cuda.device_count()
print('num GPUs: {}'.format(num_gpu))


kin_data = pd.read_csv('dataset_path/inputs.csv', index_col = 0)
output_data  = pd.read_csv('dataset_path/outputs.csv', index_col = 0)
# frame_data = np.load('dataset_path/video_frames.npy') 
frame_data = pd.read_csv('dataset_path/Frames.csv', index_col = 0)
seg_data = pd.read_csv('dataset_path/Segmentation.csv', index_col = 0)


Phase_dict = {output_data['Phase'].unique()[i]:i for i in range(len(output_data['Phase'].unique()))}
Step_dict = {output_data['Step'].unique()[i]:i for i in range(len(output_data['Step'].unique()))}
Verb_Left_dict = {output_data['Verb_Left'].unique()[i]:i for i in range(len(output_data['Verb_Left'].unique()))}
Verb_Right_dict = Verb_Left_dict

print('Output Mapping Dictionary: \n')
print('Phase Dict:', Phase_dict)
print('Step Dict:', Step_dict)
print('Verb Left Dict:', Verb_Left_dict)
print('Verb Right Dict:', Verb_Right_dict)


output_data['Phase'] = output_data['Phase'].map(Phase_dict)
output_data['Step'] = output_data['Step'].map(Step_dict)
output_data['Verb_Left'] = output_data['Verb_Left'].map(Verb_Left_dict)
output_data['Verb_Right'] = output_data['Verb_Right'].map(Verb_Right_dict)



class GeneralVideoDataset(Dataset):
    """Dataset Class for Loading Video"""

    def __init__(
        self,
        seg=None,
        frame=None,
        kin=None,
        output=None,
        time_depth=None):
        """
        Args:
            seg (data_frame/numpy array):  DF during test time /numpy array containing all
                                            imagenet encoded frames (2048) during training
            frame (data_frame/numpy array): DF during test time /numpy array containing all
                                            imagenet encoded frames (2048) during training
            kin (data_frame): pandas DF containing Kinematic data
            output (data_frame): pandas DF containing Output data
            train (bool): False if using for Test data/ True for Train data
        """
        self.kin_df = kin
        self.output_df = output
        self.time_depth = time_depth
        self.frame_df = frame
        self.seg_df = seg
        self.transform = T.Compose([T.Resize((224, 224)),
                                    T.ToTensor()])
        
        assert len(self.seg_df) == len(self.frame_df) == len(self.kin_df) == len(self.output_df)
        ### For train dataset ###
            
        self.length_df = len(self.kin_df)-(time_depth - 1)

    def __len__(self):
        return self.length_df

    def read_image(self, df, idx):
        # Load the image file
        x = []
        for t in range(self.time_depth):
            image = Image.open(df.iloc[idx+t]['file_path'])
            x.append(self.transform(image).unsqueeze(0))
        
        x = torch.cat(x, dim=0)
        return x

    def __getitem__(self, idx):

        video_frame = self.read_image(self.frame_df, idx)           
        kinematic = torch.from_numpy(self.kin_df.iloc[idx: idx+self.time_depth].to_numpy()).float()
        output = torch.from_numpy(self.output_df.iloc[idx+self.time_depth-1].to_numpy())

        sample = {
            "video": video_frame,
            "Kinematic": kinematic,
            "output": output,
        }

        return sample
    
    
    
    

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
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2).cuda()
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2).cuda()
        return (input - means)/sds

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
    
norm_layer = NormalizeLayer(means=[0.485, 0.456, 0.406],
                            sds=[0.229, 0.224, 0.225])

resnet50_imagenet = resnet50(pretrained=False)
resnet50_img_weight_path = 'pretrained_model/checkpoint.pth.tar'
checkpoint = torch.load(resnet50_img_weight_path)
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
    
resnet50_imagenet.cuda()
# resnet50_imagenet = nn.DataParallel(resnet50_imagenet)
resnet50_imagenet.eval()
print('resnet 50 loaded and layers freezed')





class petraw_model(nn.Module):
    def __init__(self, imagenet_extractor=None, input_size=28, hidden_layer_size=64, hidden_size=32, output_size=None, train_p=True, check=True, time_depth=2):
        super(petraw_model, self).__init__()
                
        self.lstm_kin = nn.LSTM(input_size, 64)
        self.linear = nn.Linear(time_depth * 64, 32)
        hidden_size = 32
        
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        
        
        ### Video Layers
        self.lstm_ex1 = nn.LSTM(2048, 256)
#         self.fc_ex1 = nn.Linear(2048, 256)
        self.fc_ex2 = nn.Linear(time_depth * 256, 64)
        ex_size = 64

        

        ## Video Frame-> Imagenet_extractor-> 2048 -> Fc_ex1 -> Fc_ex2 -> Fc_ex3 -> Ourt 1
        
        ### output-layers
        self.fc_F = nn.Linear(hidden_size + ex_size, output_size)
        self.drop1 = nn.Dropout(p=0.3)
        self.drop2 = nn.Dropout(p=0.3)
        self.drop3 = nn.Dropout(p=0.3)
        self.drop4 = nn.Dropout(p=0.3)
        


    def forward(self, input_seq, input_image):
        
        lin_out, _ = self.lstm_kin(input_seq)
        lin_out = lin_out.view(len(lin_out), -1)
        lin_out = self.relu1(lin_out)
        lin_out = self.drop1(lin_out)
        lin_out = self.linear(lin_out)
        lin_out = self.relu2(lin_out)
        lin_out = self.drop2(lin_out)
            
        if input_image is not None:
#             x = self.imagenet_extractor(input_image)
            x, _ = self.lstm_ex1(input_image)
            x = x.view(len(x), -1)
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

    
    
kin_train, kin_test, frame_train, frame_test, seg_train, seg_test, output_train, output_test = train_test_split(kin_data, frame_data, seg_data, output_data, test_size=0.30919206809643823, shuffle=False)

bs = 64
t_depth = 5
train_data = GeneralVideoDataset(kin=kin_train, 
                                 frame=frame_train, 
                                 seg=seg_train, 
                                 output=output_train,
                                 time_depth = t_depth)
#                                  train=True)
train_loader = DataLoader(train_data, batch_size = bs, shuffle=True, num_workers=23)

### test acts as validation ###
test_data = GeneralVideoDataset(kin=kin_test, 
                                frame=frame_test, 
                                seg=seg_test, 
                                output=output_test,
                                time_depth = t_depth)
#                                 train=True)
test_loader = DataLoader(test_data, batch_size = bs, shuffle=True, num_workers=23)


def calculate_loss(out, y, criterion):
    ## y (1x 4)
    ## y.T (4x1)
    l1 = torch.tensor(0.0).cuda()
    for (a,b) in zip(out, y.T):
        l1 += criterion(a, b)
    return l1

def calculate_correct(out, y):
    correct = []
    for (a,b) in zip(out, y.T):
        pred = nn.Softmax(dim=1)(a).max(dim=1)[1]
        correct.append(torch.eq(pred, b).sum().item())
    return correct

def encode_frames(image_b, feature_extractor):
    with torch.no_grad():
        b_shape = image_b.shape
        image_b = image_b.reshape(b_shape[0]*b_shape[1], b_shape[2], b_shape[3], b_shape[4])
        image_b_encoded = feature_extractor(image_b).detach()
        encoded_shape = image_b_encoded.shape
        image_b_encoded = image_b_encoded.reshape(b_shape[0], b_shape[1], encoded_shape[1])
        
    return image_b_encoded

def train(epochs, train_loader, test_loader, model_list, criterion, print_acc_epoch, name):
    sys.stdout.write("\n")
    for i in range(epoch):
        
        for m in range(len(model_list)):
            model_list[m][0] = model_list[m][0].train()
            
        epoch_loss_train = [0, 0, 0, 0]
        epoch_loss_test = [0, 0, 0, 0]

        total_train = 0
        correct_train = [0, 0, 0, 0]
        total_test = 0
        correct_test = [0, 0, 0, 0]

        for batch_idx, data_dict in enumerate(train_loader):
#             sys.stdout.write("\rBatch: {}".format(batch_idx+1))
#             sys.stdout.flush()
            
            X1, X2, y = data_dict['Kinematic'].cuda(), data_dict['video'].cuda(), data_dict['output'].cuda()
            ## PASS through feature-extractor ##
            X2 = encode_frames(X2, resnet50_imagenet)
                
            output_list = []
            for m in range(len(model_list)):
                model, optimizer = model_list[m]
                out = model(X1, X2)
                loss = criterion(out, y.T[m])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss_train[m] += loss.item()
                output_list.append(out.detach())
                model_list[m] = [model, optimizer]

            total_train += len(y)
            new_corr = calculate_correct(output_list, y)
            correct_train = [correct_train[i] + new_corr[i] for i in range(len(correct_train))]

        epoch_loss_train = [round(epoch_loss_train[k]/(batch_idx+1), 2) for k in range(len(epoch_loss_train))]
        train_acc = [round((correct_train[i]/total_train), 2) * 100 for i in range(len(correct_train))]
        
        model_save_dir = 'trained_models/'
        if not os.path.isdir(model_save_dir):
            os.makedirs(model_save_dir)
            
        model_save_path = model_save_dir + name + '.pth.tar'
        torch.save(model_list, model_save_path)
        ex_save_path = model_save_dir + name + '_ex.pth.tar'
        torch.save(resnet50_imagenet, ex_save_path)
        
        if (i+1)%print_acc_epoch!=0:
            sys.stdout.write('\nEpoch: {}\n'.format(i+1))
            sys.stdout.write('Train tasks ACC: {}\n'.format(train_acc))
            sys.stdout.write('Train correct : {}, Train total: {}\n'.format(correct_train,total_train))
            sys.stdout.write('Train Loss: {}\n'.format(epoch_loss_train))
            
        if (i+1)%print_acc_epoch==0:
            with torch.no_grad():
                
                for batch_idx, data_dict in enumerate(test_loader):
                    X1, X2, y = data_dict['Kinematic'].cuda(), data_dict['video'].cuda(), data_dict['output'].cuda()
                    ## PASS through feature-extractor ##
                    X2 = encode_frames(X2, resnet50_imagenet)
                        
                    output_list = []
                    for m in range(len(model_list)):
                        model, optimizer = model_list[m]
                        model.eval()
                        out = model(X1, X2)
                        loss = criterion(out, y.T[m])

                        epoch_loss_test[m] += loss.item()
                        output_list.append(out.detach())

                    total_test += len(y)
                    new_corr = calculate_correct(output_list, y)
                    correct_test = [correct_test[i] + new_corr[i] for i in range(len(correct_test))]

            epoch_loss_test = [round(epoch_loss_test[k]/(batch_idx+1), 2) for k in range(len(epoch_loss_test))]
            test_acc = [round((correct_test[i]/total_test), 2) * 100 for i in range(len(correct_test))]
        
            sys.stdout.write('\nEpoch: {}\n'.format(i+1))
            sys.stdout.write('Train tasks ACC: {}\n'.format(train_acc))
            sys.stdout.write('Train correct : {}, Train total: {}\n'.format(correct_train,total_train))
            sys.stdout.write('Train Loss: {}\n'.format(epoch_loss_train))
            sys.stdout.write('Test tasks ACC: {}\n'.format(test_acc))
            sys.stdout.write('Test correct : {}, Test total: {}\n'.format(correct_test,total_test))
            sys.stdout.write('Test Loss: {}\n'.format(epoch_loss_test))



            
models_opt_list = []
output_size_list = [3,13,7,7]
models_lr_list = [0.001, 0.001, 0.01, 0.01]

for i in range(4):
    kbm = petraw_model(output_size = output_size_list[i],
                       check=False, 
                       time_depth=t_depth).cuda()
#     kbm = nn.DataParallel(kbm)
    cudnn.benchmark = True

    optimizer = optim.Adam(params = kbm.parameters(),
                           lr = models_lr_list[i])
    
    models_opt_list.append([kbm, optimizer])
    
    

epoch = 30
criterion = torch.nn.CrossEntropyLoss(reduction='mean')


train(epoch, train_loader, test_loader, models_opt_list, criterion, 5, 'model_petraw_resnet50_ex_t_depth_5')




