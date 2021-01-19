import re
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
from utils.process_video_audio import LoadVideoAudio_TRAIN
from model_train import DAVE
import pdb
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import cv2

# the folder find the videos consisting of video frames and the corredponding audio wav
#VIDEO_TRAIN_FOLDER = './data_ICME20/'
VIDEO_TRAIN_FOLDER = '../dataset/GazePredi360_audiovisual-related/frame/'
# where to save the predictions
OUTPUT = 'type14_eqcb_a3/'
# where tofind the model weights
MODEL_PATH = '../dataset/model.pth.tar'

# some config parameters

IMG_WIDTH = 256
IMG_HIGHT = 320
TRG_WIDTH = 32
TRG_HIGHT = 40

device = torch.device("cuda")

loss_function = nn.KLDivLoss()
loss_function_bce = nn.BCELoss()
nb_epoch = 30



class TrainSaliency(object):

    def __init__(self):
        super(TrainSaliency, self).__init__()

        self.video_list = [os.path.join(VIDEO_TRAIN_FOLDER, p) for p in os.listdir(VIDEO_TRAIN_FOLDER)]
        self.video_list = self.video_list
        self.video_list.sort()
        # pdb.set_trace()
        self.model = DAVE()
        self.model = self.model.to(device)
        # self.model = self.model.cuda()

        self.model.load_state_dict(self._load_state_dict_(MODEL_PATH), strict=True)
        
        self.output = OUTPUT
        if not os.path.exists(self.output):
                os.mkdir(self.output)
        
       

        #self.model.eval()
    
    @staticmethod
    def _load_state_dict_(filepath):
        if os.path.isfile(filepath):
            print("=> loading checkpoint '{}'".format(filepath))
            checkpoint = torch.load(filepath, map_location=device)

            pattern = re.compile(r'module+\.*')
            state_dict = checkpoint['state_dict']
            #new_state_dict = {k : v for k, v in state_dict.items() if 'video_branch' in k}
            for key in list(state_dict.keys()):
                if 'video_branch' in key:        
                   state_dict[key[:12] + '_cubic' + key[12:]] = state_dict[key]
                
                if 'combinedEmbedding' in key: 
                    state_dict[key[:17] + '_equi_cp' + key[17:]] = state_dict[key]
                
            #pdb.set_trace()
            for key in list(state_dict.keys()):
                res = pattern.match(key)
                if res:
                    print('Y', key)
                    new_key = re.sub('module.', '', key)
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]
        return state_dict

    def train(self):
        equator_bias = cv2.resize(cv2.imread('ECB.png', 0), (10,8))
        equator_bias = torch.tensor(equator_bias).to(device, dtype=torch.float)
        equator_bias = equator_bias.cuda()
        equator_bias = equator_bias/equator_bias.max()

        optimizer = optim.Adam(self.model.parameters(), lr=1e-5)
        v_num = len(self.video_list)
       
        for epoch in tqdm(range(nb_epoch)):
            self.model.train()
            epoch_loss = 0.0
            batch_count = 0.0
            start = time.time()

            for n, v in enumerate(self.video_list[:]):
                fps = 25
                stimuli_path = v
                video_loader = LoadVideoAudio_TRAIN(stimuli_path, fps)
                vit = iter(video_loader)
                start = time.time()
                for idx in range(len(video_loader)):
                    video_data_equi, video_data_cube, audio_data, gt_salmap = next(vit)
                    
                    video_data_equi = video_data_equi.to(device=device, dtype=torch.float)
                    video_data_equi = video_data_equi.cuda()
                    video_data_cube = video_data_cube.to(device=device, dtype=torch.float)
                    video_data_cube = video_data_cube.cuda()
                    audio_data = audio_data.to(device=device, dtype=torch.float)
                    audio_data = audio_data.cuda()
                    
                    gt_salmap = gt_salmap.to(device=device, dtype=torch.float)
                    gt_salmap = gt_salmap.cuda()
                    
                    pred_salmap = self.model(video_data_equi, video_data_cube, audio_data, equator_bias)
                    loss = loss_function_bce(pred_salmap, gt_salmap)
                    
                    epoch_loss += loss.cpu().data.numpy()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
                batch_count = batch_count + len(video_loader)
                end = time.time() 
              
            print("=== Epoch {%s}  Loss: {%.8f}  Running time: {%4f}" % (str(epoch), (epoch_loss)/batch_count, end - start))
            
            
            if epoch % 1 == 0:
                torch.save(self.model, OUTPUT + 'DAVE_ep' + str(epoch) + '.pkl')

if __name__ == '__main__':

    t = TrainSaliency()
    t.train()
  
