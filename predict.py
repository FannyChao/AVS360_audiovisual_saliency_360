import re
import os

import torch
import numpy as np

from PIL import Image
from utils.process_video_audio import LoadVideoAudio

from model import DAVE
import pdb
import matplotlib.pyplot as plt
import cv2

# the folder find the videos consisting of video frames and the corredponding audio wav
VIDEO_TEST_FOLDER = './data_ICME20/'
# where to save the predictions
OUTPUT = '/media/fchang/Seagate Expansion Drive1/DAVE/evaluation/with_train_new/result_Dave_cp_eqcb/type14_a3/ep10/'
# where tofind the model weights
MODEL_PATH = '/media/fchang/Seagate Expansion Drive1/DAVE/log/Dave_cp_eqcb/type14_eqcb_a3/DAVE_ep10.pkl'
# some config parameters

IMG_WIDTH = 256
IMG_HIGHT = 320
TRG_WIDTH = 32
TRG_HIGHT = 40

device = torch.device("cuda:0")


class PredictSaliency(object):

    def __init__(self):
        super(PredictSaliency, self).__init__()

        self.video_list = [os.path.join(VIDEO_TEST_FOLDER, p) for p in os.listdir(VIDEO_TEST_FOLDER)]
        self.model = DAVE()
        self.model=torch.load(MODEL_PATH)
        self.output = OUTPUT
        if not os.path.exists(self.output):
                os.mkdir(self.output)
        self.model = self.model.cuda()
        self.model.eval()

    @staticmethod
    def _load_state_dict_(filepath):
        if os.path.isfile(filepath):
            pdb.set_trace()
            print("=> loading checkpoint '{}'".format(filepath))
            checkpoint = torch.load(filepath, map_location=device)
            
            pattern = re.compile(r'module+\.*')
            state_dict = checkpoint['state_dict']
            
            for key in list(state_dict.keys()):
                if 'video_branch' in key: 
                   state_dict[key[:12] + '_cubic' + key[12:]] = state_dict[key]
            
            for key in list(state_dict.keys()):
                res = pattern.match(key)
                if res:
                    new_key = re.sub('module.', '', key)
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]

        return state_dict

    def predict(self, stimuli_path, fps, out_path):

        equator_bias = cv2.resize(cv2.imread('ECB.png', 0), (10,8))
        equator_bias = torch.tensor(equator_bias).to(device, dtype=torch.float)
        equator_bias = equator_bias.cuda()
        equator_bias = equator_bias/equator_bias.max()

        if not os.path.exists(out_path):
            os.mkdir(out_path)
        if not os.path.exists(out_path+ '/overlay'):  
            os.mkdir(out_path + '/overlay')
        video_loader = LoadVideoAudio(stimuli_path, fps)
        vit = iter(video_loader)
        for idx in range(len(video_loader)):
            

            video_data_equi, video_data_cube, audio_data, AEM_data = next(vit)           # video_data_equi = [3, 16, 256, 512], video_data_cube = [6, 3, 16, 128, 128], audio_data = [3, 16, 64, 64], AEM_data = [1, 8, 16]
            print(idx, len(video_loader))
           
            video_data_equi = video_data_equi.to(device=device, dtype=torch.float)
            video_data_equi = video_data_equi.cuda()
            video_data_cube = video_data_cube.to(device=device, dtype=torch.float)
            video_data_cube = video_data_cube.cuda()
            AEM_data = AEM_data.to(device=device, dtype=torch.float)
            AEM_data = AEM_data.cuda()
            audio_data = audio_data.to(device=device, dtype=torch.float)
            audio_data = audio_data.cuda()
           
            prediction = self.model(video_data_equi, video_data_cube, audio_data, equator_bias)     # def forward(self, v_equi, v_cube, a,  aem, eq_b)
            
            saliency = prediction.cpu().data.numpy()
            saliency = np.squeeze(saliency)
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
            saliency = Image.fromarray((saliency*255).astype(np.uint8))
            saliency = saliency.resize((640, 480), Image.ANTIALIAS)
            saliency.save('{}/{}.jpg'.format(out_path, idx+1), 'JPEG')

    def predict_sequences(self):
        for v in self.video_list[:]:
            print(v)
            sample_rate = int(v[-2:])
            bname = os.path.basename(v[:-3])
            output_path = os.path.join(self.output, bname)
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            self.predict(v, sample_rate, output_path)


if __name__ == '__main__':

    p = PredictSaliency()
    # predict all sequences
    p.predict_sequences()
    # alternatively one can call directy for one video
    #p.predict(VIDEO_TO_LOAD, FPS, SAVE_FOLDER) # the second argument is the video FPS.

