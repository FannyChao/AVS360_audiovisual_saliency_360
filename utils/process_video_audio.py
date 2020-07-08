
# generic must imports
import os
import torch
import numpy as np
import cv2

import utils.audio_params as audio_params
import librosa as sf
from utils.audio_features import waveform_to_feature
import matplotlib.pyplot as plt

from PIL import Image
import torchvision.transforms.functional as F
from utils.equi_to_cube import Equi2Cube
import pdb
import random

__all__ = ['LoadVideoAudio']

#defined params @TODO move them to a parameter config file
DEPTH = 16
GT_WIDTH = 32
GT_HIGHT = 40

e2c = Equi2Cube(128, 256, 512)     # Equi2Cube(out_w, in_h, in_w) 

MEAN = [ 110.63666788 / 255.0, 103.16065604 / 255.0, 96.29023126 / 255.0 ]
STD = [ 38.7568578 / 255.0, 37.88248729 / 255.0, 40.02898126 / 255.0 ]

def adjust_len(a, b):
    # adjusts the len of two sorted lists
    al = len(a)
    bl = len(b)
    if al > bl:
        start = (al - bl) // 2
        end = bl + start
        a = a[start:end]
    if bl > al:
        a, b = adjust_len(b, a)
    return a, b


def create_data_packet(in_data, frame_number):

    n_frame = in_data.shape[0]

    frame_number = min(frame_number, n_frame) #if the frame number is larger, we just use the last sound one heard about
    starting_frame = frame_number - DEPTH + 1
    starting_frame = max(0, starting_frame) #ensure we do not have any negative frames
    data_pack = in_data[starting_frame:frame_number+1, :, :]
    n_pack = data_pack.shape[0]

    if n_pack < DEPTH:
        nsh = DEPTH - n_pack
        data_pack = np.concatenate((np.tile(data_pack[0,:,:], (nsh, 1, 1)), data_pack), axis=0)

    assert data_pack.shape[0] == DEPTH

    data_pack = np.tile(data_pack, (3, 1, 1, 1))

    return data_pack, frame_number


def load_wavfile(total_frame, wav_file):
    """load a wave file and retirieve the buffer ending to a given frame

    Args:
      wav_file: String path to a file, or a file-like object. The file
      is assumed to contain WAV audio data with signed 16-bit PCM samples.

      frame_number: Is the frame to be extracted as the final frame in the buffer

    Returns:
      See waveform_to_feature.
    """
    wav_data, sr = sf.load(wav_file, sr=audio_params.SAMPLE_RATE, dtype='float32')
    assert sf.get_duration(wav_data, sr) > 1
    
    features = waveform_to_feature(wav_data, sr)
    features = np.resize(features, (int(total_frame), features.shape[1], features.shape[2]))

    return features


def get_wavFeature(features, frame_number):
    
    audio_data, valid_frame_number = create_data_packet(features, frame_number)
    return torch.from_numpy(audio_data).float(), valid_frame_number


def load_maps(file_path):
    '''
        Load the gt maps
    :param file_path: path the the map
    :return: a numpy array as floating number
    '''

    with open(file_path, 'rb') as f:
        with Image.open(f) as img:
            img = img.convert('L').resize((GT_HIGHT, GT_WIDTH), resample=Image.BICUBIC)
            data = F.to_tensor(img)
    return data


def load_video_frames(end_frame, frame_number, valid_frame_number):
    # load video frames, process them and return a suitable tensor
    frame_path, frame_name = os.path.split(end_frame)
    assert int(frame_name[0:-4]) == frame_number
    frame_number = min(frame_number, valid_frame_number)
    start_frame_number = frame_number - DEPTH+1
    start_frame_number = max(0, start_frame_number)
    frame_list = [f for f in range(start_frame_number, frame_number+1)]
    if len(frame_list) < DEPTH:
        nsh = DEPTH - len(frame_list)
        frame_list = np.concatenate((np.tile(frame_list[0], (nsh)), frame_list), axis=0)
    
    frames_cube = []
    frames_equi = []
    
    for i in range(len(frame_list)):
        imgpath = os.path.join(frame_path, '{0:04d}.{1:s}'.format(frame_list[i], frame_name[-3:]))
        with open(imgpath, 'rb') as f:
            with Image.open(f) as img:
                #img = img.convert('RGB')
                img = cv2.resize(cv2.imread(imgpath), (512,256))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.0
                #pix=np.array(img.getdata()).reshape(256,320,3)/255.0 
                
                img_c = e2c.to_cube(img)
                img_cube=[]
                for face in range(6):
                    img_f = F.to_tensor(img_c[face])
                    img_f = F.normalize(img_f, MEAN, STD)
                    img_cube.append(img_f)
                img_cube_data = torch.stack(img_cube)               
                frames_cube.append(img_cube_data)
                
                img = cv2.resize(img, (320,256))
                img_equi = F.to_tensor(img)
                img_equi = F.normalize(img_equi, MEAN, STD)
                frames_equi.append(img_equi)
                
    data_cube = torch.stack(frames_cube, dim=0)
    data_equi = torch.stack(frames_equi, dim=0)
    '''
    frames = []
    for i in range(len(frame_list)):
        imgpath = os.path.join(frame_path, '{0:07d}.{1:s}'.format(frame_list[i], frame_name[-3:]))
        with open(imgpath, 'rb') as f:
            with Image.open(f) as img:
                #pdb.set_trace()
                #img = img.convert('RGB')
                img = cv2.resize(cv2.imread(imgpath), (512,256))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.0
                img = F.to_tensor(img)
                img = F.normalize(img, MEAN, STD)
                frames.append(img)
    data_equi = torch.stack(frames, dim=0)
    '''
    return data_equi.permute([1, 0, 2, 3]), data_cube.permute([1, 2, 0, 3, 4])
    #return data_cube.permute([1, 2, 0, 3, 4])


def load_AEM_frames(end_frame, frame_number, valid_frame_number):
    # load video frames, process them and return a suitable tensor
    frame_path, frame_name = os.path.split(end_frame)
    #pdb.set_trace()
    assert int(frame_name[0:-4]) == frame_number
    frame_number = min(frame_number, valid_frame_number)
    start_frame_number = frame_number - DEPTH+1
    start_frame_number = max(0, start_frame_number)
    frame_list = [f for f in range(start_frame_number, frame_number+1)]
    if len(frame_list) < DEPTH:
        nsh = DEPTH - len(frame_list)
        frame_list = np.concatenate((np.tile(frame_list[0], (nsh)), frame_list), axis=0)
    frames = np.zeros((8, 10))
    count = 0.0

    for i in range(len(frame_list)):
        imgpath = os.path.join(frame_path, '{0:04d}.{1:s}'.format(frame_list[i], frame_name[-3:]))    
        try:
            img = cv2.resize(cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE), (10, 8))
            img = img/255.0
            frames = frames + img
            count = count + 1
        except:
            continue
                    
    frames = frames/count
    if frames.sum()>0:
        frames = frames/frames.max()
    frames = F.to_tensor(frames)
    #pdb.set_trace()
    #data = torch.stack(frames, dim=0)
    return frames


def load_gt_frames(end_frame, frame_number, valid_frame_number):
    # load video frames, process them and return a suitable tensor
    frame_path, frame_name = os.path.split(end_frame)
    assert int(frame_name[0:-4]) == frame_number
    frame_number = min(frame_number, valid_frame_number)
    start_frame_number = frame_number - DEPTH+1
    start_frame_number = max(0, start_frame_number)
    frame_list = [f for f in range(start_frame_number, frame_number+1)]
    if len(frame_list) < DEPTH:
        nsh = DEPTH - len(frame_list)
        frame_list = np.concatenate((np.tile(frame_list[0], (nsh)), frame_list), axis=0)
    frames = np.zeros((32, 40))  # (32,64)
    count = 0.0
    for i in range(len(frame_list)):
        imgpath = os.path.join(frame_path, '{0:04d}.{1:s}'.format(frame_list[i], frame_name[-3:]))    
        try:
            img = cv2.resize(cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE), (40, 32)) # (64, 32)
            #img = cv2.GaussianBlur(img, (7,7),cv2.BORDER_DEFAULT)
            img = img/255.0
            frames = frames + img
            count = count + 1
        except:
            continue
                    
    frames = frames/count
    frames = frames/frames.sum()
    frames = F.to_tensor(frames)
    #pdb.set_trace()
    #data = torch.stack(frames, dim=0)
    return frames




class LoadVideoAudio(object):
    """
        load the audio video
    """

    def __init__(self, stimuli_in, vfps):
        """
        :param stimuli_in:
        :param gt_in:
        """

        #self.root_folder = stimuli_in + '/frames/'
        self.root_folder = stimuli_in
        self.sample = []
        self.batch_size = 1
        fr = vfps

        video_frames = [os.path.join(self.root_folder, f) for f in os.listdir(self.root_folder)
                        if f.endswith(('.jpg', '.jpeg', '.png'))]
        video_frames.sort()
        total_frame = str(len(video_frames))
        
        audio_file = [os.path.join(self.root_folder, f) for f in os.listdir(self.root_folder)
                      if f.endswith('.wav')]
        
        self.audio_data = load_wavfile(total_frame, audio_file[0])

        
        #pdb.set_trace()  
        cnt = 0
        for video_frame in video_frames:
            frame_number = os.path.basename(video_frame)[0:-4]
            sample = {'total_frame': total_frame, 'fps': fr,
                      'frame': video_frame, 'frame_number': frame_number}
            self.sample.append(sample)
            cnt = cnt + 1

 
        vid_name = stimuli_in.split('mono')[-1][:-3]
        self.root_folder_AEM = '/media/fchang/My Passport/ICME2020/sound_map/' + vid_name[2:] + '/frame/'
        self.sample_AEM = []
        AEM_frames = [os.path.join(self.root_folder_AEM, f) for f in os.listdir(self.root_folder_AEM)
                        if f.endswith(('.jpg', '.jpeg', '.png'))]
        AEM_frames.sort()
        cnt = 0
        total_frame = str(len(AEM_frames))
        for AEM_frame in AEM_frames:
            frame_number = os.path.basename(AEM_frame)[0:-4]
            sample_AEM = {'total_frame': total_frame, 'fps': fr,
                      'frame': AEM_frame, 'frame_number': frame_number}
            self.sample_AEM.append(sample_AEM)
            cnt = cnt + 1
            

    def __len__(self):
        #return len(self.sample)
        return int(len(self.sample)/self.batch_size)

    def __getitem__(self, item):

        sample = self.sample[item : item + self.batch_size]
        sample_AEM = self.sample_AEM[item : item + self.batch_size]
        
        video_data_equi_batch = []
        video_data_cube_batch = []
        audio_data_batch = []
        AEM_data_batch = []
        #gt_data_batch = []

        for i in range(self.batch_size):
            audio_params.EXAMPLE_HOP_SECONDS = 1/int(sample[i]['fps'])
            audio_data, valid_frame_number = get_wavFeature(self.audio_data, int(sample[i]['frame_number']))
            audio_data_batch.append(audio_data)

            video_data_equi, video_data_cube = load_video_frames(sample[i]['frame'], int(sample[i]['frame_number']), valid_frame_number)
            #video_data_cube = load_video_frames(sample[i]['frame'], int(sample[i]['frame_number']), valid_frame_number)
            video_data_equi_batch.append(video_data_equi)
            video_data_cube_batch.append(video_data_cube)

            AEM_data = load_AEM_frames(sample_AEM[i]['frame'], int(sample_AEM[i]['frame_number']), valid_frame_number)
            AEM_data_batch.append(AEM_data)
            
            #gt_data = load_gt_frames(sample[i]['gtsal_frame'], int(sample[i]['frame_number']), valid_frame_number)
            #gt_data_batch.append(gt_data)

        video_data_equi_batch = torch.stack(video_data_equi_batch, dim=0)     # [10, 3, 16, 256, 512]
        video_data_cube_batch = torch.stack(video_data_cube_batch, dim=0)     # [10, 6, 3, 16, 128, 128]
        audio_data_batch = torch.stack(audio_data_batch, dim=0)               # [10, 3, 16, 64, 64]
        AEM_data_batch = torch.stack(AEM_data_batch, dim=0)                   # [10, 1, 8, 16]
        #gt_data_batch = torch.stack(gt_data_batch, dim=0)                     # [10, 1, 8, 16]

        return video_data_equi_batch, video_data_cube_batch, audio_data_batch, AEM_data_batch


class LoadVideoAudio_TRAIN(object):
    """
        load the audio video
    """

    def __init__(self, stimuli_in, vfps):
        """
        :param stimuli_in:
        :param gt_in:
        """

        #self.root_folder = stimuli_in + '/frames/'
        self.batch_size = 6
        self.root_folder = stimuli_in
        self.gt_folder = 'GazePredi360_audiovisual-related/salmap/' + stimuli_in.split('/')[-1]
        self.sample = []
        self.max_frame_num = 250
        fr = vfps
        
        video_frames = [os.path.join(self.root_folder, f) for f in os.listdir(self.root_folder)
                        if f.endswith(('.jpg', '.jpeg', '.png'))]
        video_frames.sort()
        total_frame = str(len(video_frames))

        audio_file = [os.path.join(self.root_folder, f) for f in os.listdir(self.root_folder)
                      if f.endswith('.wav')]
        self.audio_data = load_wavfile(total_frame, audio_file[0])
        
        gtsal_frames = [os.path.join(self.gt_folder, f) for f in os.listdir(self.root_folder)
                        if f.endswith(('.jpg', '.jpeg', '.png'))]
        gtsal_frames.sort()

        cnt = 0
        '''
        video_frames = video_frames[:self.max_frame_num]
        random.shuffle(video_frames)
        video_frames = video_frames[:50]
        '''
        for v, video_frame in enumerate(video_frames[:self.max_frame_num]):
            frame_number = os.path.basename(video_frame)[0:-4]
            gtsal_frame = gtsal_frames[v]
            sample = {'total_frame': total_frame, 'fps': fr,
                      'frame': video_frame, 'gtsal_frame': gtsal_frame, 'frame_number': frame_number}
            self.sample.append(sample)
            cnt = cnt + 1
        
        #random.shuffle(self.sample)
        self.sample = self.sample[::5]

        '''
        # AEM
        vid_name = stimuli_in.split('mono')[-1][:-3]
        self.root_folder_AEM = '/media/fchang/My Passport/ICME2020/sound_map/' + vid_name[2:] + '/frame/'
        self.sample_AEM = []
        AEM_frames = [os.path.join(self.root_folder_AEM, f) for f in os.listdir(self.root_folder_AEM)
                        if f.endswith(('.jpg', '.jpeg', '.png'))]
        AEM_frames.sort()
        cnt = 0
        total_frame = str(len(AEM_frames))
        for AEM_frame in AEM_frames:
            frame_number = os.path.basename(AEM_frame)[0:-4]
            sample_AEM = {'total_frame': total_frame, 'fps': fr,
                      'frame': AEM_frame, 'frame_number': frame_number}
            self.sample_AEM.append(sample_AEM)
            cnt = cnt + 1
        '''    

    def __len__(self):
        return int(len(self.sample)/self.batch_size)

    def __getitem__(self, item):
    
        sample = self.sample[item : item + self.batch_size]
        #sample_AEM = self.sample_AEM[item : item + self.batch_size]
        
        video_data_equi_batch = []
        video_data_cube_batch = []
        audio_data_batch = []
        #AEM_data_batch = []
        gt_data_batch = []
        
        for i in range(self.batch_size):
            audio_params.EXAMPLE_HOP_SECONDS = 1/int(sample[i]['fps'])
            audio_data, valid_frame_number = get_wavFeature(self.audio_data, int(sample[i]['frame_number']))
            audio_data_batch.append(audio_data)

            video_data_equi, video_data_cube = load_video_frames(sample[i]['frame'], int(sample[i]['frame_number']), valid_frame_number)
            #video_data_cube = load_video_frames(sample[i]['frame'], int(sample[i]['frame_number']), valid_frame_number)
            video_data_equi_batch.append(video_data_equi)
            video_data_cube_batch.append(video_data_cube)

            #AEM_data = load_AEM_frames(sample_AEM[i]['frame'], int(sample[i]['frame_number']), valid_frame_number)
            #AEM_data_batch.append(AEM_data)
            
            gt_data = load_gt_frames(sample[i]['gtsal_frame'], int(sample[i]['frame_number']), valid_frame_number)
            gt_data_batch.append(gt_data)

        
        video_data_equi_batch = torch.stack(video_data_equi_batch, dim=0)     # [10, 3, 16, 256, 512]
        video_data_cube_batch = torch.stack(video_data_cube_batch, dim=0)     # [10, 6, 3, 16, 128, 128]
        audio_data_batch = torch.stack(audio_data_batch, dim=0)               # [10, 3, 16, 64, 64]
        #AEM_data_batch = torch.stack(AEM_data_batch, dim=0)                   # [10, 1, 8, 16]
        gt_data_batch = torch.stack(gt_data_batch, dim=0)                     # [10, 1, 8, 16]

        #return video_data_equi_batch, video_data_cube_batch, audio_data_batch, AEM_data_batch, gt_data_batch
        return video_data_equi_batch, video_data_cube_batch, audio_data_batch, gt_data_batch


if __name__ == "__main__":
   a = LoadVideoAudio('/ssd/VDA/test/clip_9_25', '/ssd/rtavah1/VIDEO_Saliency_database/annotation/maps/clip_9', 25)
   video_data, audio_data, gt_map = a.__getitem__(a.__len__()-1)
   print(a.__len__())
   print(video_data.shape)
   print(audio_data.shape)
   print(gt_map.shape)
