import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.resnet3D import resnet18
from utils.resnet3D_cubic import resnet18 as resnet18_cubic
#from utils.equi_to_cube import Equi2Cube
from utils.cube_to_equi import Cube2Equi
import pdb


class ScaleUp(nn.Module):

    def __init__(self, in_size, out_size):
        super(ScaleUp, self).__init__()

        self.combine = nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_size)

        #self._weights_init()

    def _weights_init(self):

        nn.init.kaiming_normal_(self.combine.weight)
        nn.init.constant_(self.combine.bias, 0.0)

    def forward(self, inputs):
        output = F.interpolate(inputs, scale_factor=2, mode='bilinear', align_corners=True)
        output = self.combine(output)
        output = F.relu(output, inplace=True)
        return output


class DAVE(nn.Module):

    def __init__(self):
        super(DAVE, self).__init__()

        self.audio_branch = resnet18(shortcut_type='A', sample_size=64, sample_duration=16, num_classes=12, last_fc=False, last_pool=True)
        self.video_branch = resnet18(shortcut_type='A', sample_size=112, sample_duration=16, last_fc=False, last_pool=False)
        self.video_branch_cubic = resnet18_cubic(shortcut_type='A', sample_size=112, sample_duration=16, last_fc=False, last_pool=False)

        self.upscale1 = ScaleUp(512, 512)
        self.upscale2 = ScaleUp(512, 128)
        self.combinedEmbedding = nn.Conv2d(1024, 512, kernel_size=1, padding=0)
        self.combinedEmbedding_equi_cp = nn.Conv2d(1024, 512, kernel_size=1, padding=0)
        #self._weights_init()
        self.saliency = nn.Conv2d(128, 1, kernel_size=1, padding=0)
        
        self.c2e = Cube2Equi(4)   # input h of equi_img
        self.w = 0.5
        
    def _weights_init(self):   
        nn.init.kaiming_normal_(self.saliency.weight)
        nn.init.constant_(self.saliency.bias, 0.0)
        nn.init.kaiming_normal_(self.combinedEmbedding.weight)
        nn.init.constant_(self.combinedEmbedding.bias, 0.0)
        
    def forward(self, v_equi, v_cube, a, aem, eq_b):   # v_equi = [10, 3, 16, 256, 512], v_cube = [10, 6, 3, 16, 128, 128], aem = [10, 1, 8, 16]
        # V video frames of 3x16x256x320          # v.shape = [1, 3, 16, 256, 320]
        # A audio frames of 3x16x64x64            # a.shape = [10, 3, 16, 64, 64]
        # return a map of 32x40
            
        xV1_equi = self.video_branch(v_equi)                                                  # xV1_equi = [10, 512, 1, 8, 16]
        
        # Cube+Equi
        xV1_cube = self.video_branch_cubic(v_cube.view(6*v_cube.size(0), 3, 16, 128, 128))     # xV1_cube = [6, 512, 1, 4, 4]
        xV1_cube = xV1_cube.view(v_cube.size(0), 6, 512, 1, 4, 4)
        xV1_cube = torch.squeeze(xV1_cube, 3)                                                  # xV1_cube = [10, 6, 512, 4, 4]
        xV1_cube_equi = self.c2e.to_equi_nn(xV1_cube)
        xV1_cube_equi = F.interpolate(xV1_cube_equi, (8,10), mode='bilinear', align_corners=True)
        xV1_cube_equi = torch.unsqueeze(xV1_cube_equi, 2)                                      # xV1_cube_equi = [10, 512, 1, 8, 16]
        
        eq_b = eq_b.expand_as(xV1_cube_equi)
        aem = aem.expand_as(xV1_cube_equi)
        
        xV1_equi = xV1_equi*self.w + xV1_cube_equi*(1-self.w)          # xV1_equi = [10, 512, 1, 8, 16]
        
        xA1 = self.audio_branch(a)                                     # xA1 = [10, 512, 1, 1, 1]
        xA1 = xA1.expand_as(xV1_equi)                                  # xA1 = [10, 512, 1, 8, 10]
        
        #combine audio and video
        xC = torch.cat((xV1_equi, xA1), dim=1)                         # xC = [10, 1024, 1, 8, 16]
        xC = torch.squeeze(xC, dim=2)                                  # xC = [10, 1024, 8, 16]
        x = self.combinedEmbedding(xC)                                 # x = [10, 512, 8, 16]
        
        eq_b = torch.squeeze(eq_b, dim=2)
        eq_b = eq_b*(1.0 + aem)                                         # aem_type11
        x = eq_b*(x + x.max())                                         # type8, type12
        
        x = F.relu(x, inplace=True)                                    # x = [10, 512, 8, 16]

        x = torch.squeeze(x, dim=2)                                    # x = [10, 512, 8, 16]
        x = self.upscale1(x)                                           # x = [10, 512, 16, 32]
        x = self.upscale2(x)                                           # x = [10, 512, 32, 64]
        sal = self.saliency(x)                                         # x = [10, 1, 32, 64]
        sal = F.relu(sal, inplace=True)                                # x = [10, 1, 32, 64]
        
        sal = sal/sal.view(sal.size(0),-1).sum(1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        
        return sal

