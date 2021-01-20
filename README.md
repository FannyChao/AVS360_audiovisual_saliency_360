# Towards Audio-Visual Saliency Prediction for Omnidirectional Video with Spatial Audio
- This repo contains the codes that used in paper [*Towards Audio-Visual Saliency Prediction for Omnidirectional Video with Spatial Audio (VCIP2020)*](https://hal.archives-ouvertes.fr/hal-03104425) by Fang-Yi Chao, Cagri Ozcinar, Lu Zhang, Wassim Hamidouche, Olivier Deforges, Aljosa Smolic.

## Abstract
Omnidirectional videos (ODVs) with spatial audio enable viewers to perceive 360° directions of audio and visual signals during the consumption of ODVs with head-mounted displays (HMDs). By predicting salient audio-visual regions, ODV systems can be optimized to provide an immersive sensation of audio-visual stimuli with high-quality. Despite the intense recent effort for ODV saliency prediction, the current literature still does not consider the impact of auditory information in ODVs. In this work, we propose an audio-visual saliency (AVS360) model that incorporates 360° spatial-temporal visual representation and spatial auditory information in ODVs. The proposed AVS360 model is composed of two 3D residual networks (ResNets) to encode visual and audio cues. The first one is embedded with a spherical representation technique to extract 360° visual features, and the second one extracts the features of audio using the log mel-spectrogram. We emphasize sound source locations by integrating audio energy map (AEM) generated from spatial audio description (i.e., ambisonics) and equator viewing behavior with equator center bias (ECB). The audio and visual features are combined and fused with AEM and ECB via attention mechanism. Our experimental results show that the AVS360 model has significant superiority over five state-of-the-art saliency models. To the best of our knowledge, it is the first work that develops the audio-visual saliency model in ODVs.

## Architecture
![diagram](https://github.com/FannyChao/AVS360_audiovisual_saliency_360/blob/master/figs/AVS360_diagram.jpg)

## Visual Results
- Qualitative results on [*360AV-HM dataset*](https://github.com/cozcinar/360_Audio_Visual_ICMEW2020)
![results](https://github.com/FannyChao/AVS360_audiovisual_saliency_360/blob/master/figs/results.jpg)

## Requirements
- python2

## Pretrained AVS360 model
- [AVS360 Model]()

## Installation

```bash
conda create -n salmap

conda activate salmap

pip install torch===1.5.0 torchvision===0.6.0 -f https://download.pytorch.org/whl/torch_stable.html

pip install librosa

pip install numba==0.48
```

## Usage
- Test: To predict saliency maps, run ```predict.py``` after specifying the paths of ```VIDEO_TEST_FOLDER```, ```OUTPUT```, and ```MODEL_PATH```.
- Train: Run ```train.py``` after specifying the paths of ```VIDEO_TRAIN_FOLDER```, ```OUTPUT```, and ```MODEL_PATH```.


## Citing
```
@INPROCEEDINGS{9301766,
  author={F. -Y. {Chao} and C. {Ozcinar} and L. {Zhang} and W. {Hamidouche} and O. {Deforges} and A. {Smolic}},
  booktitle={2020 IEEE International Conference on Visual Communications and Image Processing (VCIP)}, 
  title={Towards Audio-Visual Saliency Prediction for Omnidirectional Video with Spatial Audio}, 
  year={2020},
  volume={},
  number={},
  pages={355-358},
  doi={10.1109/VCIP49819.2020.9301766}}
```
