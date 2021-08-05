import numpy as np
import glob
import os
import pdb
import cv2
from skimage.transform import resize
from numpy import random
import matplotlib.pyplot as plt
import pandas as pd
#import imageio
import math

EPSILON = np.finfo('float').eps

def normalize(x, method='standard', axis=None):
	x = np.array(x, copy=False)
	if axis is not None:
		y = np.rollaxis(x, axis).reshape([x.shape[axis], -1])
		shape = np.ones(len(x.shape))
		shape[axis] = x.shape[axis]
		if method == 'standard':
			res = (x - np.mean(y, axis=1).reshape(shape)) / np.std(y, axis=1).reshape(shape)
		elif method == 'range':
			res = (x - np.min(y, axis=1).reshape(shape)) / (np.max(y, axis=1) - np.min(y, axis=1)).reshape(shape)
		elif method == 'sum':
			res = x / np.float_(np.sum(y, axis=1).reshape(shape))
		else:
			raise ValueError('method not in {"standard", "range", "sum"}')
	else:
		if method == 'standard':
			res = (x - np.mean(x)) / np.std(x)
		elif method == 'range':
			res = (x - np.min(x)) / (np.max(x) - np.min(x))
		elif method == 'sum':
			res = x / float(np.sum(x))
		else:
			raise ValueError('method not in {"standard", "range", "sum"}')
	return res

def KLD(p, q):
	p = normalize(p, method='sum')
	q = normalize(q, method='sum')
	return np.sum(np.where(p != 0, p * np.log((p+EPSILON) / (q+EPSILON)), 0))

def CC(saliency_map1, saliency_map2):
	map1 = np.array(saliency_map1, copy=False)
	map2 = np.array(saliency_map2, copy=False)
	if map1.shape != map2.shape:
		map1 = resize(map1, map2.shape, order=3, mode='constant') # bi-cubic/nearest is what Matlab imresize() does by default
	# Normalize the two maps to have zero mean and unit std
	map1 = normalize(map1, method='standard')
	map2 = normalize(map2, method='standard')
	# Compute correlation coefficient
	return np.corrcoef(map1.ravel(), map2.ravel())[0,1]

def NSS(saliency_map, fixation_map):
	s_map = np.array(saliency_map, copy=False)
	f_map = np.array(fixation_map, copy=False) > 0.5

	if not np.any(f_map):
		print('no fixation to predict')
		return np.nan
      
	if s_map.shape != f_map.shape:
		s_map = resize(s_map, f_map.shape)


	# Normalize saliency map to have zero mean and unit std
	s_map = normalize(s_map, method='standard')
	# Mean saliency value at fixation locations
	return np.mean(s_map[f_map])


def genERP(i,j,N):
    val = math.pi/N
    # w_map[i+j*w] = cos ((j - (h/2) + 0.5) * PI/h)
    w = math.cos( (j - (N/2) + 0.5) * val )
    return w

def compute_map_ws(h, w):
    """calculate weights for the sphere, the function provide weighting map for a given video
        :img    the input original video
    """
    equ = np.zeros((h, w))

    for j in range(0,equ.shape[0]):
        for i in range(0,equ.shape[1]):
            equ[j,i] = genERP(i,j,equ.shape[0])
    equ = equ/equ.max()
    return equ

def AUC_Judd(saliency_map, fixation_map, jitter=False):
	saliency_map = np.array(saliency_map, copy=False)
	fixation_map = np.array(fixation_map, copy=False) > 0.5
	# If there are no fixation to predict, return NaN
	if not np.any(fixation_map):
		print('no fixation to predict')
		return np.nan
	# Make the saliency_map the size of the fixation_map
	if saliency_map.shape != fixation_map.shape:
		saliency_map = resize(saliency_map, fixation_map.shape, order=3, mode='constant')

	# Jitter the saliency map slightly to disrupt ties of the same saliency value
	if jitter:
		saliency_map += random.rand(*saliency_map.shape) * 1e-7
	# Normalize saliency map to have values between [0,1]
	saliency_map = normalize(saliency_map, method='range')
	#saliency_map = normalize(saliency_map, method='standard')
	#saliency_map = normalize(saliency_map, method='sum')

	S = saliency_map.ravel()
	F = fixation_map.ravel()
	S_fix = S[F] # Saliency map values at fixation locations
	n_fix = len(S_fix)
	n_pixels = len(S)
	# Calculate AUC
	thresholds = sorted(S_fix, reverse=True)
	tp = np.zeros(len(thresholds)+2)
	fp = np.zeros(len(thresholds)+2)
	tp[0] = 0; tp[-1] = 1
	fp[0] = 0; fp[-1] = 1
	for k, thresh in enumerate(thresholds):
		above_th = np.sum(S >= thresh) # Total number of saliency map values above threshold
		tp[k+1] = (k + 1) / float(n_fix) # Ratio saliency map values at fixation locations above threshold
		fp[k+1] = (above_th - k - 1) / float(n_pixels - n_fix) # Ratio other saliency map values above threshold
	return np.trapz(tp, fp) # y, x

def AUC_Borji(saliency_map, fixation_map, n_rep=100, step_size=0.1, rand_sampler=None):
	saliency_map = np.array(saliency_map, copy=False)
	fixation_map = np.array(fixation_map, copy=False) > 0.5
	# If there are no fixation to predict, return NaN
	if not np.any(fixation_map):
		print('no fixation to predict')
		return np.nan
	# Make the saliency_map the size of the fixation_map
	if saliency_map.shape != fixation_map.shape:
		saliency_map = resize(saliency_map, fixation_map.shape, order=3, mode='constant')
	# Normalize saliency map to have values between [0,1]
	saliency_map = normalize(saliency_map, method='range')

	S = saliency_map.ravel()
	F = fixation_map.ravel()
	S_fix = S[F] # Saliency map values at fixation locations
	n_fix = len(S_fix)
	n_pixels = len(S)
	# For each fixation, sample n_rep values from anywhere on the saliency map
	if rand_sampler is None:
		r = random.randint(0, n_pixels, [n_fix, n_rep])
		S_rand = S[r] # Saliency map values at random locations (including fixated locations!? underestimated)
	else:
		S_rand = rand_sampler(S, F, n_rep, n_fix)
	# Calculate AUC per random split (set of random locations)
	auc = np.zeros(n_rep) * np.nan
	for rep in range(n_rep):
		thresholds = np.r_[0:np.max(np.r_[S_fix, S_rand[:,rep]]):step_size][::-1]
		tp = np.zeros(len(thresholds)+2)
		fp = np.zeros(len(thresholds)+2)
		tp[0] = 0; tp[-1] = 1
		fp[0] = 0; fp[-1] = 1
		for k, thresh in enumerate(thresholds):
			tp[k+1] = np.sum(S_fix >= thresh) / float(n_fix)
			fp[k+1] = np.sum(S_rand[:,rep] >= thresh) / float(n_fix)
		auc[rep] = np.trapz(tp, fp)
	return np.mean(auc) # Average across random splits


if __name__ == "__main__":
    
    folder = 'D:/datasets/360AV-HM/fixation_saliency/' # ground truth  
    gt_saliency_folder = folder + 'salmap/'
    gt_fixation_folder = folder + 'fixmap/'
    
    pd_saliency_folder = 'C:/Projects/AVS360-master/evaluation/ep25/' #predicted saliency map from AVS360
    csv_path = pd_saliency_folder + 'AVS360_score.csv'
    
    stretch_map = compute_map_ws(480, 640)
    list_vid = ['idLVnagjl_s', 'ey9J7w98wlI', 'kZB3KMhqqyI', 'MzcdEI-tSUc', '8ESEI0bqrJ4','1An41lDIJ6Q','6QUCaLvQ_3I', '8feS1rNYEbg','ByBF08H-wDA','fryDy9YcbI4', 'RbgxpagCY_c','dd39herpgXA']   
    
    list_id = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    fps_list=[30, 50, 30, 25, 25, 25, 60, 30, 30, 30, 30, 25]
    
    sound_type = ['none', 'mono', 'ambix']
    AUCj_score = np.zeros((3, len(list_vid)))
    AUCb_score = np.zeros((3, len(list_vid)))
    NSS_score = np.zeros((3, len(list_vid)))
    CC_score = np.zeros((3, len(list_vid)))
    KLD_score = np.zeros((3, len(list_vid)))

    score_total = np.zeros((5, 3, len(list_vid), 500))

    start = True
    for v, vid_name in enumerate(list_vid[:]):
        #print v, vid_name
        print(v, vid_name)
        for s, st in enumerate(sound_type[:]):
            gt_fixmap_list = [k for k in glob.glob(os.path.join(gt_fixation_folder, st, vid_name, '*.jpg'))]
            gt_fixmap_list.sort()
            
            gt_fixmap_list =  gt_fixmap_list[::5]
            gt_fixmap_len = len(gt_fixmap_list)

            gt_salmap_list = [k for k in glob.glob(os.path.join(gt_saliency_folder, st, vid_name, '*.jpg'))]
            gt_salmap_list.sort()
            gt_salmap_list =  gt_salmap_list[::5]
            gt_salmap_len = len(gt_salmap_list)

            AUCj_score_v = np.zeros(gt_fixmap_len)
            AUCb_score_v = np.zeros(gt_fixmap_len)
            NSS_score_v = np.zeros(gt_fixmap_len)
            CC_score_v = np.zeros(gt_fixmap_len)
            KLD_score_v = np.zeros(gt_fixmap_len)
            
            pd_salmap_path = [k for k in glob.glob(os.path.join(pd_saliency_folder, 'mono_-'+vid_name, '*.jpg'))]
            pd_salmap_path.sort()
            pd_salmap_path = pd_salmap_path[::5]

            for lnum, fpath in enumerate(gt_fixmap_list):
                
                gt_fixmap = cv2.imread(fpath, 0)
                fid = fpath.split('.jpg')[0].split('/')[-1]
            
                gt_salmap = cv2.resize(cv2.imread(gt_salmap_list[lnum], 0), (640,480))
                gt_salmap = gt_salmap*stretch_map
                pd_salmap_path =  [k for k in glob.glob(os.path.join(pd_saliency_folder, 'mono_-'+vid_name, str(int(fid)+1)+'.jpg'))][0]
                
                pd_salmap = cv2.imread(pd_salmap_path, cv2.IMREAD_GRAYSCALE)

                #pd_salmap = np.load(pd_salmap_path[lnum])
                pd_salmap = cv2.resize(pd_salmap, (640,480))
                pd_salmap = pd_salmap * stretch_map

                #pd_fixmap = cv2.imread(pd_fixmap_path[lnum], cv2.IMREAD_GRAYSCALE)
              
                if pd_salmap.sum() == 0:
                    print('pd_salmap.sum()=', pd_salmap.sum())
                    AUCj_score_v[lnum] = np.nan
                    AUCb_score_v[lnum] = np.nan
                    NSS_score_v[lnum] = np.nan
                    CC_score_v[lnum] = np.nan 
                    KLD_score_v[lnum] = np.nan 
                else: 
                    AUCj_score_v[lnum] = AUC_Judd(pd_salmap, gt_fixmap)
                    AUCb_score_v[lnum] = AUC_Borji(pd_salmap, gt_fixmap)
                    NSS_score_v[lnum] = NSS(pd_salmap, gt_fixmap) 
                    CC_score_v[lnum] = CC(pd_salmap, gt_salmap)
                    KLD_score_v[lnum] =  KLD(pd_salmap, gt_salmap)
                  
                print(v, st, vid_name, lnum, gt_fixmap_len)
                
                #print(lnum, 'AUCj=', AUCj_score_v[lnum], 'AUCb=', AUCb_score_v[lnum], 'NSS=', NSS_score_v[lnum], 'CC=', CC_score_v[lnum], 'KLD=', KLD_score_v[lnum])
            #pdb.set_trace()
            AUCj_score[s, v] = np.nanmean(AUCj_score_v)
            AUCb_score[s, v] = np.nanmean(AUCb_score_v)
            NSS_score[s, v] = np.nanmean(NSS_score_v)
            CC_score[s, v] = np.nanmean(CC_score_v)
            KLD_score[s, v] = np.nanmean(KLD_score_v)
            print(st, vid_name, 'AUCj=', AUCj_score[s, v],'AUCb=', AUCb_score[s, v], 'NSS=',  NSS_score[s, v], 'CC=', CC_score[s,v], 'KLD=', KLD_score[s,v])
            
            #plt.plot(NSS_score_v)
            score_total[0, s, v, 0:len(AUCj_score_v)] = AUCj_score_v
            score_total[1, s, v, 0:len(NSS_score_v)] = NSS_score_v
            score_total[2, s, v, 0:len(CC_score_v)] = CC_score_v
            score_total[3, s, v, 0:len(KLD_score_v)] = KLD_score_v
            score_total[4, s, v, 0:len(KLD_score_v)] = AUCb_score_v

           
            pdData = {
                    "ID": str(list_id[v]),
                    "file": vid_name,
                    "sound": st,
                    "avgAUCj": np.nanmean(AUCj_score_v),
                    "avgAUCb": np.nanmean(AUCb_score_v),
                    "avgNSS": np.nanmean(NSS_score_v),
                    "avgCC" : np.nanmean(CC_score_v),
                    "avgKLD": np.nanmean(KLD_score_v),
                    "maxAUCj": np.nanmax(AUCj_score_v),
                    "minAUCj": np.nanmin(AUCj_score_v),
                    "maxAUCb": np.nanmax(AUCb_score_v),
                    "minAUCb": np.nanmin(AUCb_score_v),
                    "maxNSS": np.nanmax(NSS_score_v),
                    "minNSS": np.nanmin(NSS_score_v),
                    "maxCC": np.nanmax(CC_score_v),
                    "minCC": np.nanmin(CC_score_v),
                    "maxKLD": np.nanmax(KLD_score_v),
                    "minKLD": np.nanmin(KLD_score_v),
                    "stdAUCj": np.nanstd(AUCj_score_v),
                    "stdAUCb": np.nanstd(AUCb_score_v),
                    "stdNSS": np.nanstd(NSS_score_v),
                    "stdCC": np.nanstd(CC_score_v),
                    "stdKLD": np.nanstd(KLD_score_v),
                }
            pdData = pd.DataFrame([pdData], index=None)
            if start:
               pdData.to_csv(csv_path, header = True, mode='a')
               start = False
            else:
               pdData.to_csv(csv_path, header = False, mode='a')
            

 
    AUCj_score[AUCj_score==0.0] = np.nan
    AUCb_score[AUCb_score==0.0] = np.nan
    NSS_score[NSS_score==0.0] = np.nan
    CC_score[CC_score==0.0] = np.nan
    KLD_score[KLD_score==0.0] = np.nan
    start = True
    for s, st in enumerate(sound_type):
        pdData_t = {
                "sound": st,
                "allvid_avgAUCj": np.nanmean(AUCj_score[s, :]),
                "allvid_stdAUCj": np.nanstd(AUCj_score[s, :]),
                "allvid_avgAUCb": np.nanmean(AUCb_score[s, :]),
                "allvid_stdAUCb": np.nanstd(AUCb_score[s, :]),
                "allvid_avgNSS": np.nanmean(NSS_score[s, :]),  
                "allvid_stdNSS": np.nanstd(NSS_score[s, :]),
                "allvid_avgCC": np.nanmean(CC_score[s, :]),  
                "allvid_stdCC": np.nanstd(CC_score[s, :]),
                "allvid_avgKLD": np.nanmean(KLD_score[s, :]),  
                "allvid_stdKLD": np.nanstd(KLD_score[s, :]),
                  }
        pdData_t = pd.DataFrame([pdData_t],index=None)
        pdData_t.to_csv(csv_path, header = start, mode='a')
        start = False

    start = True
    for s, st in enumerate(sound_type):
        pdData_t = {
                "sound": st,
                "convid_avgAUCj": np.nanmean(AUCj_score[s, :4]),
                "convid_stdAUCj": np.nanstd(AUCj_score[s, :4]),
                "convid_avgAUCb": np.nanmean(AUCb_score[s, :4]),
                "convid_stdAUCb": np.nanstd(AUCb_score[s, :4]),
                "convid_avgNSS": np.nanmean(NSS_score[s, :4]),  
                "convid_stdNSS": np.nanstd(NSS_score[s, :4]),
                "convid_avgCC": np.nanmean(CC_score[s, :4]),  
                "convid_stdCC": np.nanstd(CC_score[s, :4]),
                "convid_avgKLD": np.nanmean(KLD_score[s, :4]),  
                "convid_stdKLD": np.nanstd(KLD_score[s, :4]),
                }
        pdData_t = pd.DataFrame([pdData_t],index=None)
        pdData_t.to_csv(csv_path, header = start, mode='a')
        start = False
    
    start = True
    for s, st in enumerate(sound_type):
        pdData_t = {
                "sound": st,
                "musvid_avgAUCj": np.nanmean(AUCj_score[s, 4:8]),
                "musvid_stdAUCj": np.nanstd(AUCj_score[s, 4:8]),
                "musvid_avgAUCb": np.nanmean(AUCb_score[s, 4:8]),
                "musvid_stdAUCb": np.nanstd(AUCb_score[s, 4:8]),
                "musvid_avgNSS": np.nanmean(NSS_score[s, 4:8]),  
                "musvid_stdNSS": np.nanstd(NSS_score[s, 4:8]),
                "musvid_avgCC": np.nanmean(CC_score[s, 4:8]),  
                "musvid_stdCC": np.nanstd(CC_score[s, 4:8]),
                "musvid_avgKLD": np.nanmean(KLD_score[s, 4:8]),  
                "musvid_stdKLD": np.nanstd(KLD_score[s, 4:8]),
                }
        pdData_t = pd.DataFrame([pdData_t],index=None)
        pdData_t.to_csv(csv_path, header = start, mode='a')
        start = False

    start = True
    for s, st in enumerate(sound_type):
        pdData_t = {
                "sound": st,
                "envvid_avgAUCj": np.nanmean(AUCj_score[s, 8:]),
                "envvid_stdAUCj": np.nanstd(AUCj_score[s, 8:]),
                "envvid_avgAUCb": np.nanmean(AUCb_score[s, 8:]),
                "envvid_stdAUCb": np.nanstd(AUCb_score[s, 8:]),
                "envvid_avgNSS": np.nanmean(NSS_score[s, 8:]),  
                "envvid_stdNSS": np.nanstd(NSS_score[s, 8:]),
                "envvid_avgCC": np.nanmean(CC_score[s, 8:]),  
                "envvid_stdCC": np.nanstd(CC_score[s, 8:]),
                "envvid_avgKLD": np.nanmean(KLD_score[s, 8:]),  
                "envvid_stdKLD": np.nanstd(KLD_score[s, 8:]),
                }
        pdData_t = pd.DataFrame([pdData_t],index=None)
        pdData_t.to_csv(csv_path, header = start, mode='a')
        start = False
    
    
