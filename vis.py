import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import cv2
from tqdm import tqdm

def merge(x):
    B, C, H, W = x.shape
    N = int(np.sqrt(C))+1
    res = np.zeros((H*N, W*N))
    for i in range(C):
        res[(i//N)*H:(i//N+1)*H, (i%N)*W:(i%N+1)*W] = x[0,i]
    return res

def normalize(x):
    return x
    #return (x-x.min())/(x.max()-x.min())

def show(f):
    a = torch.load(f)
    plt.imsave(f'{f}_x.png', normalize(a['x'][0,0]))
    plt.imsave(f'{f}_y.png', a['y'][0].astype(np.uint8))
    plt.imsave(f'{f}_f.png', normalize(merge(a['f'])))

path = 'vis_CT'
for f in tqdm(os.listdir(path)):
    show(os.path.join(path, f))


