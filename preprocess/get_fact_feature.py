#!/usr/bin/env python3

''' Script to precompute image features using a Pytorch ResNet CNN, using 36 discretized views
    at each viewpoint in 30 degree increments, and the provided camera WIDTH, HEIGHT 
    and VFOV parameters. '''

import os
import sys


import argparse
import numpy as np
import json
import math
import h5py
import copy
from PIL import Image
import time

import torch

from tqdm import tqdm
from torch.utils.data import DataLoader
import clip

dataloader = json.load(open('../datasets/kerm_data/vg.json','r'))
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("../models/ViT-B-16.pt", device=device)
index = 0

with torch.no_grad():
    with h5py.File("../datasets/kerm_data/vg.hdf5", 'w') as outf:
        for batch in tqdm(dataloader):
            text = clip.tokenize(batch).to(device)
            fts = model.encode_text(text).detach().cpu().numpy()
            for j in range(fts.shape[0]):
                key = str(index)
                index += 1
                data = fts[j]
                outf.create_dataset(key, data.shape, dtype='float', compression='gzip')
                outf[key][...] = data



