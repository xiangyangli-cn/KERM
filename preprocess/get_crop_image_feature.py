#!/usr/bin/env python3

''' Script to precompute image features using a Pytorch ResNet CNN, using 36 discretized views
    at each viewpoint in 30 degree increments, and the provided camera WIDTH, HEIGHT 
    and VFOV parameters. '''

import os
import sys
import itertools
import MatterSim
import argparse
import numpy as np
import json
import math
import h5py
import copy
from PIL import Image
import time
from progressbar import ProgressBar

import torch
import torch.nn.functional as F
import torch.multiprocessing
mp = torch.multiprocessing.get_context("spawn")
from torchvision.transforms import functional as F

from utils import load_viewpoint_ids
from tqdm import tqdm
from torch import optim

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomHorizontalFlip, RandomResizedCrop

from easydict import EasyDict as edict
import clip

os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

def BGR_to_RGB(cvimg):
    pilimg = cvimg.copy()
    pilimg[:, :, 0] = cvimg[:, :, 2]
    pilimg[:, :, 2] = cvimg[:, :, 0]
    return pilimg



TSV_FIELDNAMES = ['scanId', 'viewpointId', 'image_w', 'image_h', 'vfov', 'features']
VIEWPOINT_SIZE = 36 # Number of discretized views from one viewpoint
FEATURE_SIZE = 768

WIDTH = 640
HEIGHT = 480
VFOV = 60


def build_feature_extractor(checkpoint_file=None):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.set_grad_enabled(False)
    model, img_transforms = clip.load(checkpoint_file, device=device)

   
    model.eval()

    return model, img_transforms, device

def build_simulator(connectivity_dir, scan_dir):
    sim = MatterSim.Simulator()
    sim.setNavGraphPath(connectivity_dir)
    sim.setDatasetPath(scan_dir)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.setDepthEnabled(False)
    sim.setPreloadingEnabled(False)
    sim.setBatchSize(1)
    sim.initialize()
    return sim

def five_crop(image, ratio=0.6):
        w, h = image.size
        hw = (h*ratio, w*ratio)
        return F.five_crop(image, hw)

def nine_crop(image, ratio=0.4):
        w, h = image.size

        t = (0, int((0.5-ratio/2)*h), int((1.0 - ratio)*h))
        b = (int(ratio*h), int((0.5+ratio/2)*h), h)
        l = (0, int((0.5-ratio/2)*w), int((1.0 - ratio)*w))
        r = (int(ratio*w), int((0.5+ratio/2)*w), w)
        h, w = list(zip(t, b)), list(zip(l, r))

        images = []
        for s in itertools.product(h, w):
            h, w = s
            top, left = h[0], w[0]
            height, width = h[1]-h[0], w[1]-w[0]
            images.append(F.crop(image, top, left, height, width))
        
        return images

def process_features(proc_id, out_queue, scanvp_list, args):
    print('start proc_id: %d' % proc_id)
    gpu_count = torch.cuda.device_count()
    local_rank = proc_id % gpu_count

    torch.cuda.set_device('cuda:{}'.format(local_rank))
    # Set up the simulator

    sim = build_simulator(args.connectivity_dir, args.scan_dir)

    # Set up PyTorch CNN model
    torch.set_grad_enabled(False)
    model, img_transforms, device = build_feature_extractor(args.checkpoint_file)
    
    for scan_id, viewpoint_id in scanvp_list:
 
        # Loop all discretized views from this location

        images = []
        for ix in range(VIEWPOINT_SIZE):
            if ix == 0:
                sim.newEpisode([scan_id], [viewpoint_id], [0], [math.radians(-30)])
            elif ix % 12 == 0:
                sim.makeAction([0], [1.0], [1.0])
            else:
                sim.makeAction([0], [1.0], [0])
            state = sim.getState()[0]
            assert state.viewIndex == ix

            image = np.array(state.rgb, copy=True) # in BGR channel
            image = BGR_to_RGB(image)
            image = Image.fromarray(image) #cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            image = list(five_crop(image))   #five image
            images.extend(image)

        images = torch.stack([img_transforms(image).to(device) for image in images], 0)

        fts = []
        with torch.no_grad():
            for k in range(0, len(images), args.batch_size):
                b_fts = model.encode_image(images[k: k+args.batch_size])
                b_fts = b_fts.data.cpu().numpy()
                fts.append(b_fts)

        fts = np.concatenate(fts, 0)
        out_queue.put((scan_id, viewpoint_id, fts))

    out_queue.put(None)


def build_feature_file(args):
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    scanvp_list = load_viewpoint_ids(args.connectivity_dir)

    num_workers = min(args.num_workers, len(scanvp_list))
    num_data_per_worker = len(scanvp_list) // num_workers

    out_queue = mp.Queue()
    processes = []
    for proc_id in range(num_workers):
        sidx = proc_id * num_data_per_worker
        eidx = None if proc_id == num_workers - 1 else sidx + num_data_per_worker

        process = mp.Process(
            target=process_features,
            args=(proc_id, out_queue, scanvp_list[sidx: eidx], args)
        )
        process.start()
        processes.append(process)
    
    num_finished_workers = 0
    num_finished_vps = 0

    progress_bar = ProgressBar(max_value=len(scanvp_list))
    progress_bar.start()

    with h5py.File(args.output_file, 'w') as outf:
        while num_finished_workers < num_workers:
            res = out_queue.get()
            if res is None:
                num_finished_workers += 1
            else:
                scan_id, viewpoint_id, fts = res
                key = '%s_%s'%(scan_id, viewpoint_id)
                
                data = fts
                outf.create_dataset(key, data.shape, dtype='float', compression='gzip')
                outf[key][...] = data
                outf[key].attrs['scanId'] = scan_id
                outf[key].attrs['viewpointId'] = viewpoint_id
                outf[key].attrs['image_w'] = WIDTH
                outf[key].attrs['image_h'] = HEIGHT
                outf[key].attrs['vfov'] = VFOV

                num_finished_vps += 1
                progress_bar.update(num_finished_vps)

    progress_bar.finish()
    for process in processes:
        process.join()
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_file', default='../models/ViT-B-16.pt') #CLIP (ViT-B/16) models
    parser.add_argument('--connectivity_dir', default='../../connectivity')
    parser.add_argument('--scan_dir', default='../../data/v1/scans')
    parser.add_argument('--output_file',default='../datasets/kerm_data/clip_crop_image.hdf5') # Output of cropped image features
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', type=int, default=1)
    args = parser.parse_args()

    build_feature_file(args)


