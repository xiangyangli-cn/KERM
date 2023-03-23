import torch
import json
import h5py
import numpy as np
import os
from tqdm import tqdm

def load_viewpoint_ids(connectivity_dir):
    viewpoint_ids = []
    with open(os.path.join(connectivity_dir, 'scans.txt')) as f:
        scans = [x.strip() for x in f]
    for scan in scans:
        with open(os.path.join(connectivity_dir, '%s_connectivity.json'%scan)) as f:
            data = json.load(f)
            viewpoint_ids.extend([(scan, x['image_id']) for x in data if x['included']])
    print('Loaded %d viewpoints' % len(viewpoint_ids))
    return viewpoint_ids


navigation_text = json.load(open("../datasets/kerm_data/vg.json","r"))

class FeaturesDB(object):
    def __init__(self, img_ft_file):
        self.img_ft_file = img_ft_file
        self._feature_store = h5py.File(self.img_ft_file, 'r')

    def get_feature(self, key):

        ft = self._feature_store[key][...][:].astype(np.float32)

        return ft

imageDB = FeaturesDB("../datasets/kerm_data/clip_crop_image.hdf5")

textDB = FeaturesDB("../datasets/kerm_data/vg.hdf5")

scanvp_list = load_viewpoint_ids("../../connectivity")

text_len = len(navigation_text)
text_dict = []
for i in tqdm(range(text_len)):
    feature = textDB.get_feature(str(i))
    feature = np.expand_dims(feature,axis=0)
    text_dict.append(feature)

text_dict = np.concatenate(text_dict,axis=0)

text_dict = torch.tensor(text_dict).to("cuda:0")
text_dict = (text_dict / text_dict.norm(dim=-1, keepdim=True)).t()

retrieval_result = {}

CROP_SIZE = 5

for scan_id, viewpoint_id in tqdm(scanvp_list):
    # Loop all discretized views from this location

    key = '%s_%s'%(scan_id, viewpoint_id)
    feature = imageDB.get_feature(key)
    feature = torch.tensor(feature).to("cuda:0")

    for ix in range(36*CROP_SIZE):
        img_tensor = feature[ix:ix+1] / feature[ix:ix+1].norm(dim=-1, keepdim=True)

        logits = img_tensor @ text_dict

        logits = logits.view(-1)
        value,index = torch.topk(logits,5)
        key = '%s_%s_%d_%d'%(scan_id, viewpoint_id,ix//CROP_SIZE,ix%CROP_SIZE)
        retrieval_result[key]=[]

        for j in index.cpu().numpy().tolist():
            retrieval_result[key].append(j)

        for j in value.cpu().numpy().tolist():
            retrieval_result[key].append(j)

json.dump(retrieval_result,open("../datasets/kerm_data/knowledge.json",'w'))


