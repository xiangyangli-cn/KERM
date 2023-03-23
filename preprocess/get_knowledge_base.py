from PIL import Image
from pathlib import Path
import numpy as np
import json
import itertools
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from transformers import CLIPProcessor
from torchvision.transforms import functional as F

LENGTH_LIMIT = 75

class VisualGenomeCaptions():
    def __init__(self, ann_dir):
        super().__init__()
        escapes = ''.join([chr(char) for char in range(0, 32)])
        self.translator = str.maketrans('', '', escapes)

        self.caps = self.parse_annotations(Path(ann_dir))

    @staticmethod
    def combination(l1, l2):
        return [" ".join(x) for x in itertools.product(l1, l2)]
    
    def process_word(self, s):
        return s.lower().strip().translate(self.translator)
    
    def process_synset(self, s):
        return s.lower().strip().translate(self.translator).split(".")[0]
    
    def parse_annotations(self, ann_dir):
        print("loading object attributes...")
        objs = {}
        with open(ann_dir/"attributes.json", "r") as f:
            attributes = json.load(f)
        for x in tqdm(attributes, dynamic_ncols=True):
            for a in x["attributes"]:
                _names = set(self.process_synset(y) for y in a.get("synsets", list()))
                _attrs = set(self.process_word(y) for y in a.get("attributes", list()))

                for n in _names:
                    try:
                        objs[n] |= _attrs
                    except KeyError:
                        objs[n] = _attrs
        del attributes

        print("loading object relationships...")
        rels = set()
        with open(ann_dir/"relationships.json", "r") as f:
            relationships = json.load(f)
        for x in tqdm(relationships, dynamic_ncols=True):
            for r in x["relationships"]:
                _pred = self.process_word(r["predicate"])
                _subj = set(self.process_synset(y) for y in r["subject"]["synsets"])
                _obj = set(self.process_synset(y) for y in r["object"]["synsets"])

                for s in _subj:
                    for o in _obj:
                        rels.add(f"{s}<sep>{_pred}<sep>{o}")
        del relationships

        print("parsing object attributes...")
        caps_obj = []
        for o in tqdm(objs.keys()):
            for a in objs[o]:
                if a != "":
                    caps_obj.append(f"{a} {o}")


        print("parsing object relationships...")
        caps_rel = []
        for r in tqdm(rels):
            s, p, o = r.split("<sep>")
            caps_rel.append(f"{s} {p} {o}")

        caps = np.unique(caps_obj + caps_rel).tolist()

        return caps

vg = VisualGenomeCaptions("../datasets/kerm_data/vg_annotations")
json.dump(vg.caps,open('../datasets/kerm_data/vg.json','w'))