# KERM: Knowledge Enhanced Reasoning for Vision-and-Language Navigation

This repository is the official implementation of [KERM: Knowledge Enhanced Reasoning for Vision-and-Language Navigation]

Vision-and-language navigation (VLN) is the task to enable an embodied agent to navigate to a remote location following the natural language instruction in real scenes. Most of the previous approaches utilize the entire features or object-centric features to represent navigable candidates. However, these representations are not efficient enough for an agent to navigate to the target location. As knowledge provides crucial information which is complementary to visible content, in this paper, we propose a knowledge enhanced reasoning model (KERM) to leverage knowledge to improve agent navigation ability. Specifically, we first retrieve facts for the navigation views from the constructed knowledge base. And than we build a knowledge enhanced reasoning network, containing purification, fact-aware interaction, and instruction-guided aggregation modules, to integrate the visual features, history features, instruction features, and fact features for action prediction. Extensive experiments are conducted on the REVERIE, R2R, and SOON datasets. Experimental results demonstrate the effectiveness of the proposed method.


## Requirements

1. Install Matterport3D simulators: follow instructions [here](https://github.com/peteanderson80/Matterport3DSimulator). We use the latest version.
```
export PYTHONPATH=Matterport3DSimulator/build:$PYTHONPATH
```

2. Install requirements:
```setup
conda create --name KERM python=3.8.0
conda activate KERM
pip install -r requirements.txt
```

3. Download dataset from [Dropbox](https://www.dropbox.com/sh/u3lhng7t2gq36td/AABAIdFnJxhhCg2ItpAhMtUBa?dl=0), including processed annotations, features and pretrained models from [VLN-DUET](https://github.com/cshizhe/VLN-DUET). Put the data in `datasets' directory.

4. Download pretrained lxmert
```
mkdir -p datasets/pretrained 
wget https://nlp.cs.unc.edu/data/model_LXRT.pth -P datasets/pretrained
```

5. Download preprocessed data and features of KERM from [Baidu Netdisk](https://pan.baidu.com/s/1V-dmZaesy18_eARBRMUOqQ?pwd=ah8t), including features of knowledge base (vg.json), annotations of retrieved facts (knowledge.json), cropped image features (clip_crop_image.hdf5). Put the 'kerm_data' in `datasets' directory.


6. Download trained KERM models from [Baidu Netdisk](https://pan.baidu.com/s/1V-dmZaesy18_eARBRMUOqQ?pwd=ah8t).
