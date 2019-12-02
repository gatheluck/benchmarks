import sys
import os
from pathlib import Path
import torch
import torch.utils.data as data
import warnings
import urllib.request
import zipfile
import json
import re
from collections import OrderedDict
from glob import glob
import numpy as np
import random

from tqdm import tqdm
import scipy.sparse
import tarfile
from PIL import Image

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(base)

import kaolin as kal
from kaolin.rep.TriangleMesh import TriangleMesh

# Synset to Label mapping (for ShapeNet core classes)
synset_to_label = {'04379243': 'table', '03211117': 'monitor', '04401088': 'phone',
                   '04530566': 'watercraft', '03001627': 'chair', '03636649': 'lamp',
                   '03691459': 'speaker', '02828884': 'bench', '02691156': 'plane',
                   '02808440': 'bathtub', '02871439': 'bookcase', '02773838': 'bag',
                   '02801938': 'basket', '02880940': 'bowl', '02924116': 'bus',
                   '02933112': 'cabinet', '02942699': 'camera', '02958343': 'car',
                   '03207941': 'dishwasher', '03337140': 'file', '03624134': 'knife',
                   '03642806': 'laptop', '03710193': 'mailbox', '03761084': 'microwave',
                   '03928116': 'piano', '03938244': 'pillow', '03948459': 'pistol',
                   '04004475': 'printer', '04099429': 'rocket', '04256520': 'sofa',
                   '04554684': 'washer', '04090263': 'rifle', '02946921': 'can'}

# Label to Synset mapping (for ShapeNet core classes)
label_to_synset = {v: k for k, v in synset_to_label.items()}

def _convert_categories(categories):
    assert categories is not None, 'List of categories cannot be empty!'
    if not (c in synset_to_label.keys() + label_to_synset.keys()
            for c in categories):
        warnings.warn('Some or all of the categories requested are not part of \
            ShapeNetCore. Data loading may fail if these categories are not avaliable.')
    synsets = [label_to_synset[c] if c in label_to_synset.keys()
               else c for c in categories]
    return synsets

class ShapeNetDataset(data.Dataset):
    def __init__(self, 
                 root: str,
                 categories: list = ['chair'],
                 train: bool = True, 
                 split: float = 0.7,
                 views: int = 24,
                 transform = None,
                 no_progress: bool = False):
        
        """
        dataset dirctory should be like below, 
            ShapeNet/models/02691156/fff513f407e00e85a9ced22d91ad7027/models/model_normalized.obj
                            ...
                    /images/02691156/fff513f407e00e85a9ced22d91ad7027/rendering/00.png
                                                                                00_mask.png
                            ...
        """
        self.root = root
        self.synsets = _convert_categories(categories)
        self.labels = [synset_to_label[s] for s in self.synsets]
        self.transform = transform
        self.views = views
        self.img_names = []
        self.obj_names = []
        self.synset_idx = []

        shapenet_img_root = os.path.join(self.root, 'images')
        shapenet_obj_root = os.path.join(self.root, 'models')
        # check if images exist
        if not os.path.exists(shapenet_img_root):
            raise ValueError('ShapeNet images were not found at location {0}.'.format(shapenet_img_root))
        if not os.path.exists(shapenet_obj_root):
            raise ValueError('ShapeNet models were not found at location {0}.'.format(shapenet_obj_root))
        
        # find all needed images and models
        for i in tqdm(range(len(self.synsets)), disable=no_progress):
            syn = self.synsets[i]
            img_class_target = os.path.join(shapenet_img_root, syn)
            obj_class_target = os.path.join(shapenet_obj_root, syn)

            assert os.path.exists(img_class_target), "Image of ShapeNet class, {0}, is not found".format(syn)
            assert os.path.exists(obj_class_target), "Model of ShapeNet class, {0}, is not found".format(syn)

            img_instances = sorted(glob(os.path.join(img_class_target, '*')))
            obj_instances = sorted(glob(os.path.join(obj_class_target, '*')))

            assert len(img_instances)==len(obj_instances), "Length of both instances sould be same."

            # split train and test set
            stop = int(len(img_instances) * split)
            if train:
                img_instances = img_instances[:stop]
                obj_instances = obj_instances[:stop]
            else:
                img_instances = img_instances[stop:]
                obj_instances = obj_instances[stop:]
            
            self.img_names += img_instances
            self.obj_names += obj_instances

            self.synset_idx += [i] * len(img_instances)

    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, index):
        data = dict()
        attributes = dict()

        img_name = self.img_names[index]
        obj_name = self.obj_names[index]

        view_num = random.randrange(0, self.views)
        # load image and mesh
        img  = Image.open(os.path.join(img_name, 'rendering/{:02d}.png'.format(view_num)))
        mesh = TriangleMesh.from_obj(os.path.join(obj_name, 'models/model_normalized.obj'))
        
        # apply transformations to img
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.FloatTensor(np.array(img))
            img = img.permute(2, 0, 1)
            img = img / 255.
        # load and process cam
        param_location = os.path.join(img_name, 'rendering/rendering_metadata.txt')
        azimuth, elevation, _, distance, _ = np.loadtxt(param_location)[view_num]
        cam_params = kal.mathutils.geometry.transformations.compute_camera_params(azimuth, elevation, distance)

        data['images'] = img
        #data['vertices'] = mesh.vertices
        #data['faces'] = mesh.faces
        data['params'] = dict()
        data['params']['cam_mat'] = cam_params[0]
        data['params']['cam_pos'] = cam_params[1]
        data['params']['azi'] = azimuth
        data['params']['elevation'] = elevation
        data['params']['distance'] = distance
        attributes['img_name'] = img_name
        attributes['obj_name'] = obj_name
        attributes['synset'] = self.synsets[self.synset_idx[index]]
        attributes['label'] = self.labels[self.synset_idx[index]]
        return {'data': data, 'attributes': attributes}