# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

from kaolin.mathutils.geometry import transformations

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

class ShapeNet_Images(data.Dataset):
    r"""ShapeNet Dataset class for images.

    Arguments:
        root (str): Path to the root directory of the ShapeNet dataset.
        categories (str): List of categories to load from ShapeNet. This list may
                contain synset ids, class label names (for ShapeNetCore classes),
                or a combination of both.
        train (bool): if true use the training set, else use the test set
        split (float): amount of dataset that is training out of
        views (int): number of viewpoints per object to load
        transform (torchvision.transforms) : transformation to apply to images
        no_progress (bool): if True, disables progress bar

    Returns:
        .. code-block::

        dict: {
            attributes: {name: str, path: str, synset: str, label: str},
            data: {vertices: torch.Tensor, faces: torch.Tensor}
            params: {
                cam_mat: torch.Tensor,
                cam_pos: torch.Tensor,
                azi: float,
                elevation: float,
                distance: float
            }
        }

    Example:
        >>> from torch.utils.data import DataLoader
        >>> images = ShapeNet_Images(root='../data/ShapeNetImages')
        >>> train_loader = DataLoader(images, batch_size=10, shuffle=True, num_workers=8)
        >>> obj = next(iter(train_loader))
        >>> image = obj['data']['imgs']
        >>> image.shape
        torch.Size([10, 4, 137, 137])
    """

    def __init__(self, root: str, categories: list = ['chair'], train: bool = True,
                 split: float = .7, views: int = 24, transform=None,
                 no_progress: bool = False):
        self.root = root
        self.synsets = _convert_categories(categories)
        self.labels = [synset_to_label[s] for s in self.synsets]
        self.transform = transform
        self.views = views
        self.names = []
        self.synset_idx = []

        shapenet_img_root = os.path.join(self.root, 'images')
        # check if images exist
        if not os.path.exists(shapenet_img_root):
            raise ValueError('ShapeNet images were not found at location {0}.'.format(shapenet_img_root))

        # find all needed images
        for i in tqdm(range(len(self.synsets)), disable=no_progress):
            syn = self.synsets[i]
            class_target = os.path.join(shapenet_img_root, syn)
            assert os.path.exists(class_target), "ShapeNet class, {0}, is not found".format(syn)

            models = sorted(glob(os.path.join(class_target, '*')))
            stop = int(len(models) * split)
            if train:
                models = models[:stop]
            else:
                models = models[stop:]
            self.names += models

            self.synset_idx += [i] * len(models)

    def __len__(self):
        """Returns the length of the dataset. """
        return len(self.names)

    def __getitem__(self, index):
        """Returns the item at index idx. """
        data = dict()
        attributes = dict()
        name = self.names[index]
        view_num = random.randrange(0, self.views)
        # load and process image
        img = Image.open(os.path.join(name, 'rendering/{:02d}.png'.format(view_num)))
        # apply transformations
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.FloatTensor(np.array(img))
            img = img.permute(2, 1, 0)
            img = img / 255.
        # load and process camera parameters
        param_location = os.path.join(name, 'rendering/rendering_metadata.txt')
        azimuth, elevation, _, distance, _ = np.loadtxt(param_location)[view_num]
        cam_params = transformations.compute_camera_params(azimuth, elevation, distance)

        data['images'] = img
        data['params'] = dict()
        data['params']['cam_mat'] = cam_params[0]
        data['params']['cam_pos'] = cam_params[1]
        data['params']['azimuth'] = azimuth
        data['params']['elevation'] = elevation
        data['params']['distance'] = distance
        attributes['name'] = name
        attributes['synset'] = self.synsets[self.synset_idx[index]]
        attributes['label'] = self.labels[self.synset_idx[index]]
        return {'data': data, 'attributes': attributes}
