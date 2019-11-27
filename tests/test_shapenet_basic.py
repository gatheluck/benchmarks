import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import torch

from kaolin.datasets.shapenet import ShapeNet_Images

def test_shapenet_images():
	dataset_root = '/home/gatheluck/Scratch/benchmarks/data/ShapeNetRendering'
	categories = ['02691156']
	dataset = ShapeNet_Images(dataset_root, categories)