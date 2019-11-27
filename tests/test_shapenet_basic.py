import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import torch

from kaolin.datasets.shapenet import ShapeNet_Images

def test_shapenet_images():
	dataset_root = '/home/gatheluck/Scratch/benchmarks/data/ShapeNetRendering'
	categories = ['02691156']
	dataset = ShapeNet_Images(dataset_root, categories, transform=None, split=1.0)
	
	loader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=8)

	for i, batch in enumerate(loader):
		data = batch['data']
		attributes = batch['attributes']
		print(data['images'].shape)
		print(data['params'])
		print(attributes)
		
	raise NotImplementedError