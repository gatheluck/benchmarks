import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import torch

from modules.camera import QuatPredictor


def test_quad_predictor():
	B = 16
	feat = torch.randn(B,100)
	quat_predictor = QuatPredictor(100)
	quat_predictor.init_to_zero_rotation()

	print(quat_predictor.fc.bias)
	assert torch.equal(quat_predictor.fc.bias, torch.FloatTensor([1,0,0,0]))

	quat = quat_predictor(feat)
	print(quat.shape)
	for i in range(B):
		assert abs(torch.norm(quat[i,:]).item()-1.0) < 0.000001