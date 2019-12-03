import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import math
import torch

from modules.camera import PosePredictorQuat, PosePredictorAEC, CameraPredictor, MutiCameraPredictor


def test_pose_predictor_quat():
	B = 16
	feat = torch.randn(B,100)
	pose_predictor = PosePredictorQuat(100)
	pose_predictor.init_to_zero_rotation()

	print(pose_predictor.fc.bias)
	assert torch.equal(pose_predictor.fc.bias, torch.FloatTensor([1,0,0,0]))

	quat = pose_predictor(feat)
	print(quat.shape)
	for i in range(B):
		assert abs(torch.norm(quat[i,:]).item()-1.0) < 0.000001

def test_pose_predictor_aec():
	B = 16
	feat = torch.randn(B,100)
	pose_predictor = PosePredictorAEC(100)

	pose = pose_predictor(feat)
	print(pose.size())
	assert pose.size(0) == B
	assert pose.size(1) == 3
	assert len(pose.size()) == 2

def test_camera_predictor():
	B=16
	feat = torch.randn(B,100)
	# type == 'aec'
	camera_predictor = CameraPredictor(100, sc_bias=1.8, sc_scaling=0.05,
									   aec_scaling_ang=0.1,
									   aec_scaling_azm=math.pi/6.0,
									   aec_scaling_ele=math.pi/9.0,
									   aec_scaling_cyc=math.pi/9.0,
									   pose_type='aec')
	cam = camera_predictor(feat)
	assert cam.size(0) == B
	assert cam.size(1) == 7
	assert len(cam.size()) == 2

	# type == 'quat'
	camera_predictor = CameraPredictor(100, sc_bias=1.8, sc_scaling=0.05,
									   aec_scaling_ang=0.1,
									   aec_scaling_azm=math.pi/6.0,
									   aec_scaling_ele=math.pi/9.0,
									   aec_scaling_cyc=math.pi/9.0,
									   pose_type='quat')
	cam = camera_predictor(feat)
	assert cam.size(0) == B
	assert cam.size(1) == 8
	assert len(cam.size()) == 2

def test_multi_camera_predictor():
	B=16
	num_cams=10
	feat = torch.randn(B,100)
	# type == 'aec'
	print('test_multi_camera_predictor: pose_type==aec')
	multi_camera_predictor = MutiCameraPredictor(100, num_cams=num_cams,
									   			 sc_bias=1.8, sc_scaling=0.05,
									   			 aec_scaling_ang=0.1,
									   			 aec_scaling_azm=math.pi/6.0,
									   			 aec_scaling_ele=math.pi/9.0,
									   			 aec_scaling_cyc=math.pi/9.0,
									   			 pose_type='aec')

	cam, idx, prob = multi_camera_predictor(feat)
	assert cam.size(0) == B
	assert cam.size(1) == 6
	assert len(cam.size()) == 2

	# type == 'quat'
	print('test_multi_camera_predictor: pose_type==quat')
	multi_camera_predictor = MutiCameraPredictor(100, num_cams=num_cams,
									   			 sc_bias=1.8, sc_scaling=0.05,
									   			 aec_scaling_ang=0.1,
									   			 aec_scaling_azm=math.pi/6.0,
									   			 aec_scaling_ele=math.pi/9.0,
									   			 aec_scaling_cyc=math.pi/9.0,
									   			 pose_type='quat')

	cam, idx, prob = multi_camera_predictor(feat)
	assert cam.size(0) == B
	assert cam.size(1) == 7
	assert len(cam.size()) == 2