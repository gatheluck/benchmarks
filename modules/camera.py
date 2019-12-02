import math

import torch

class QuatPredictor(torch.nn.Module):
	"""
	"""
	def __init__(self, input_dim):
		super(QuatPredictor, self).__init__()
		self.fc = torch.nn.Linear(input_dim, 4)

	def forward(self, feat):
		quat = self.fc(feat)
		quat = torch.nn.functional.normalize(quat,dim=-1)
		return quat

	def init_to_zero_rotation(self):
		self.fc.weight.data.normal_(0, 0.02)
		self.fc.bias = torch.nn.Parameter(torch.FloatTensor([1,0,0,0]).type(self.fc.bias.data.type()))

class ScalePredictor(torch.nn.Module):
	"""
	"""
	def __init__(self, input_dim, bias, scale_factor):
		super(ScalePredictor, self).__init__()
		self.fc = torch.nn.Linear(input_dim, 1)
		self.bias = bias
		self.scale_factor = scale_factor

	def forward(self, feat):
		scale = self.fc(feat)*self.scale_factor + self.bias
		scale = torch.nn.functional.relu(scale) + 1e-12
		return scale

class CameraPredictor(torch.nn.Module):
	"""
	"""
	def __init__(self, input_dim, scale_bias, scale_factor):
		super(CameraPredictor, self).__init__()

		output_dim = input_dim # output is same size as input
		self.encoder = torch.nn.Sequential(
			torch.nn.Linear(input_dim, output_dim),
			torch.nn.LeakyReLU(0.1, inplace=True),
			torch.nn.Linear(output_dim, output_dim),
			torch.nn.LeakyReLU(0.1, inplace=True)
		)

		self.scale_predictor = ScalePredictor(input_dim, scale_bias, scale_factor)
		self.trans_predictor = torch.nn.Linear(input_dim, 2)
		self.quat_predictor = QuatPredictor(input_dim)
		self.prob_predictor = torch.nn.Linear(input_dim, 1)

	def forward(self, feat):
		feat = self.encoder(feat)
		scale = self.scale_predictor(feat)
		trans = self.trans_predictor(feat)
		quat  = self.quat_predictor(feat)
		logit = self.prob_predictor(feat)
		return torch.cat([scale, trans, quat, logit], dim=-1)


class MutiCameraPredictor(torch.nn.Module):
	"""
	"""
	def __init__(self, input_dim, scale_bias, scale_factor, num_cams):
		super(MutiCameraPredictor, self).__init__()
		assert input_dim >= 1
		assert scale_bias >= 0.0
		assert scale_factor > 0.0
		assert num_cams >= 1

		self.num_cams = num_cams
		output_dim = input_dim # output is same size as input
		
		self.encoder = torch.nn.Sequential(
			torch.nn.Linear(input_dim, output_dim),
			torch.nn.LeakyReLU(0.1, inplace=True),
			torch.nn.Linear(output_dim, output_dim),
			torch.nn.LeakyReLU(0.1, inplace=True)
		)

		self.camera_predictors = torch.nn.ModuleList([
			CameraPredictor(input_dim, scale_bias, scale_factor) for i in range(self.num_cams)
		])

	def forward(self, feat):
		"""
		Args:
		- feat (torch.Tensor/[B,dim]):
		Return:
		- camera_sampled (torch.Tensor/[B,7]): 
		- idx (torch.Tensor/[B,1]): 
		- prob (torch.Tensor/[B,N]):
		"""
		feat = self.encoder(feat) # (1, 512)

		# prediction loop for multiple cameras
		camera_preds = []
		for i in range(self.num_cams):
			camera_preds.append(self.camera_predictors[i].forward(feat))
		camera_preds = torch.stack(camera_preds, dim=1)

		# convert logit to probability
		scale_trans_quat = camera_preds[:,:,0:7] 	#(B,N,7)
		logit = camera_preds[:,:,7].unsqueeze(-1) #(B,N,1)
		prob  = torch.nn.functional.softmax(logit, dim=1)
		camera_preds = torch.cat([scale_trans_quat, prob], dim=2) #(B,N,8)

		# sample one camera based on Multinomial distribution
		camera_sampled, idx = self.sample(camera_preds)
		
		return camera_sampled, idx, prob.squeeze(-1)

	def sample(self, camera_preds):
		"""
		Args:
		- camera_preds (torch.Tensor/[B,N,8]):
		Return:
		- camera_sampled (torch.Tensor/[B,7]): 
		- idx (torch.Tensor/[B,1]): 
		"""
		assert camera_preds.size(2) == 8
		dist = torch.distributions.multinomial.Multinomial(probs=camera_preds[:,:,7])
		sample = dist.sample() #(B,N)
		idx = torch.nonzero(sample)[:,1].reshape(-1,1) #(B,1)
		camera_sampled = torch.gather(camera_preds[:, :, 0:7], dim=1, index=idx.unsqueeze(1).repeat(1,1,7)) #(B,1,7)
		camera_sampled = camera_sampled.squeeze(1) #(B,7)

		return camera_sampled, idx