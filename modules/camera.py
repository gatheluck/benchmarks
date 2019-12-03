import math

import torch

class PosePredictorQuat(torch.nn.Module):
	"""
	"""
	def __init__(self, input_dim):
		super(PosePredictorQuat, self).__init__()
		self.fc = torch.nn.Linear(input_dim, 4)

	def forward(self, feat):
		quat = self.fc(feat)
		quat = torch.nn.functional.normalize(quat,dim=-1)
		return quat

	def init_to_zero_rotation(self):
		self.fc.weight.data.normal_(0, 0.02)
		self.fc.bias = torch.nn.Parameter(torch.FloatTensor([1,0,0,0]).type(self.fc.bias.data.type()))

class PosePredictorAEC(torch.nn.Module):
	"""
	Camera pose predictor based on Azimuth, Elevation, Cycle rotation (AEC).
	"""
	def __init__(self, input_dim: int, 
				 scaling_factor_ang: float = 0.1,
				 scaling_factor_azm: float = math.pi/6.0,
				 scaling_factor_ele: float = math.pi/9.0,
				 scaling_factor_cyc: float = math.pi/9.0):
		super(PosePredictorAEC, self).__init__()
		# scale factors
		self.scaling_factors = dict()
		self.scaling_factors['ang'] = scaling_factor_ang
		self.scaling_factors['azm'] = scaling_factor_azm
		self.scaling_factors['ele'] = scaling_factor_ele
		self.scaling_factors['cyc'] = scaling_factor_cyc

		self.fc = torch.nn.Linear(input_dim, 3)

	def forward(self, feat: torch.Tensor):
		"""
		Args
		- feat (B, input_dim): 
		"""
		angles = torch.tanh(self.fc(feat) * self.scaling_factors['ang'])
		azm = angles[:, 0] * self.scaling_factors['azm']
		ele = angles[:, 1] * self.scaling_factors['ele']
		cyc = angles[:, 2] * self.scaling_factors['cyc']

		return torch.stack([azm, ele, cyc], dim=-1)

class ScalePredictor(torch.nn.Module):
	"""
	"""
	def __init__(self, input_dim: int, 
				 bias: float, 
				 scaling_factor: float):
		super(ScalePredictor, self).__init__()
		self.fc = torch.nn.Linear(input_dim, 1)
		self.bias = bias
		self.scaling_factor = scaling_factor

	def forward(self, feat):
		scale = self.fc(feat)*self.scaling_factor + self.bias
		scale = torch.nn.functional.relu(scale) + 1e-12
		return scale

class CameraPredictor(torch.nn.Module):
	"""
	"""
	def __init__(self, input_dim: int, 
				 sc_bias: float, 
				 sc_scaling: float,
				 aec_scaling_ang: float, 
				 aec_scaling_azm: float,
				 aec_scaling_ele: float,
				 aec_scaling_cyc: float,
				 pose_type: str = 'aec'):
		super(CameraPredictor, self).__init__()
		if pose_type not in ['aec', 'quat']: raise ValueError

		output_dim = input_dim # output is same size as input
		self.encoder = torch.nn.Sequential(
			torch.nn.Linear(input_dim, output_dim),
			torch.nn.LeakyReLU(0.1, inplace=True),
			torch.nn.Linear(output_dim, output_dim),
			torch.nn.LeakyReLU(0.1, inplace=True)
		)

		# pose predictor
		if pose_type == 'aec':
			self.pose_predictor = PosePredictorAEC(input_dim,
												   aec_scaling_ang,
												   aec_scaling_azm,
												   aec_scaling_ele,
												   aec_scaling_cyc)
		elif pose_type == 'quat':
			self.pose_predictor = PosePredictorQuat(input_dim)
		else:	
			raise NotImplementedError

		self.scale_predictor = ScalePredictor(input_dim, 
											  sc_bias, 
											  sc_scaling)
		self.trans_predictor = torch.nn.Linear(input_dim, 2)
		self.prob_predictor = torch.nn.Linear(input_dim, 1)

	def forward(self, feat):
		feat = self.encoder(feat)
		scale = self.scale_predictor(feat)
		trans = self.trans_predictor(feat)
		pose  = self.pose_predictor(feat)
		logit = self.prob_predictor(feat)
		return torch.cat([scale, trans, pose, logit], dim=-1)


class MutiCameraPredictor(torch.nn.Module):
	"""
	"""
	def __init__(self, input_dim:int, 
				 num_cams: int,
				 sc_bias: float, 
				 sc_scaling: float,
				 aec_scaling_ang: float, 
				 aec_scaling_azm: float,
				 aec_scaling_ele: float,
				 aec_scaling_cyc: float,
				 pose_type: str = 'aec'):

		super(MutiCameraPredictor, self).__init__()
		assert input_dim >= 1
		assert num_cams >= 1
		assert sc_bias >= 0.0
		assert sc_scaling > 0.0
		assert aec_scaling_ang > 0.0
		assert aec_scaling_azm > 0.0
		assert aec_scaling_ele > 0.0
		assert aec_scaling_cyc > 0.0

		self.num_cams = num_cams
		output_dim = input_dim # output is same size as input
		
		self.encoder = torch.nn.Sequential(
			torch.nn.Linear(input_dim, output_dim),
			torch.nn.LeakyReLU(0.1, inplace=True),
			torch.nn.Linear(output_dim, output_dim),
			torch.nn.LeakyReLU(0.1, inplace=True)
		)

		self.camera_predictors = torch.nn.ModuleList([
			CameraPredictor(input_dim, 
							sc_bias,
							sc_scaling,
							aec_scaling_ang,
							aec_scaling_azm,
							aec_scaling_ele,
							aec_scaling_cyc,
							pose_type) for i in range(self.num_cams)
		])

	def forward(self, feat):
		"""
		Args:
		- feat (torch.Tensor/[B,dim]):
		Return:
		- camera_sampled (torch.Tensor/[B,7])or(torch.Tensor/[B,6]): 
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
		scale_trans_pose = camera_preds[:,:,0:-1] 	#(B,N,7)or(B,N,6)
		logit = camera_preds[:,:,-1].unsqueeze(-1) #(B,N,1)
		prob  = torch.nn.functional.softmax(logit, dim=1)
		camera_preds = torch.cat([scale_trans_pose, prob], dim=2) #(B,N,8)or(B,N,7)

		# sample one camera based on Multinomial distribution
		camera_sampled, idx = self.sample(camera_preds)
		
		return camera_sampled, idx, prob.squeeze(-1)

	def sample(self, camera_preds: torch.Tensor):
		"""
		Args:
		- camera_preds (B,N,8)or(B,N,7):
		Return:
		- camera_sampled (B,7)or(B,6): 
		- idx (torch.Tensor/[B,1]): 
		"""
		camera_dim = camera_preds.size(-1)
		assert (camera_dim==8) or (camera_dim==7)

		dist = torch.distributions.multinomial.Multinomial(probs=camera_preds[:,:,-1])
		sample = dist.sample() #(B,N)
		idx = torch.nonzero(sample)[:,1].reshape(-1,1) #(B,1)
		camera_sampled = torch.gather(camera_preds[:, :, 0:-1], dim=1, index=idx.unsqueeze(1).repeat(1,1,camera_dim-1)) #(B,1,7) or (B,1,8)
		camera_sampled = camera_sampled.squeeze(1) #(B,7) or (B,8)

		return camera_sampled, idx