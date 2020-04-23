import torch
from .variational_dist import Q_FCDenseNet103_MCDropout
from .utils.torch_utils import apply_dropout
import torch.nn as nn
import torch.nn.functional as F

class MCD_seg(nn.Module):
	def __init__(self, num_classes, model_type):
		super().__init__()
		self.model_type = model_type
		self.num_classes = num_classes
		self.model = Q_FCDenseNet103_MCDropout(out_chans=num_classes, p=0.2).cuda()

	def predict(self, x, S=50, return_probs=False):
		self.model.eval()
		if self.model_type=='deterministic':
			mean, logvar_aleatoric = self.model.forward(x)
			rescaled_logits = mean * torch.exp(-logvar_aleatoric)
			probs = F.softmax(rescaled_logits, dim=1).squeeze(0)
		elif self.model_type=='mcd':
			self.model.apply(apply_dropout)
			probs_list = []
			for _ in range(S):
				with torch.no_grad():
					mean, logvar_aleatoric = self.model.forward(x)
				rescaled_logits = mean * torch.exp(-logvar_aleatoric)
				probs = F.softmax(rescaled_logits, dim=1)
				probs_list.append(probs)
			probs = torch.cat(probs_list, 0).mean(0)
		self.model.train()
		if return_probs:
			out_class = probs.argmax(0)
			out_probs = probs
			out_entropy = - (torch.log(probs + 1e-12) * probs).sum(0)
			return out_class.view(1, -1), out_entropy.view(1, -1), out_probs.view(1, probs.size(0), -1)
		else:
			out_class = probs.argmax(0)
			return out_class

	def predict_runtime(self, x, S):
		self.model.eval()
		B, _, H, W = x.size()
		assert B == 1, 'Predict on one test input'
		start = torch.cuda.Event(enable_timing=True)
		end = torch.cuda.Event(enable_timing=True)
		if self.model_type=='deterministic':
			with torch.no_grad():
				start.record()
				mean, logvar_aleatoric = self.model.forward(x)
				rescaled_logits = mean * torch.exp(-logvar_aleatoric)
				probs = F.softmax(rescaled_logits, dim=1)
				out_class = rescaled_logits.argmax(1)
				out_entropy = - (torch.log(probs + 1e-12) * probs).sum(1)
				end.record()
				torch.cuda.synchronize()
				time = start.elapsed_time(end)
			print('Inference time elapsed (ms): {}'.format(time))
			return time
		elif self.model_type=='mcd':
			self.model.apply(apply_dropout)
			probs = torch.zeros((B, self.num_classes, H, W)).cuda()
			with torch.no_grad():
				start.record()
				for _ in range(S):
					mean, logvar_aleatoric = self.model.forward(x)
					rescaled_logits = mean * torch.exp(-logvar_aleatoric)
					_probs = F.softmax(rescaled_logits, dim=1)
					probs += _probs / S
				out_class = probs.argmax(1)
				out_entropy = - (torch.log(probs + 1e-12) * probs).sum(1)
				end.record()
				torch.cuda.synchronize()
				time = start.elapsed_time(end)
			print('Inference time elapsed (ms): {}'.format(time))
			return time

	def forward(self, x):
		mean, logvar_aleatoric = self.model.forward(x)
		return mean, logvar_aleatoric
