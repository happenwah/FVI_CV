import torch
import torch.nn as nn
if __name__ == '__main__':
	from cnn_gp_prior import CNN_GP_prior
else:
	from .cnn_gp_prior import CNN_GP_prior

class f_prior_BNN(nn.Module):
	def __init__(self, out_size, device, num_channels_output=1):
		super().__init__()
		self.P = out_size[1] * out_size[0]
		self.C = num_channels_output
		self.model = CNN_GP_prior(out_size, num_channels_output=num_channels_output)
		self.device = device
		self.diag_noise = 1e-1
	def forward(self, x_t, x_c=None):
		assert len(x_t.size()) == 4
		if x_c is not None:
			x = torch.cat((x_t, x_c), 0)
		else:
			x = x_t
		N = x.size(0)
		prior_cov = self.model.compute_K(x)
		#Depth estimation
		if self.C == 1:
			prior_mean = 0.5 * torch.ones((N, self.P*self.C)).to(self.device)
		#Semantic segmentation
		elif self.C > 1:
			prior_mean = torch.ones((N, self.P*self.C)).to(self.device)
		for i in range(N):
			prior_cov[i,:,i] += self.diag_noise
		return prior_mean, prior_cov

from ..model.densenet import FCDenseNet_103

class BNN(nn.Module):
	def __init__(self, **kwargs):
		super().__init__()
		self.model = FCDenseNet_103(**kwargs)
		self.model.cuda()
		self.model.eval()
		self.model.requires_grad_(False)
		self.STD_PRIOR = .039
		self.diag_noise = 1e-1

	def _set_weights_normal(self, m):
		if type(m) == torch.nn.Conv2d or type(m) == torch.nn.ConvTranspose2d:
			m.weight.data.normal_(0., self.STD_PRIOR)
			m.bias.data.normal_(0., self.STD_PRIOR)

	def _sample(self, x):
		self.model.apply(self._set_weights_normal)
		out = self.model.forward(x)
		return out

	def forward(self, x_t, x_c=None, S=20):
		assert len(x_t.size()) == 4
		if x_c is not None:
			x = torch.cat((x_t, x_c), 0)
		else:
			x = x_t
		B = x.size(0)
		out_list = []
		for _ in range(S):
			out = self._sample(x)
			out_list.append(out.unsqueeze(0))
		outs = torch.cat(out_list, 0).contiguous().view(S, B, -1)
		out_mean = outs.mean(0)
		phi = outs - out_mean.unsqueeze(0).expand(S, -1, -1)
		out_cov = torch.einsum('ijk,ilk->jkl', phi, phi)
		for i in range(B):
			out_cov[i,:,i] += self.diag_noise
		return out_mean, out_cov
