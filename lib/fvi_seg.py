import math
import torch
import torch.nn as nn
from .prior.priors import *
import torch.nn.functional as F
from .variational_dist import Q_FCDenseNet103_FVI

class FVI_seg(nn.Module):
	def __init__(self, x_size, num_classes, **args):
		super(FVI_seg, self).__init__()
		self.num_classes = num_classes
		self.L = 20
		self.S = 20
		self.device = args['device']
		self.x_inducing_var = args['x_inducing_var']
		self.n_inducing = args['n_inducing']
		self.prior_type = args['f_prior']
		self.add_cov_diag = args['add_cov_diag']
		self.standard_cross_entropy = args['standard_cross_entropy']
		assert self.prior_type == 'cnn_gp'
		assert self.add_cov_diag

		self.q = Q_FCDenseNet103_FVI(L=self.L, diag=self.add_cov_diag, out_chans=num_classes)
		if self.prior_type == 'cnn_gp':
			self.prior = f_prior_BNN(x_size, self.device, num_channels_output=num_classes)
			print('GP Prior, Bayesian CNN equivalent kernel')

	def _generate_x_c(self, x_t):
		randn = math.sqrt(self.x_inducing_var) * torch.randn_like(x_t[:self.n_inducing])
		x_c = (x_t[:self.n_inducing] + randn).clamp_(0., 1.)
		return x_c

	def _sample_functions(self, q_mean, q_cov, num_samples):
		N, P, _ = q_cov.size()
		q_cov_diag = torch.diagonal(q_cov, dim1=0, dim2=2).t()
		Z = torch.randn(num_samples, N, P).to(self.device)
		f = torch.einsum('ij,lij->lij', q_cov_diag.sqrt(), Z)
		f_values = q_mean.unsqueeze(0).expand(num_samples, -1, -1) + f
		return f_values

	def q_params(self, input_q):
		q_mean, q_cov_out, q_cov_out_diag, q_logvar_aleatoric = self.q(input_q)
		N = q_mean.size(0)
		#shape: (N, H*W*num_classes)
		q_mean = q_mean.contiguous().view(N, -1)
		#shape: (N, L, H*W*num_classes)
		q_cov_out = q_cov_out.contiguous().view(N, self.L, -1)
		q_cov = torch.einsum('ijk,mjk->ikm', q_cov_out, q_cov_out) / self.L
		q_cov_out_diag = q_cov_out_diag.contiguous().view(N, -1).exp()
		for i in range(N):
			q_cov[i,:,i] += q_cov_out_diag[i,:]
		q_logvar_aleatoric = q_logvar_aleatoric.contiguous().view(N, -1)
		return q_mean, q_cov, q_cov_out, q_logvar_aleatoric

	def q_FVI_seg(self, x_t, x_c, S=None):
		N_t = x_t.size(0)
		if x_c is not None:
			x = torch.cat((x_t, x_c), 0)
		else:
			x = x_t
		q_mean, q_cov, q_cov_out, q_logvar_aleatoric = self.q_params(x)
		q_logvar_aleatoric = q_logvar_aleatoric[:N_t]
		if S is None:
			S = self.S
		#Samples with epistemic uncertainty
		f_samples = self._sample_functions(q_mean[:N_t,:], q_cov[:N_t,:,:N_t], S)
		#Bolztmann likelihood
		if not self.standard_cross_entropy:
			q_logvar_aleatoric = q_logvar_aleatoric.unsqueeze(0).expand(S, -1, -1)
			f_samples = f_samples * torch.exp(-q_logvar_aleatoric)
		f_samples = f_samples.contiguous().view(S, N_t, self.num_classes, -1)
		return f_samples, q_mean, q_cov

	def predict(self, x_t, S=50, return_probs=False):
		self.q.eval()
		with torch.no_grad():
			f_samples, _, _ = self.q_FVI_seg(x_t, x_c=None, S=S)
			f_dist = F.softmax(f_samples, 2).mean(0)
			f_pred = f_dist.argmax(1)
			if return_probs:
				f_probs = f_dist
				f_entropy = - (torch.log(f_dist + 1e-12) * f_dist).sum(1)
				return f_pred, f_entropy, f_probs
		self.q.train()
		return f_pred

	def predict_runtime(self, x_t, S):
		self.q.eval()
		self.prior.cpu()
		B, _, H, W = x_t.size()
		assert B == 1, "Predict one image at a time"
		Z = torch.randn((S, B, self.num_classes, H, W)).to(self.device)
		start = torch.cuda.Event(enable_timing=True)
		end = torch.cuda.Event(enable_timing=True)
		with torch.no_grad():
			start.record()
			q_mean, q_cov_out, q_cov_out_diag, q_logvar_aleatoric = self.q(x_t)
			q_cov_out = q_cov_out.contiguous().view(B, self.L, -1, H, W)
			q_var = torch.einsum('ijklm,ijklm->iklm', q_cov_out, q_cov_out) / self.L + q_cov_out_diag.exp()
			f_samples = q_mean.unsqueeze(0).expand(S, B, -1, H, W) + torch.einsum('ijklm,jklm->ijklm', Z, q_var.sqrt())
			f_samples = torch.einsum('ijklm,jklm->ijklm', f_samples, torch.exp(-q_logvar_aleatoric))
			f_dist = F.softmax(f_samples, 2).mean(0)
			f_pred = f_dist.argmax(1)
			f_entropy = - (torch.log(f_dist + 1e-12) * f_dist).sum(1)
			end.record()
			torch.cuda.synchronize()
			time = start.elapsed_time(end)
			print('Inference time elapsed (ms): {}'.format(time))
		self.q.train()
		return time

	def forward(self, x_t):
		x_c = self._generate_x_c(x_t)
		f_samples, q_mean, q_cov= self.q_FVI_seg(x_t, x_c)
		prior_mean, prior_cov = self.prior(x_t, x_c)
		return f_samples, q_mean, q_cov, prior_mean, prior_cov
