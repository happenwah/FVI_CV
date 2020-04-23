import math
import torch
import torch.nn as nn
from .prior.priors import *
from .variational_dist import Q_FCDenseNet103_FVI

class FVI(nn.Module):
	def __init__(self, x_size, **args):
		super(FVI, self).__init__()
		self.x_size = x_size
		self.L = 20
		self.device = args['device']
		self.x_inducing_var = args['x_inducing_var']
		self.n_inducing = args['n_inducing']
		self.prior_type = args['f_prior']
		self.add_cov_diag = args['add_cov_diag']
		assert self.prior_type == 'cnn_gp'
		assert self.add_cov_diag
		
		self.q = Q_FCDenseNet103_FVI(L=self.L, diag=self.add_cov_diag)
		if self.prior_type == 'cnn_gp':
			self.prior = f_prior_BNN(x_size, self.device)
			print('GP Prior, Bayesian CNN equivalent kernel')

	def _generate_x_c(self, x_t):
		randn = math.sqrt(self.x_inducing_var) * torch.randn_like(x_t[:self.n_inducing])
		x_c = (x_t[:self.n_inducing] + randn).clamp_(0., 1.)
		return x_c

	def q_params(self, input_q):
		q_sample_mean, q_sample_cov_out, q_cov_out_diag, q_logvar_aleatoric = self.q(input_q)
		N = q_sample_mean.size(0)
		q_sample_mean = q_sample_mean.contiguous().view(N, -1)
		q_sample_cov_out = q_sample_cov_out.contiguous().view(N, self.L, -1)
		q_sample_cov = torch.einsum('ijk,mjk->ikm', q_sample_cov_out, q_sample_cov_out) / self.L
		q_cov_out_diag = q_cov_out_diag.contiguous().view(N, -1).exp()
		for i in range(N):
			q_sample_cov[i,:,i] += q_cov_out_diag[i,:]
		q_logvar_aleatoric = q_logvar_aleatoric.contiguous().view(N, -1)

		return q_sample_mean, q_sample_cov, q_logvar_aleatoric

	def q_FVI(self, x_t, x_c):
		N_t = x_t.size(0)
		if x_c is not None:
			x = torch.cat((x_t, x_c), 0)
		else:
			x = x_t
		q_mean, q_cov, q_logvar_aleatoric = self.q_params(x)
		q_logvar_aleatoric = q_logvar_aleatoric[:N_t]
		return q_mean, q_cov, q_logvar_aleatoric

	def predict(self, x_t):
		self.q.eval()
		with torch.no_grad():
			q_mean, q_cov, q_logvar_aleatoric = self.q_params(x_t)
			q_aleatoric = q_logvar_aleatoric.exp()
			q_cov = torch.diagonal(q_cov, dim1=0, dim2=2).t()
		self.q.train()
		return q_mean, q_cov, q_aleatoric

	def predict_runtime(self, x_t):
		self.q.eval()
		start = torch.cuda.Event(enable_timing=True)
		end = torch.cuda.Event(enable_timing=True)
		with torch.no_grad():
			start.record()
			q_mean, q_cov_out, q_cov_out_diag, q_logvar_aleatoric = self.q(x_t)
			q_var_aleatoric = q_logvar_aleatoric.exp()
			q_cov_diag = torch.einsum('ijkl,ijkl->ikl',q_cov_out, q_cov_out) / self.L + q_cov_out_diag.exp()
			end.record()
			torch.cuda.synchronize()
			time = start.elapsed_time(end)
		print('Inference time elapsed (s): {}'.format(time))
		self.q.train()
		return time

	def forward(self, x_t):
		x_c = self._generate_x_c(x_t)
		q_mean, q_cov, lik_logvar = self.q_FVI(x_t, x_c)
		prior_mean, prior_cov = self.prior(x_t, x_c)
		return lik_logvar, q_mean, q_cov, prior_mean, prior_cov
