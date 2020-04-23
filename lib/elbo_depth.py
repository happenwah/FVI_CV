"""
	Evidence Lower Bound (ELBO) for Functional Variational Inference (FVI)
	Depth experiments. Gaussian, Laplace and reverse Huber Likelihoods
"""

import torch
import math
from .utils.torch_utils import bmat_inv_logdet, bdiag_prod

def weight_aleatoric(c):
	dist = torch.distributions.normal.Normal(0., 1.)
	Phi_minus_c = dist.cdf(-c.sqrt())
	Z_0 = 2 * (1. - torch.exp(-c) +  torch.exp(-0.5 * c) * (2 * math.pi * c).sqrt() * Phi_minus_c)
	den = 4. - 4 * (c + 1.) * torch.exp(-c)  + 2 * c * torch.exp(-0.5 * c) * (2 * math.pi * c).sqrt() * Phi_minus_c
	result = den/Z_0
	return result

def gaussian_log_prob(samples, mean, logvar):
	aux = samples - mean
	result = - 0.5 * (torch.pow(aux, 2) * torch.exp(-logvar) + logvar + math.log(2 * math.pi))
	return result

def laplacian_log_prob(samples, mean, logvar):
	aux = samples - mean
	log_scale = 0.5 * logvar
	scale_inv = torch.exp(-log_scale)
	result = - (torch.abs(aux) * scale_inv + log_scale)
	return result

def log_lik_exact(y_t, q_mean, q_cov, logvar):
	aux = y_t - q_mean
	q_cov_trace = torch.diagonal(q_cov, offset=0, dim1=0, dim2=2).t()
	result = - 0.5 * ((torch.pow(aux, 2) + q_cov_trace) * torch.exp(-logvar) + logvar + math.log(2 * math.pi))
	return result

def kl_div(q_mean, q_cov, p_mean, p_cov):
	N, P, _ = q_cov.size()
	jitter = 1e-3
	q_cov_stable = torch.zeros_like(q_cov)
	for i in range(N):
		q_cov_stable[i,:,i] = q_cov[i,:,i] + jitter
	_, log_det_q = bmat_inv_logdet(q_cov_stable)
	p_cov_inv, log_det_p = bmat_inv_logdet(p_cov)
	logdet = log_det_p - log_det_q
	trace = torch.diagonal(bdiag_prod(p_cov_inv, q_cov_stable), dim1=0, dim2=2).sum(1).sum(0)
	m_ = (p_mean - q_mean).unsqueeze(2)
	Q_aux = bdiag_prod(p_cov_inv, m_).contiguous().view(-1, 1)
	Q = (m_.contiguous().view(-1, 1).t() @ Q_aux).squeeze(1).squeeze(0)
	return 0.5 * (logdet - N * P + trace + Q)

def fELBO_gaussian_depth(mask, y_t, lik_logvar, q_mean, q_cov, prior_mean, prior_cov, print_loss=False):
	N = y_t.size(0)

	y_t = y_t.view(N, -1)
	mask = mask.view(N, -1)
	P = y_t.size(1)
	log_lik_ = log_lik_exact(y_t, q_mean[:N,:], q_cov[:N,:,:N], lik_logvar)[mask]
	log_lik = log_lik_.mean(0)
	N_mask = mask.long().sum()

	kl = kl_div(q_mean, q_cov, prior_mean, prior_cov)
	#Lower bound to mini-batch log marginal likelihood
	kl /= N_mask

	if print_loss:
		print('No. of pixels after masking: ', N_mask.item())
		logger = 'Pixel-averaged Log Likelihood: {:.3f} || Pixel-averaged KL divergence: {:.3f}'.format(log_lik.item(), kl.item())
		print(logger)

	elbo = log_lik - kl

	return elbo

def fELBO_berhu_depth(mask, f_samples, y_t, lik_logvar, q_mean, q_cov, prior_mean, prior_cov, print_loss=False):               
	S, N, P = f_samples.size()
	y_t = y_t.view(N, -1)
	mask = mask.view(N, -1)
	q_mean_lik = f_samples
	y_t = y_t.unsqueeze(0).expand(S, -1, -1)
	lik_logvar = lik_logvar.unsqueeze(0).expand(S, -1, -1)
	c = 0.2 * torch.abs(y_t - q_mean_lik).max()
	c_no_grad = c.detach()
	Z_abs = torch.abs((y_t - q_mean_lik) * torch.exp(-0.5 * lik_logvar))
	mask_l1 = (Z_abs <= c_no_grad).type(torch.cuda.FloatTensor)
	mask_l2 = (Z_abs > c_no_grad).type(torch.cuda.FloatTensor)
	mask_l1.requires_grad = False
	mask_l2.requires_grad = False
	log_lik = - Z_abs * mask_l1 - (0.5 * (torch.pow(y_t - q_mean_lik, 2) * torch.exp(-lik_logvar) + torch.pow(c_no_grad, 2))/c_no_grad) * mask_l2 - 0.5 * lik_logvar
	log_lik = log_lik.mean(0)[mask].mean()
	N_mask = mask.long().sum()

	kl = kl_div(q_mean, q_cov, prior_mean, prior_cov)
	#Lower bound to mini-batch log marginal likelihood
	kl /= N_mask

	if print_loss:
		logger = 'Pixel-averaged Log Likelihood: {:.3f} || Pixel-averaged KL divergence: {:.3f}'.format(log_lik.item(), kl.item())                
		print(logger)
	elbo = log_lik - kl
	return -elbo, c_no_grad

def fELBO_laplace_depth(mask, f_samples, y_t, lik_logvar, q_mean, q_cov, prior_mean, prior_cov, print_loss=False):               
	S, N, _ = f_samples.size()
	y_t = y_t.view(N, -1)
	mask = mask.view(N, -1)
	q_mean_lik = f_samples
	y_t = y_t.unsqueeze(0).expand(S, -1, -1)
	lik_logvar = lik_logvar.unsqueeze(0).expand(S, -1, -1)
	log_lik = laplacian_log_prob(y_t, q_mean_lik, lik_logvar)
	log_lik = log_lik.mean(0)[mask].mean()
	N_mask = mask.long().sum()

	kl = kl_div(q_mean, q_cov, prior_mean, prior_cov)
	#Lower bound to mini-batch log marginal likelihood
	kl /= N_mask

	if print_loss:
		logger = 'Pixel-averaged Log Likelihood: {:.3f} || Pixel-averaged KL divergence: {:.3f}'.format(log_lik.item(), kl.item())                
		print(logger)
	elbo = log_lik - kl
	return -elbo
