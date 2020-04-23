import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from ..elbo_depth import weight_aleatoric
import cv2

def berhu_cdf(y, mean, std, c):
	dist = torch.distributions.normal.Normal(0., 1.)
	Phi_minus_c = dist.cdf(-c.sqrt())
	Z_0 = 2 * (1. - torch.exp(-c) +  torch.exp(-0.5 * c) * (2 * math.pi * c).sqrt() * Phi_minus_c)
	mask_left = (y < -c).type(torch.FloatTensor)
	mask_mid = ((y >= -c) & (y <= c)).type(torch.FloatTensor)
	mask_right = (y > c).type(torch.FloatTensor)
	normal_dist = torch.distributions.Normal(mean, std * c.sqrt())
	laplace_dist = torch.distributions.Laplace(mean, std)
	C_normal = (2 * math.pi * c).sqrt() * torch.exp(-0.5 * c)
	C_laplace = 2.
	result_left = mask_left * normal_dist.cdf(y) * C_normal
	result_mid = mask_mid * ((laplace_dist.cdf(y) - laplace_dist.cdf(-c)) * C_laplace + normal_dist.cdf(-c) * C_normal)
	result_right = mask_right * (((normal_dist.cdf(y) - normal_dist.cdf(c) + normal_dist.cdf(-c)) * C_normal) + (laplace_dist.cdf(c) - laplace_dist.cdf(-c)) * C_laplace)
	result = (result_left + result_mid + result_right) / Z_0
	return result

def run_runtime_fvi(model, test_set, likelihood, exp_name, c_threshold=None):
	K = 100
	if likelihood == 'berhu':
		w_threshold = weight_aleatoric(c_threshold)
	elif likelihood == 'laplace':
		w_threshold = 2.
	X_test = torch.from_numpy(list(test_set)[0][0]).unsqueeze(0).cuda()
	model.eval()
	#First forward pass, ignore
	time_list = []
	for _ in range(K + 1):
		with torch.no_grad():
			if not likelihood == 'gaussian':
				end = model.predict_runtime(X_test, w_threshold)
			elif likelihood == 'gaussian':
				end = model.predict_runtime(X_test)
		time_list.append(end)
	time_list = np.array(time_list)[1:]
	time_mean = time_list.mean()
	time_std = time_list.std()
	np.savetxt('{}_mean_runtime.txt'.format(exp_name), [time_mean])
	np.savetxt('{}_std_runtime.txt'.format(exp_name), [time_std]) 
	print(time_list)
	print('Inference time elapsed (s), mean: {} || std: {}'.format(time_mean, time_std))
	model.train()

def pixel_wise_calibration_curve(true_img, mean_pred, std_pred_model, std_pred_aleatoric, mask_test, likelihood, c_threshold=None, S=50):
	true_img = true_img.view(-1).cpu()
	mean_pred = mean_pred.view(-1).cpu()
	mask = mask_test.view(-1).cpu()
	std_pred_model = std_pred_model.view(-1).cpu()
	std_pred_aleatoric = std_pred_aleatoric.view(-1).cpu()
	std_pred = (torch.pow(std_pred_model, 2) + torch.pow(std_pred_aleatoric, 2)).sqrt()
	if likelihood == 'gaussian':
		dist = torch.distributions.Normal(mean_pred, std_pred)
		probs_img = dist.cdf(true_img).cpu().detach().numpy()
	else:
		probs_img = torch.zeros_like(true_img)
		for s in range(S):
			eps = torch.randn_like(std_pred_model)
			f = mean_pred + std_pred_model * eps
			if likelihood == 'laplace':
				scale = std_pred_aleatoric/np.sqrt(2.)
				dist = torch.distributions.Laplace(f, scale)
				probs_img += dist.cdf(true_img) / S
			elif likelihood == 'berhu':
				w_threshold = weight_aleatoric(c_threshold).cpu().numpy()
				scale = std_pred_aleatoric/np.sqrt(w_threshold)
				probs_img += berhu_cdf(true_img, f, scale, c_threshold.cpu()) / S
		probs_img = probs_img.numpy()
	#Sharpness score, removing missing depth locations
	sharpness = ((std_pred ** 2)[mask]).mean().item()
	n_levels = 10
	true_freq = np.linspace(0., 1., n_levels)
	pred_freq = np.zeros_like(true_freq)
	probs_masked = probs_img[mask.numpy()]
	for i,level in enumerate(true_freq):
		mask_level = (probs_masked <= level).astype(np.float32)
		if mask_level.sum() > 0.:
			pred_freq[i] = mask_level.mean()
		else:
			pred_freq[i] = 0.
	#Calibration score, uniform weighting bins
	calibration = ((true_freq - pred_freq)**2 * 1.).sum()
	return pred_freq, true_freq, calibration, sharpness

def test(model, img_test, y_test, idx, mask_test, dataset, exp_name, likelihood, c_threshold=None, trainset=False, saveplot=False):
	if likelihood == 'berhu':
		w_threshold = weight_aleatoric(c_threshold)
		mean_pred, var_model_pred, var_aleatoric_pred = model.predict(img_test, w_threshold)
	elif likelihood == 'laplace':
		w_threshold = 2.
		mean_pred, var_model_pred, var_aleatoric_pred = model.predict(img_test, w_threshold)
	elif likelihood == 'gaussian':
		mean_pred, var_model_pred, var_aleatoric_pred = model.predict(img_test)
	if dataset == 'make3d':
		C = 70.
		H, W = 168, 224
	std_model_pred = var_model_pred.sqrt()
	std_aleatoric_pred = var_aleatoric_pred.sqrt()
	pred_freq, true_freq, calibration, sharpness = pixel_wise_calibration_curve(y_test, mean_pred, std_model_pred, std_aleatoric_pred, mask_test, likelihood, c_threshold=c_threshold)
	#Original scale
	sharpness *= (C**2)
	mean_pred *= C
	y_test *= C
	std_model_pred *= C
	std_aleatoric_pred *= C
	im_ratio = float(H/W)
	img_test = img_test.view(3, H, W).permute(1, 2, 0).cpu().numpy()
	#BGR -> RGB
	_img_test = np.zeros_like(img_test)
	_img_test[:,:,0] = img_test[:,:,2]
	_img_test[:,:,1] = img_test[:,:,1]
	_img_test[:,:,2] = img_test[:,:,0]
	img_test = _img_test
	mean_pred = mean_pred.view(H, W).cpu()
	y_test = y_test.view(H, W).cpu()
	mask_test = mask_test.view(H, W).cpu()
	std_model_pred = std_model_pred.view(H, W).cpu()
	std_aleatoric_pred = std_aleatoric_pred.view(H, W).cpu()
	RMSE = torch.sqrt(torch.pow(mean_pred - y_test, 2)[mask_test].mean())
	mean_log10_error = torch.abs(torch.log10(mean_pred) - torch.log10(y_test))[mask_test].mean()
	mean_abs_rel_error = torch.abs((mean_pred - y_test)/y_test)[mask_test].mean()
	if saveplot:
		print('Img idx: {} || Calibration score: {:4f} || Sharpness score: {:4f}'.format(idx, calibration, sharpness))
		rgb_true = cv2.resize(img_test, None, fx=float(H/W), fy=float(W/H))
		img_true = cv2.resize(y_test.numpy(), None, fx=float(H/W), fy=float(W/H))
		img_pred = cv2.resize(mean_pred.numpy(), None, fx=float(H/W), fy=float(W/H))
		vmin_,vmax_ = 1.75, 35.
		img_std_aleatoric = cv2.resize(std_aleatoric_pred.numpy(), None, fx=float(H/W), fy=float(W/H))
		img_std_model = cv2.resize(std_model_pred.numpy(), None, fx=float(H/W), fy=float(W/H))
		fig = plt.figure(1,figsize=(12, 2))
		ax1 = plt.subplot(171)
		im1 = ax1.imshow(rgb_true)
		ax1.axis('off')
		ax2 = plt.subplot(172)
		im2 = ax2.imshow(img_true,cmap='magma',vmin=vmin_,vmax=vmax_)
		ax2.axis('off')
		cb2 = fig.colorbar(im2, ax=ax2, fraction=0.046*im_ratio, pad=0.04)
		cb2.ax.tick_params(labelsize=5)
		cb2.ax.tick_params(size=0)
		ax3 = plt.subplot(173)
		im3 = ax3.imshow(img_pred,cmap='magma',vmin=vmin_,vmax=vmax_)
		ax3.axis('off')
		cb3 = fig.colorbar(im3, ax=ax3, fraction=0.046*im_ratio, pad=0.04)
		cb3.ax.tick_params(labelsize=5)
		cb3.ax.tick_params(size=0)
		ax4 = plt.subplot(174)
		#im4 = ax4.imshow(img_std_model,cmap='nipy_spectral')
		im4 = ax4.imshow(np.sqrt(img_std_aleatoric**2 + img_std_model**2),cmap='nipy_spectral')
		ax4.axis('off')
		cb4 = fig.colorbar(im4, ax=ax4, fraction=0.046*im_ratio, pad=0.04, format="%d")
		cb4.ax.tick_params(labelsize=5)
		cb4.ax.tick_params(size=0)
		ax5 = plt.subplot(175)
		ax5.set_aspect(im_ratio)
		ax5.xaxis.set_tick_params(labelsize=5)
		ax5.yaxis.set_tick_params(labelsize=5)
		ax5.plot(pred_freq, true_freq, color='red')
		ax5.plot([0., 1.], [0., 1.], 'g--')
		plt.tight_layout()
		if trainset:
			plt.savefig('{}_{}_results_train_pred_{}.pdf'.format(dataset, exp_name, idx), bbox_inches='tight', pad_inches=0.1)
		else:
			plt.savefig('{}_{}_results_test_pred_{}.pdf'.format(dataset, exp_name, idx), bbox_inches='tight', pad_inches=0.1)
		plt.close()
	return calibration, sharpness, RMSE, mean_log10_error, mean_abs_rel_error, mean_pred.view(1, -1), y_test.view(1, -1), mask_test.view(1, -1)

def run_test_fvi_per_image(epoch, model, test_set, N_test, dataset, exp_name, likelihood, c_threshold=None, mkdir=False):
	#No. of test plots to save
	if mkdir:
		import os
		new_dir = './results_{}'.format(exp_name)
		os.makedirs(new_dir, exist_ok=True)
		os.chdir(new_dir)
		n_save = N_test
	else:
		n_save = 15
	pred_list = []
	y_list = []
	mask_list = []
	model.eval()
	test_generator = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
	for k, (X_test, Y_test) in enumerate(list(test_generator)):
		img_test = X_test[0].unsqueeze(0).cuda()
		y_test = Y_test[0].unsqueeze(0)
		if dataset == 'make3d':
			mask_test = (y_test < 1.0)
		if k < n_save:
			cal, sh, RMSE, mean_log10_error, mean_abs_rel_error, pred, y, mask = test(model, img_test, y_test, k, mask_test, dataset, exp_name, likelihood, c_threshold=c_threshold, saveplot=True)
			np.savetxt('{}_{}_img_{}_calibration.txt'.format(dataset, exp_name, k), [cal])
			np.savetxt('{}_{}_img_{}_sharpness_mean.txt'.format(dataset, exp_name, k), [sh])
			np.savetxt('{}_{}_img_{}_rmse.txt'.format(dataset, exp_name, k), [RMSE])
			np.savetxt('{}_{}_img_{}_log10e.txt'.format(dataset, exp_name, k), [mean_log10_error])
			np.savetxt('{}_{}_img_{}_mean_abs_rel_error.txt'.format(dataset, exp_name, k), [mean_abs_rel_error])
		else:
			cal, sh, RMSE, mean_log10_error, mean_abs_rel_error, pred, y, mask = test(model, img_test, y_test, k, mask_test, dataset, exp_name, likelihood, c_threshold=c_threshold)
		pred_list.append(pred)
		y_list.append(y)
		mask_list.append(mask)
		del img_test, y_test, mask_test, pred, y, mask
	preds = torch.cat(pred_list, 0)
	Y = torch.cat(y_list, 0)
	masks = torch.cat(mask_list, 0)
	preds_masked = preds[masks]
	Y_masked = Y[masks]
	test_rmse = torch.pow(preds_masked - Y_masked, 2).mean().sqrt().item()
	test_mare = torch.abs((preds_masked - Y_masked)/Y_masked).mean().item()
	test_log10e = torch.abs(torch.log10(preds_masked) - torch.log10(Y_masked)).mean().item()
	np.savetxt('{}_{}_rmse_epoch_{}.txt'.format(dataset, exp_name, epoch), [test_rmse])
	np.savetxt('{}_{}_log10e_epoch_{}.txt'.format(dataset, exp_name, epoch), [test_mare])
	np.savetxt('{}_{}_mean_abs_rel_error_epoch_{}.txt'.format(dataset, exp_name, epoch), [test_log10e])
	print('Test rmse: {} || Test rel: {} || Test log10: {}'.format(test_rmse, test_mare, test_log10e))
	model.train()
