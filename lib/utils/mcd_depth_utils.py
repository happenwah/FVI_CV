from lib.elbo_depth import gaussian_log_prob, laplacian_log_prob
from lib.utils.torch_utils import apply_dropout
from tqdm import tqdm

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from time import time

def _compute_runtime_mcd_depth(model, X_test, S=50):
	model.eval()
	model.apply(apply_dropout)
	pred_mean = torch.zeros((X_test.size(0), 1, X_test.size(2), X_test.size(3))).cuda()
	pred_pow = pred_mean
	var_aleatoric = pred_mean
	start = torch.cuda.Event(enable_timing=True)
	end = torch.cuda.Event(enable_timing=True)
	with torch.no_grad():
		start.record()
		for _ in range(S):
			pred, logvar_aleatoric = model(X_test)
			pred_mean += pred
			pred_pow += torch.pow(pred, 2)
			var_aleatoric += logvar_aleatoric.exp()
		pred_mean /= S
		var = pred_pow/S - torch.pow(pred_mean, 2) + var_aleatoric/S
		end.record()
		torch.cuda.synchronize()
		time = start.elapsed_time(end)
	return time

def run_runtime_mcd_depth(model, dataset, exp_name, S=50):
	K = 100
	X_test = list(dataset)[0][0]
	X_test = torch.from_numpy(X_test).unsqueeze(0).cuda()
	time_list = []
	for _ in range(K + 1):
		pred_list = []
		var_aleatoric_list = []
		time = _compute_runtime_mcd_depth(model, X_test, S=S)
		print('Time elapsed (ms): {}'.format(time))
		time_list.append(time)
	time_list = np.array(time_list)[1:]
	time_mean = time_list.mean()
	time_std = time_list.std()
	np.savetxt('{}_mean_runtime.txt'.format(exp_name), [time_mean])
	np.savetxt('{}_std_runtime.txt'.format(exp_name), [time_std])
	print('Inference time elapsed (s), mean: {} || std: {}'.format(time_mean, time_std))
	model.train()
			

def pixel_wise_calibration_curve(model, x_t, true_img, mask, l1_likelihood, S=50):
	true_img = true_img.view(-1)
	mask = mask.view(-1)
	preds = []
	vars_aleatoric = []
	model.eval()
	model.apply(apply_dropout)
	with torch.no_grad():
		for _ in range(S):
			pred, logvar_aleatoric = model(x_t)
			preds.append(pred.cpu().view(-1))
			vars_aleatoric.append(logvar_aleatoric.exp().cpu().view(-1))
		del x_t
	model.train()
	probs_img = np.zeros(true_img.size(0))
	#CDF of gaussian mixture with S components
	for i in range(S):
		if l1_likelihood:
			dist = torch.distributions.Laplace(preds[i], vars_aleatoric[i].sqrt())
		else:
			dist = torch.distributions.Normal(preds[i], vars_aleatoric[i].sqrt())
		probs_img += dist.cdf(true_img).numpy() / S
		preds[i] = preds[i].numpy()
		vars_aleatoric[i] = vars_aleatoric[i].numpy()
	var_aleatoric = np.array(vars_aleatoric).mean(0)
	#Sharpness score, removing missing depth locations
	preds = np.array(preds)
	preds_mean = preds.mean(0)
	preds_var_model = np.square(preds).mean(0) - np.square(preds_mean) + 1e-8
	preds_var = preds_var_model + var_aleatoric
	sharpness = preds_var[mask.numpy()].mean()
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
	calibration = ((true_freq - pred_freq) ** 2 * 1.).sum()
	return pred_freq, true_freq, calibration, sharpness, preds_mean, preds_var_model, var_aleatoric

def test(model, x_test, y_test, idx, mask_test, epoch, dataset, l1_likelihood, trainset=False, saveplot=False):
	pred_freq, true_freq, calibration, sharpness, img_pred, var_model, var_aleatoric = pixel_wise_calibration_curve(model, x_test, y_test, mask_test, l1_likelihood)
	if dataset == 'make3d':
		C = 70.
		H, W = 168, 224
		im_ratio = float(H/W)
	sharpness *= (C**2) 
	x_test = x_test.view(3, H, W).permute(1, 2, 0).cpu().numpy()
	#BGR -> RGB
	_x_test = np.zeros_like(x_test)
	_x_test[:,:,0] = x_test[:,:,2]
	_x_test[:,:,1] = x_test[:,:,1]
	_x_test[:,:,2] = x_test[:,:,0]
	x_test = _x_test
	y_test = (C * y_test).view(H,W).numpy()
	img_pred = (C * img_pred).reshape(H, W)
	img_std_model = (C * np.sqrt(var_model)).reshape(H, W)
	img_std_aleatoric = C * np.sqrt(var_aleatoric).reshape(H, W)  
	mask_test = mask_test.view(H, W).numpy()
	RMSE = np.sqrt(np.square((img_pred - y_test)[mask_test]).mean())
	mean_log10_error = (np.abs(np.log10(img_pred) - np.log10(y_test))[mask_test]).mean()
	mean_abs_rel_error = (np.abs((img_pred - y_test)/y_test)[mask_test]).mean()
	_rgb_true = cv2.resize(x_test, None, fx=float(H/W), fy=float(W/H))
	_img_true = cv2.resize(y_test, None, fx=float(H/W), fy=float(W/H))
	_img_pred = cv2.resize(img_pred, None, fx=float(H/W), fy=float(W/H))
	_img_std_model = cv2.resize(img_std_model, None, fx=float(H/W), fy=float(W/H))
	_img_std_aleatoric = cv2.resize(img_std_aleatoric, None, fx=float(H/W), fy=float(W/H))
	if saveplot:
		print('Img idx: {} || Calibration score: {:4f} || Sharpness score: {:4f}'.format(idx, calibration, sharpness))
		vmin_,vmax_ = 1.75, 35.
		fig = plt.figure(1,figsize=(12, 2))
		ax1 = plt.subplot(171)
		im1 = ax1.imshow(_rgb_true)
		ax1.axis('off')
		ax2 = plt.subplot(172)
		im2 = ax2.imshow(_img_true,cmap='magma',vmin=vmin_,vmax=vmax_)
		ax2.axis('off')
		cb2 = fig.colorbar(im2, ax=ax2, fraction=0.046*im_ratio, pad=0.04)
		cb2.ax.tick_params(labelsize=5)
		cb2.ax.tick_params(size=0)
		ax3 = plt.subplot(173)
		im3 = ax3.imshow(_img_pred,cmap='magma',vmin=vmin_,vmax=vmax_)
		ax3.axis('off')
		cb3 = fig.colorbar(im3, ax=ax3, fraction=0.046*im_ratio, pad=0.04)
		cb3.ax.tick_params(labelsize=5)
		cb3.ax.tick_params(size=0)
		ax4 = plt.subplot(174)
		im4 = ax4.imshow(np.sqrt(_img_std_aleatoric**2 + _img_std_model**2),cmap='nipy_spectral')
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
			plt.savefig('results_{}_mcd_results_train_pred_{}.pdf'.format(dataset, idx), bbox_inches='tight', pad_inches=0.1)
		else:
			plt.savefig('results_{}_mcd_results_test_pred_{}.pdf'.format(dataset, idx), bbox_inches='tight', pad_inches=0.1)
		plt.close()
	return calibration, sharpness, RMSE, mean_log10_error, mean_abs_rel_error, img_pred.reshape(1, -1), y_test.reshape(1, -1), mask_test.reshape(1, -1)

def run_test_mcd(epoch, model, test_set, N_test, dataset, exp_name, l1_likelihood, mkdir=False):
	pred_list = []
	y_list = []
	mask_list = []
	if mkdir:
		import os
		new_dir = './results_{}'.format(exp_name)
		os.makedirs(new_dir, exist_ok=True)
		os.chdir(new_dir)
		n_save = N_test
	else:
		n_save = 15
	test_generator = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
	for k, (X_test, Y_test) in enumerate(test_generator):
		x_test = X_test[0].unsqueeze(0).cuda()
		y_test = Y_test[0].unsqueeze(0)
		if dataset == 'make3d':
			mask_test = (y_test < 1.0)
		if k < n_save:          
			cal, sh, RMSE, mean_log10_error, mean_abs_rel_error, pred, y, mask = test(model, x_test, y_test, k, mask_test, epoch, dataset, l1_likelihood, saveplot=True)
			np.savetxt('{}_{}_img_{}_calibration.txt'.format(dataset, exp_name, k), [cal])
			np.savetxt('{}_{}_img_{}_sharpness_mean.txt'.format(dataset, exp_name, k), [sh])
			np.savetxt('{}_{}_img_{}_rmse.txt'.format(dataset, exp_name, k), [RMSE])
			np.savetxt('{}_{}_img_{}_log10e.txt'.format(dataset, exp_name, k), [mean_log10_error])
			np.savetxt('{}_{}_img_{}_mean_abs_rel_error.txt'.format(dataset, exp_name, k), [mean_abs_rel_error])
		else:                                 
			cal, sh, RMSE, mean_log10_error, mean_abs_rel_error, pred, y, mask = test(model, x_test, y_test, k, mask_test, epoch, dataset, l1_likelihood, epoch)
		
		pred_list.append(pred)
		y_list.append(y)
		mask_list.append(mask)
		del x_test, y_test, mask_test
	preds = np.concatenate(pred_list, 0)
	Y = np.concatenate(y_list, 0)
	masks = np.concatenate(mask_list, 0)
	preds_masked = preds[masks]
	Y_masked = Y[masks]
	test_rmse = np.sqrt(np.square(preds_masked - Y_masked).mean())
	test_mare = np.abs((preds_masked - Y_masked)/Y_masked).mean()
	test_log10e = np.abs(np.log10(preds_masked) - np.log10(Y_masked)).mean()
	print('Epoch: {} || Test RMSE: {:.5f}'.format(epoch, test_rmse))
	np.savetxt('{}_{}_rmse_epoch_{}.txt'.format(dataset, exp_name, epoch), [test_rmse])
	np.savetxt('{}_{}_log10e_epoch_{}.txt'.format(dataset, exp_name, epoch), [test_mare])
	np.savetxt('{}_{}_mean_abs_rel_error_epoch_{}.txt'.format(dataset, exp_name, epoch), [test_log10e])
	print('Test rmse: {} || Test rel: {} || Test log10: {}'.format(test_rmse, test_mare, test_log10e))
