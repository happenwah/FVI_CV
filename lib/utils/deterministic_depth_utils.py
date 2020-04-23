import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def berhu_loss(y_t, pred, mask):
	res = y_t - pred
	c = 0.2 * torch.abs(res).max().detach()
	mask_l1 = (res <= c).type(torch.cuda.FloatTensor)
	mask_l2 = (res > c).type(torch.cuda.FloatTensor)
	loss = torch.abs(res) * mask_l1 + ((torch.pow(res, 2) + torch.pow(c, 2)) / (2 * c)) * mask_l2
	loss_masked = loss[mask]
	return loss_masked.mean()

def run_runtime_deterministic(model, test_set, exp_name):
	K = 100
	X_test = torch.from_numpy(test_set[0][0]).unsqueeze(0).cuda()
	model.eval()
	#First forward pass, ignore
	time_list = []
	start = torch.cuda.Event(enable_timing=True)
	end = torch.cuda.Event(enable_timing=True)
	for _ in range(K + 1):
		with torch.no_grad():
			start.record()
			output = model(X_test)
			end.record()
			torch.cuda.synchronize()
			time = start.elapsed_time(end)
			time_list.append(time)
	time_list = np.array(time_list)[1:]
	time_mean = time_list.mean()
	time_std = time_list.std()
	np.savetxt('{}_mean_runtime.txt'.format(exp_name), [time_mean])
	np.savetxt('{}_std_runtime.txt'.format(exp_name), [time_std])
	print(time_list)
	print('Inference time elapsed (s), mean: {} || std: {}'.format(time_mean, time_std))
	model.train()

def run_test_deterministic(epoch, model, test_set, N_test, dataset, exp_name):
	test_generator = torch.utils.data.DataLoader(test_set, batch_size=1)
	if dataset == 'make3d':
		C = 70.
		P = 168*224
	mask_list = []
	Y_list = []
	pred_mean_list = []
	model.eval()
	for X, Y in tqdm(test_generator):
		X = X.cuda(non_blocking=True)
		N = X.size(0)
		with torch.no_grad():
			output = model(X)
		output = C * output.cpu()
		Y *= C
		if dataset == 'make3d':
			mask = (Y < C)
			mask_list.append(mask.view(N, -1))
			Y_list.append(Y.view(N, -1))
			pred_mean_list.append(output.view(N, -1))
	model.train()
	pred_mean = torch.cat(pred_mean_list, 0)
	pred_Y = torch.cat(Y_list, 0)
	pred_mask = torch.cat(mask_list, 0)
	pred_mean_masked = pred_mean[pred_mask].numpy()
	pred_Y_masked = pred_Y[pred_mask].numpy()
	test_rmse = np.sqrt(np.square(pred_mean_masked - pred_Y_masked).mean(0))
	test_mare = np.abs((pred_mean_masked - pred_Y_masked)/pred_Y_masked).mean(0)
	test_log10e = np.abs(np.log10(pred_mean_masked) - np.log10(pred_Y_masked)).mean(0)
	np.savetxt('{}_{}_epoch_{}_rmse.txt'.format(dataset, exp_name, epoch), [test_rmse])
	np.savetxt('{}_{}_epoch_{}_log10e.txt'.format(dataset, exp_name, epoch), [test_log10e])
	np.savetxt('{}_{}_epoch_{}_mean_abs_rel_error.txt'.format(dataset, exp_name, epoch), [test_mare])
	print('Test rmse: {} || Test rel: {} || Test log10: {}'.format(test_rmse, test_mare, test_log10e))
