import torch
import numpy as np
from tqdm import tqdm
import argparse
import os

from lib.elbo_depth import gaussian_log_prob, laplacian_log_prob
from lib.utils.torch_utils import adjust_learning_rate
from lib.utils.mcd_depth_utils import run_test_mcd, run_runtime_mcd_depth

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--p', type=float, default=0.2)
parser.add_argument('--dataset', type=str, default='make3d')
parser.add_argument('--l1_likelihood', type=bool, default=True, help='Use laplacian likelihood')
parser.add_argument('--randseed', type=int, default=0)
parser.add_argument('--n_epochs', type=int, default=4000)
parser.add_argument('--save_results', type=int, default=500, help='save results every x epochs')
parser.add_argument('--base_dir', type=str, default='/rdsgpfs/general/user/etc15/home/', help='directory in which datasets are contained')
parser.add_argument('--training_mode', action='store_true')
parser.add_argument('--load', action='store_true', help='Load model for resuming training, default: False')
parser.add_argument('--test_mode', action='store_true')
parser.add_argument('--test_runtime_mode', action='store_true')

args = parser.parse_args()

if args.training_mode:
	torch.backends.cudnn.benchmark = True
elif args.test_runtime_mode:
	torch.backends.cudnn.deterministic = True
np.random.seed(args.randseed)
torch.manual_seed(args.randseed)
torch.cuda.manual_seed(args.randseed)

if args.l1_likelihood:
	print('Using laplacian likelihood')

from lib.variational_dist import Q_FCDenseNet103_MCDropout
Q = Q_FCDenseNet103_MCDropout

if args.dataset == 'make3d':
	from lib.utils.make3d_loader import Make3dDataset
	H, W = 168, 224
	dir_train = os.path.join(args.base_dir, 'datasets/make3d/make3d_train.npz')
	dir_test = os.path.join(args.base_dir, 'datasets/make3d/make3d_test.npz')
	training_set_full_size = Make3dDataset(train=True, dir=dir_train)
	test_set = Make3dDataset(train=False, dir=dir_test)

exp_name = '{}_mc_dropout_p={}'.format(args.dataset, str(args.p))

N_test = test_set.__len__()

def train(num_epochs, optimizer):
	model.train()
	train_loss = float('inf')

	params = {'batch_size': args.batch_size,
			'shuffle': True,
			'num_workers': 0}

	N_train = training_set_full_size.__len__()
	training_generator = torch.utils.data.DataLoader(training_set_full_size, **params)

	for s in range(args.n_epochs + 1):
		model.train()
		pred_list = []
		target_list = []
		mask_list = []
		for X, Y in tqdm(training_generator):
			x_t = X.to(device)
			y_t = Y.to(device)                                                                                                                                              

			if args.dataset == 'make3d':
				mask = (y_t < 1.)
			if mask.long().sum() > 1000.:
				optimizer.zero_grad()                                                              
				mean, logvar_aleatoric = model(x_t)
				N = x_t.size(0)

			
				if args.l1_likelihood:
					loss_ = - laplacian_log_prob(y_t.view(N, -1), mean.view(N, -1), logvar_aleatoric.view(N, -1))[mask.view(N, -1)]
					loss = loss_.mean(0)
				else:
					loss_ = - gaussian_log_prob(y_t.view(N, -1), mean.view(N, -1), logvar_aleatoric.view(N, -1))[mask.view(N, -1)]
					loss = loss_.mean(0)

				loss.backward()
				pred_list.append(mean.view(N, -1).cpu().detach())
				target_list.append(y_t.view(N, -1).cpu().detach())
				mask_list.append(mask.view(N, -1).cpu().detach())
				torch.nn.utils.clip_grad_norm_(model.parameters(), 1.) 
				optimizer.step()
			else:
				continue
			del y_t, mean, mask, logvar_aleatoric
		train_masks = torch.cat(mask_list, 0)
		train_preds = torch.cat(pred_list, 0)[train_masks]
		train_targets = torch.cat(target_list, 0)[train_masks]
		train_mse = torch.pow(train_preds - train_targets, 2).mean().item()
		if args.dataset=='make3d':
			train_rmse = 70. * (train_mse ** 0.5)
			print('Epoch: {} || Train RMSE: {:.5f}'.format(s, train_rmse))
		np.savetxt('{}_{}_epoch_{}_train_rmse.txt'.format(args.dataset, exp_name, s), [train_rmse])
		del train_preds, train_targets, train_masks

		if s % args.save_results == 0:
			torch.save(model.state_dict(), 'model_{}_{}.bin'.format(args.dataset, exp_name))
			torch.save(optimizer.state_dict(), 'optimizer_{}_{}.bin'.format(args.dataset, exp_name))
			run_test_mcd(s, model, test_set, N_test, args.dataset, exp_name, args.l1_likelihood)
if __name__ == '__main__':
	device = torch.device("cuda")
	model = Q(p=0.2).to(device)

	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

	if args.load:
		load_dir_model = os.path.join(args.base_dir, 'FVI/model_{}_{}.bin'.format(args.dataset, exp_name))
		load_dir_optimizer = os.path.join(args.base_dir, 'FVI/optimizer_{}_{}.bin'.format(args.dataset, exp_name)) 
		model.load_state_dict(torch.load(load_dir_model))
		optimizer.load_state_dict(torch.load(load_dir_optimizer)) 
		print('Loading MC Dropout model..')

	if args.training_mode:
		train(args.n_epochs, optimizer)
	if args.test_mode:
		load_dir_model = os.path.join(args.base_dir, 'FVI/models_test/model_mcd_test.bin')
		model.load_state_dict(torch.load(load_dir_model))
		run_test_mcd(-1, model, test_set, N_test, args.dataset, exp_name, args.l1_likelihood, mkdir=True)
	if args.test_runtime_mode:
		load_dir_model = os.path.join(args.base_dir, 'FVI/models_test/model_mcd_test.bin')
		model.load_state_dict(torch.load(load_dir_model))
		run_runtime_mcd_depth(model, test_set, exp_name)

