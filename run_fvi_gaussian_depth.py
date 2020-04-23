import torch
import numpy as np
import argparse
import os
from tqdm import tqdm

from lib.fvi_gaussian_depth import FVI
from lib.utils.torch_utils import adjust_learning_rate
from lib.utils.fvi_depth_utils import run_test_fvi_per_image, run_runtime_fvi

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--randseed', type=int, default=0)
parser.add_argument('--n_epochs', type=int, default=4000)
parser.add_argument('--dataset', type=str, default='make3d', help='interiornet or make3d')
parser.add_argument('--add_cov_diag', type=bool, default=True, help='Add Diagonal component to Q covariance')
parser.add_argument('--f_prior', type=str, default='cnn_gp', help='Type of GP prior: cnn_gp')
parser.add_argument('--x_inducing_var', type=float, default=0.1, help='Pixel-wise variance for inducing inputs')
parser.add_argument('--n_inducing', type=int, default=1, help='No. of inducing inputs, <= batch_size')
parser.add_argument('--save_results', type=int, default=500, help='save results every few epochs')
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

params = {'batch_size': args.batch_size,
          'shuffle': True,
          'num_workers': 0}

if args.dataset == 'make3d':
	from lib.utils.make3d_loader import Make3dDataset
	H, W = 168, 224
	dir_train = os.path.join(args.base_dir, 'datasets/make3d/make3d_train.npz')
	dir_test = os.path.join(args.base_dir, 'datasets/make3d/make3d_test.npz')
	print('Make3d train data dir: ', dir_train)
	training_set_full_size = Make3dDataset(train=True, dir=dir_train) 
	test_set = Make3dDataset(train=False, dir=dir_test)

if args.f_prior == 'cnn_gp':
	exp_name = '{}_fvi_gaussian_gp_bnn'.format(args.dataset)

from lib.elbo_depth import fELBO_gaussian_depth as fELBO

N_test = test_set.__len__()
test_generator = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

def train(num_epochs, FVI):
	FVI.train()

	params = {'batch_size': args.batch_size,
			'shuffle': True, 'num_workers': 0}
	N_train = training_set_full_size.__len__()
	training_generator = torch.utils.data.DataLoader(training_set_full_size, **params)

	for s in range(args.n_epochs + 1):
		train_ll = 0.
		FVI.train()
		for X, Y in tqdm(training_generator):
			x_t = X.to(device)
			y_t = Y.to(device)

			if args.dataset == 'make3d':
				mask = (y_t < 1.0)

			if mask.long().sum() > 1000.:

				lik_logvar, q_mean, q_cov, prior_mean, prior_cov = FVI(x_t)

				loss_minus = fELBO(mask, y_t, lik_logvar, q_mean, 
						q_cov, prior_mean, prior_cov, print_loss=True)

				optimizer.zero_grad()
				loss = - loss_minus
				loss.backward()
				train_ll += -loss.item() * (x_t.size(0))
				torch.nn.utils.clip_grad_norm_(FVI.q.parameters(), 1.) 
				optimizer.step()
			else:
				continue
		train_ll /= N_train
		np.savetxt('{}_{}_epoch_{}_average_train_ll.txt'.format(args.dataset, exp_name, s), [train_ll])

		if s % args.save_results == 0:
			run_test_fvi_per_image(s, FVI, test_set, N_test, args.dataset, exp_name, 'gaussian')
			torch.save(FVI.state_dict(), 'model_{}_{}.bin'.format(args.dataset, exp_name))
			torch.save(optimizer.state_dict(), 'optimizer_{}_{}.bin'.format(args.dataset, exp_name)) 

if __name__ == '__main__':
	device = torch.device("cuda")

	keys = ('device', 'x_inducing_var', 'f_prior', 'n_inducing', 'add_cov_diag')
	values = (device, args.x_inducing_var, args.f_prior, args.n_inducing, args.add_cov_diag)
	fvi_args = dict(zip(keys, values))

	FVI = FVI(x_size=(H, W), **fvi_args).to(device)
	optimizer = torch.optim.AdamW(FVI.parameters(), lr=args.lr, weight_decay=1e-4)

	if args.load:
		model_load_dir = os.path.join(args.base_dir, 'FVI/model_{}_{}.bin'.format(args.dataset, exp_name))
		optimizer_load_dir = os.path.join(args.base_dir, 'FVI/optimizer_{}_{}.bin'.format(args.dataset, exp_name)) 
		FVI.load_state_dict(torch.load(model_load_dir))
		optimizer.load_state_dict(torch.load(optimizer_load_dir))
		print('Loading FVI gaussian model..')

	if args.training_mode:
		print('Training FVI gaussian for {} epochs'.format(args.n_epochs))
		train(args.n_epochs, FVI)
	if args.test_mode:
		print('FVI gaussian on test mode')
		load_dir_model = os.path.join(args.base_dir, 'FVI/models_test/model_{}_fvi_gaussian_test.bin'.format(args.dataset))
		FVI.load_state_dict(torch.load(load_dir_model))
		run_test_fvi_per_image(-1, FVI, test_set, N_test, args.dataset, exp_name, 'gaussian', mkdir=True)
	if args.test_runtime_mode:
		load_dir_model = os.path.join(args.base_dir, 'FVI/models_test/model_{}_fvi_gaussian_test.bin'.format(args.dataset))
		FVI.load_state_dict(torch.load(load_dir_model))
		run_runtime_fvi(FVI, test_set, 'gaussian', exp_name)
