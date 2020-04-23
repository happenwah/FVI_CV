import torch
import numpy as np
import argparse
import os
from tqdm import tqdm

from lib.fvi_laplace_berhu_depth import FVI
from lib.utils.torch_utils import adjust_learning_rate
from lib.elbo_depth import weight_aleatoric
from lib.prior.priors import f_prior_BNN
from lib.utils.fvi_depth_utils import run_test_fvi_per_image, run_runtime_fvi

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--randseed', type=int, default=0)
parser.add_argument('--n_epochs', type=int, default=4000)
parser.add_argument('--dataset', type=str, default='make3d', help='interiornet or make3d')
parser.add_argument('--f_prior', type=str, default='cnn_gp', help='Type of GP prior: cnn_gp')
parser.add_argument('--likelihood', type=str, default='laplace', help='Choose from berhu or laplace')
parser.add_argument('--add_cov_diag', type=bool, default=True, help='Add Diagonal component to Q covariance')
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
	exp_name = '{}_fvi_{}_gp_bnn'.format(args.dataset, args.likelihood)

if args.likelihood == 'berhu':
	from lib.elbo_depth import fELBO_berhu_depth as fELBO
elif args.likelihood == 'laplace':
	from lib.elbo_depth import fELBO_laplace_depth as fELBO

params = {'batch_size': args.batch_size,
			'shuffle': True,
			'num_workers': 0}

N_test = test_set.__len__()

def train(num_epochs, FVI, optimizer):
	FVI.train()

	params = {'batch_size': args.batch_size,
			'shuffle': True,
			'num_workers': 0}

	N_train = training_set_full_size.__len__()
	training_generator = torch.utils.data.DataLoader(training_set_full_size, **params)

	for s in range(args.n_epochs + 1):
		train_ll = 0.
		FVI.train()
		if args.likelihood == 'berhu':
			c_test = torch.cuda.FloatTensor([0.])
		else:
			c_test = None
		for X, Y in tqdm(training_generator):
			x_t = X.to(device)
			y_t = Y.to(device)

			if args.dataset == 'make3d':
				mask = (y_t < 1.0)

			if mask.long().sum() > 1000.:
	
				f_samples, lik_logvar, q_mean, q_cov, prior_mean, prior_cov = FVI(x_t)

				if args.likelihood == 'laplace':
					loss = fELBO(mask, f_samples, y_t, lik_logvar, q_mean, q_cov, prior_mean, prior_cov, print_loss=False)
				elif args.likelihood == 'berhu':
					loss, c = fELBO(mask, f_samples, y_t, lik_logvar, q_mean, q_cov, prior_mean, prior_cov, print_loss=False)
					
				optimizer.zero_grad()
				loss.backward()
				if args.likelihood == 'berhu':
					if c > c_test:
						c_test = c
						np.savetxt('{}_{}_c_test.txt'.format(args.dataset, exp_name), [c_test.item()])
				train_ll += loss.item() * (x_t.size(0))
				torch.nn.utils.clip_grad_norm_(FVI.q.parameters(), 1.) 
				optimizer.step()
				del x_t, y_t, mask, f_samples, lik_logvar, q_mean, q_cov, prior_mean, prior_cov
			else:
				continue
		train_ll /= N_train
		np.savetxt('{}_{}_epoch_{}_average_train_ll.txt'.format(args.dataset, exp_name, s), [train_ll])

		if s % args.save_results == 0 or s==args.n_epochs:
			run_test_fvi_per_image(s, FVI, test_set, N_test, args.dataset, exp_name, args.likelihood, c_threshold=c_test)
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
		load_dir_model = os.path.join(args.base_dir, 'FVI/model_{}_{}.bin'.format(args.dataset, exp_name))
		#load_dir_optimizer = os.path.join(args.base_dir, 'SIVI/fivi_regression_v2_img/optimizer_{}_{}.bin'.format(args.dataset, exp_name))
		FVI.load_state_dict(torch.load(load_dir_model))
		#optimizer.load_state_dict(torch.load(load_dir_optimizer))
		print('Loading FVI {} model..'.format(args.likelihood))
	
	if args.training_mode:
		train(args.n_epochs, FVI, optimizer)
	if args.test_mode:
		print('FVI {} on test mode'.format(args.likelihood))
		load_dir_model = os.path.join(args.base_dir, 'FVI/models_test/model_fvi_{}_test.bin'.format(args.likelihood))
		FVI.load_state_dict(torch.load(load_dir_model))
		test_generator = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
		if args.likelihood == 'berhu':
			load_dir_c_test = os.path.join(args.base_dir, 'FVI/models_test/c_test.txt') 
			c_test = torch.FloatTensor(np.loadtxt(load_dir_c_test))
		else:
			c_test = None
		run_test_fvi_per_image(-1, FVI, test_set, N_test, args.dataset, exp_name, args.likelihood, c_threshold=c_test, mkdir=True)
	if args.test_runtime_mode:
		load_dir_model = os.path.join(args.base_dir, 'FVI/models_test/model_fvi_{}_test.bin'.format(args.likelihood))
		FVI.load_state_dict(torch.load(load_dir_model))
		if args.likelihood == 'berhu':
			load_dir_c_test = os.path.join(args.base_dir, 'FVI/models_test/c_test.txt')
			c_test = torch.FloatTensor(np.loadtxt(load_dir_c_test))
		else:
			c_test = None
		run_runtime_fvi(FVI, test_set, args.likelihood, exp_name, c_threshold=c_test)
